#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os
import time
import urllib.request
from typing import List, Tuple

import torch

from iaxl.utils.logger import setup_root_logger

setup_root_logger(show_pid_tid=False)


_env_parser = argparse.ArgumentParser(add_help=False)
_env_parser.add_argument(
    "--quant", type=str, default=os.environ.get("IAXL_KV_LOSSY_TRUNC", "0")
)
_env_parser.add_argument(
    "--data-shuffle",
    type=int,
    choices=(0, 1),
    default=int(os.environ.get("IAXL_KV_DATA_SHUFFLE", "0")),
)
_env_args, _ = _env_parser.parse_known_args()
os.environ["IAXL_KV_LOSSY_TRUNC"] = str(_env_args.quant)
os.environ["IAXL_KV_DATA_SHUFFLE"] = str(_env_args.data_shuffle)

from iaxl.kvstore import KVStore, get_accelerator_device

DEFAULT_KV_CACHE_SHAPE = (2, 1024, 16, 4, 128)


DEFAULT_METRICS_URL = "http://127.0.0.1:18800/v1/cache/metrics"
DEFAULT_KV_DATA_DIR = "/_data/kvstore_benchmark"
DEFAULT_MODEL_SEQ_LEN = 16384
SEED = 42
BLOCK_DIM = 1
FP8_MAX = 448.0
INT4_QMAX = 7.0
INT4_GROUP_SIZE = 128

# int4 is stored as two 4-bit values packed into one uint8.
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp8_e4m3": torch.float8_e4m3fn,
    "int4": torch.uint8,
}


def add_env_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quant",
        type=str,
        default=os.environ.get("IAXL_KV_LOSSY_TRUNC", "0"),
        help="Quantization: 'auto', 0 (off), or N LSB bits to truncate "
        "(env IAXL_KV_LOSSY_TRUNC).",
    )
    parser.add_argument(
        "--data-shuffle",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("IAXL_KV_DATA_SHUFFLE", "0")),
        help="Enable data shuffle before compression (env IAXL_KV_DATA_SHUFFLE).",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark KVStore PUT/GET with generated or file-backed data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kv-cache-shape",
        "--shape",
        type=int,
        nargs=5,
        default=DEFAULT_KV_CACHE_SHAPE,
        metavar=("KV", "CACHE_BLOCKS", "TOKENS", "KV_HEADS", "HEAD_DIM"),
        help="Shape of one KV cache layer. KV_HEADS is 8 for TP=1, 1 for TP=8.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_MAP),
        default="bf16",
        help="KV cache tensor dtype.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="Number of KV cache layers to allocate, mimicking a real model. "
        "PUT/GET and their waits are issued layer by layer like vLLM. "
        "Each layer holds a full --kv-cache-shape tensor, so GPU memory "
        "scales with this value.",
    )
    parser.add_argument(
        "--data-source",
        choices=("mock", "model", "file"),
        default=None,
        help="Where the KV cache content comes from: a real transformer prefill, "
        "synthetic mock data, or raw file bytes. Defaults to 'model', or to "
        "'file' when --data-file is set.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Optional test-data file whose bytes fill the KV cache (cycled "
        "to fit). If omitted, deterministic mock data is generated.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL"),
        help="Hugging Face model ID or local path used by --data-source model "
        "(env MODEL).",
    )
    parser.add_argument(
        "--model-seq-len",
        type=int,
        default=DEFAULT_MODEL_SEQ_LEN,
        help="Prefill length of one forward pass when generating model data. "
        "Must be a multiple of the block TOKENS dimension.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Text file tokenized (and cycled) into the prefill prompt. "
        "If omitted, deterministic random token IDs are used.",
    )
    parser.add_argument(
        "--kv-data-dir",
        default=DEFAULT_KV_DATA_DIR,
        help="Directory holding generated model KV cache files, named "
        "<model>_<dtype>.pt and reused whenever they match the requested shape.",
    )
    parser.add_argument(
        "--metrics-url",
        default=DEFAULT_METRICS_URL,
        help="KVStore REST endpoint for compression throughput metrics.",
    )
    add_env_arguments(parser)
    args = parser.parse_args()
    shape = tuple(args.kv_cache_shape)
    if shape[0] != 2:
        parser.error("the first --kv-cache-shape dimension must be 2 (key/value)")
    if any(dim <= 0 for dim in shape):
        parser.error("all --kv-cache-shape dimensions must be greater than 0")
    if args.num_layers <= 0:
        parser.error("--num-layers must be greater than 0")
    if args.dtype == "int4" and shape[-1] % 2:
        parser.error("--dtype int4 requires an even HEAD_DIM")
    if args.data_source is None:
        use_file = args.data_file is not None and args.data_file != "random"
        args.data_source = "file" if use_file else "model"
    if args.data_source == "file" and not args.data_file:
        parser.error("--data-source file requires --data-file")
    if args.data_source == "model":
        if not args.model:
            parser.error("--data-source model requires --model or the MODEL env var")
        if args.model_seq_len <= 0 or args.model_seq_len % shape[2]:
            parser.error("--model-seq-len must be a positive multiple of TOKENS")
    args.kv_cache_shape = shape
    return args


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def make_block_hashes(num_blocks: int) -> List[str]:
    return [str(i) for i in range(num_blocks)]


def storage_shape(shape: Tuple[int, ...], dtype_name: str) -> Tuple[int, ...]:
    if dtype_name == "int4":
        return shape[:-1] + (shape[-1] // 2,)
    return shape


def quantize_kv(values: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """Quantize a float KV tensor [2, ...] the way vLLM does."""
    if dtype_name == "bf16":
        return values.to(torch.bfloat16).contiguous()

    values = values.float()
    if dtype_name == "fp8_e4m3":
        # vLLM stores one per-tensor k_scale/v_scale per layer: scale = amax / 448.
        reduce_dims = tuple(range(1, values.ndim))
        scale = values.abs().amax(dim=reduce_dims, keepdim=True) / FP8_MAX
        scaled = (values / scale.clamp_min(1e-12)).clamp_(-FP8_MAX, FP8_MAX)
        return scaled.to(torch.float8_e4m3fn).contiguous()

    # vLLM int4: symmetric group-wise uint4b8, scale = amax / 7, low nibble first.
    head_dim = values.shape[-1]
    group = INT4_GROUP_SIZE if head_dim % INT4_GROUP_SIZE == 0 else head_dim
    grouped = values.unflatten(-1, (-1, group))
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / INT4_QMAX).clamp_min(1e-6)
    codes = (grouped / scale).round_().clamp_(-8, 7).add_(8)
    codes = codes.to(torch.uint8).flatten(-2)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def generate_mock_kv_cache(shape: Tuple[int, ...], dtype_name: str) -> torch.Tensor:
    kv_count, cache_blocks, block_tokens, kv_heads, head_dim = shape
    latent_dim = min(32, max(8, head_dim // 8))
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    block_context = torch.randn(
        (cache_blocks, 1, kv_heads, latent_dim),
        device="cuda",
        generator=generator,
    )
    token_innovations = torch.randn(
        (cache_blocks, block_tokens, kv_heads, latent_dim),
        device="cuda",
        generator=generator,
    )
    block_drift = torch.randn(
        (cache_blocks, 1, kv_heads, latent_dim),
        device="cuda",
        generator=generator,
    )
    token_positions = torch.linspace(-1.0, 1.0, block_tokens, device="cuda").view(
        1, block_tokens, 1, 1
    )
    hidden_states = (
        0.65 * block_context
        + 0.70 * token_innovations
        + 0.15 * token_positions * block_drift
    )

    projections = torch.randn(
        (kv_count, kv_heads, latent_dim, head_dim),
        device="cuda",
        generator=generator,
    ) / math.sqrt(latent_dim)
    kv_cache = torch.einsum("bthr,khrd->kbthd", hidden_states, projections)

    channel_scales = (
        torch.randn(
            (kv_count, 1, 1, kv_heads, head_dim),
            device="cuda",
            generator=generator,
        )
        .mul_(0.25)
        .exp_()
    )
    kv_cache.mul_(channel_scales)
    return quantize_kv(kv_cache, dtype_name)


def _layer_keys_values(past_key_values, index: int):
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        return layers[index].keys, layers[index].values
    return past_key_values.key_cache[index], past_key_values.value_cache[index]


def _to_blocks(
    tensor: torch.Tensor, block_tokens: int, kv_heads: int, head_dim: int
) -> torch.Tensor:
    """[1, heads, seq, head_dim] -> [blocks, block_tokens, kv_heads, head_dim]."""
    tokens = tensor[0].transpose(0, 1)
    if tokens.shape[1] < kv_heads:
        repeats = -(-kv_heads // tokens.shape[1])
        tokens = tokens.repeat(1, repeats, 1)
    return tokens[:, :kv_heads, :head_dim].reshape(-1, block_tokens, kv_heads, head_dim)


def _make_token_ids(
    tokenizer, vocab_size: int, total_tokens: int, prompt_file
) -> torch.Tensor:
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8", errors="replace") as fp:
            text = fp.read()
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"prompt file {prompt_file!r} produced no tokens")
        ids = (ids * (-(-total_tokens // len(ids))))[:total_tokens]
        print(f"[model] tokenized {prompt_file}, cycled to {total_tokens} tokens")
        return torch.tensor(ids, dtype=torch.long)

    generator = torch.Generator().manual_seed(SEED)
    print(f"[model] using {total_tokens} deterministic random token IDs")
    return torch.randint(
        0, vocab_size, (total_tokens,), generator=generator, dtype=torch.long
    )


def generate_model_kv_cache(
    shape: Tuple[int, ...], dtype_name: str, args: argparse.Namespace
) -> List[torch.Tensor]:
    """Prefill a real transformer and return one CPU KV tensor per model layer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, cache_blocks, block_tokens, kv_heads, head_dim = shape
    total_tokens = cache_blocks * block_tokens

    print(f"[model] loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()

    token_ids = _make_token_ids(
        tokenizer, model.config.vocab_size, total_tokens, args.prompt_file
    )
    num_model_layers = min(model.config.num_hidden_layers, args.num_layers)
    parts: List[List[torch.Tensor]] = [[] for _ in range(num_model_layers)]
    backbone = getattr(model, "model", model)  # skip the LM head, only KV is needed

    for start in range(0, total_tokens, args.model_seq_len):
        chunk = token_ids[start : start + args.model_seq_len].unsqueeze(0).to("cuda")
        with torch.no_grad():
            past_key_values = backbone(input_ids=chunk, use_cache=True).past_key_values
        for layer in range(num_model_layers):
            keys, values = _layer_keys_values(past_key_values, layer)
            if keys.shape[-1] < head_dim:
                raise ValueError(
                    f"model head_dim {keys.shape[-1]} is smaller than the "
                    f"requested HEAD_DIM {head_dim}"
                )
            kv = torch.stack(
                (
                    _to_blocks(keys, block_tokens, kv_heads, head_dim),
                    _to_blocks(values, block_tokens, kv_heads, head_dim),
                ),
                dim=0,
            )
            parts[layer].append(kv.cpu())
        del past_key_values
        torch.cuda.empty_cache()
        done = min(start + args.model_seq_len, total_tokens)
        print(f"[model] prefilled {done}/{total_tokens} tokens")

    del model
    torch.cuda.empty_cache()

    # Quantize whole layers so fp8 gets one k_scale/v_scale per layer, like vLLM.
    layers = []
    for part in parts:
        merged = torch.cat(part, dim=1)[:, :cache_blocks].to("cuda")
        part.clear()
        layers.append(quantize_kv(merged, dtype_name).cpu())
        del merged
    torch.cuda.empty_cache()
    return layers


def model_cache_path(args: argparse.Namespace, dtype_name: str) -> str:
    model_tag = os.path.basename(args.model.rstrip("/"))
    return os.path.join(args.kv_data_dir, f"{model_tag}_{dtype_name}.pt")


def load_or_generate_model_kv_cache(
    args: argparse.Namespace, shape: Tuple[int, ...], dtype_name: str
) -> List[torch.Tensor]:
    path = model_cache_path(args, dtype_name)
    expected = storage_shape(shape, dtype_name)
    if os.path.exists(path):
        layers = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(layers[0].shape) == expected:
            print(f"[model] reusing {len(layers)} cached KV layers from {path}")
            return layers
        print(
            f"[model] {path} holds shape {tuple(layers[0].shape)}, "
            f"regenerating for {expected}"
        )
        del layers

    layers = generate_model_kv_cache(shape, dtype_name, args)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(layers, path)
    print(f"[model] saved {len(layers)} KV layers to {path}")
    return layers


def load_kv_cache_from_file(
    path: str, shape: Tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    with open(path, "rb") as fp:
        data = fp.read()
    if not data:
        raise ValueError(f"test-data file {path!r} is empty")

    total_bytes = math.prod(shape) * dtype.itemsize
    repeats = -(-total_bytes // len(data))
    raw = bytearray(data) * repeats
    del raw[total_bytes:]

    flat = torch.frombuffer(raw, dtype=torch.uint8).view(dtype)
    print(f"[data] loaded {path} ({len(data)} bytes, cycled to fill cache)")
    return flat.reshape(shape).to("cuda")


def print_result(name: str, elapsed: float, num_blocks: int, num_bytes: int) -> None:
    print(
        f"{name:>4}: {elapsed:.3f} s, "
        f"{num_blocks / elapsed:,.2f} blocks/s, "
        f"{num_bytes / elapsed / (1024**3):.3f} GiB/s"
    )


def _rest_get_json(url: str, query: str = "") -> dict:
    if query:
        url = f"{url}?{query}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def cache_metrics_rest(url: str, query: str = "") -> dict:
    return _rest_get_json(url, query)


def cache_status_rest(metrics_url: str) -> dict:
    status_url = metrics_url.rsplit("/", 1)[0] + "/status"
    return _rest_get_json(status_url)


def run_benchmark(args: argparse.Namespace) -> bool:
    if get_accelerator_device() != "cuda":
        print("No CUDA device available; KVStore benchmark cannot run.")
        return False

    shape: Tuple[int, ...] = args.kv_cache_shape
    store_shape = storage_shape(shape, args.dtype)
    dtype = DTYPE_MAP[args.dtype]
    num_layers = args.num_layers
    cache_blocks = shape[BLOCK_DIM]

    block_indices = list(range(0, cache_blocks, 2))
    num_blocks = len(block_indices)
    bytes_per_block = math.prod(store_shape) // cache_blocks * dtype.itemsize

    total_blocks = num_blocks * num_layers
    total_bytes = bytes_per_block * num_blocks * num_layers

    layer_names = [str(i) for i in range(num_layers)]

    print("=" * 80)
    print("KVStore Benchmark")
    print("=" * 80)
    print(f"KV cache shape: {list(shape)} (one layer)")
    print(f"Layers:         {num_layers}")
    print(f"Blocks tested:  {num_blocks} (even indices, strided) x {num_layers} layers")
    print(f"Dtype:          {args.dtype}")
    if store_shape != shape:
        print(f"Stored shape:   {list(store_shape)} ({dtype}, two int4 per byte)")
    print(f"Data source:    {args.data_source}")
    print(f"Transfer size:  {format_bytes(total_bytes)}")

    if args.data_source == "model":
        print(f"\nBuilding KV cache from {args.model}...")
        layers = load_or_generate_model_kv_cache(args, shape, args.dtype)
        kv_caches = {
            name: layers[i % len(layers)].to("cuda")
            for i, name in enumerate(layer_names)
        }
        del layers
    else:
        if args.data_source == "file":
            print(f"\nLoading KV cache from {args.data_file}...")
            base = load_kv_cache_from_file(args.data_file, store_shape, dtype)
        else:
            print("\nGenerating mock KV cache data...")
            base = generate_mock_kv_cache(shape, args.dtype)
        kv_caches = {name: base.clone() for name in layer_names}
        del base

    index = torch.tensor(
        [block_indices[0], block_indices[num_blocks // 2], block_indices[-1]],
        device="cuda",
    )
    expected = {
        name: tensor.index_select(BLOCK_DIM, index).clone()
        for name, tensor in kv_caches.items()
    }

    block_hashes = make_block_hashes(num_blocks)

    kvstore = KVStore(
        model_name="kvstore_benchmark",
        kv_caches=kv_caches,
        block_dim=BLOCK_DIM,
    )

    cache_metrics_rest(args.metrics_url, "enable=1&reset=1")

    print("\nWarming up...")
    warmup_hashes = [f"warmup_{h}" for h in block_hashes]
    warmup_put = kvstore.put(block_indices, warmup_hashes, description="warmup PUT")
    if not kvstore.put_wait(warmup_put):
        raise RuntimeError("warm-up PUT did not complete")
    warmup_get = kvstore.get(block_indices, warmup_hashes, description="warmup GET")
    if not kvstore.get_wait(warmup_get):
        raise RuntimeError("warm-up GET did not complete")
    torch.cuda.synchronize()

    cache_metrics_rest(args.metrics_url, "reset=1")

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.cuda.nvtx.range("benchmark PUT"):
        put_tasks = {}
        for name in layer_names:
            put_tasks.update(
                kvstore.put(
                    block_indices,
                    block_hashes,
                    layer_names=[name],
                    description="benchmark PUT",
                )
            )
        if not kvstore.put_wait(put_tasks):
            raise RuntimeError("PUT did not complete")
    put_time = time.perf_counter() - start

    for tensor in kv_caches.values():
        tensor.zero_()
    torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.cuda.nvtx.range("benchmark GET"):
        get_tasks = kvstore.get(
            block_indices, block_hashes, layer_names=None, description="benchmark GET"
        )
        for name in layer_names:
            if not kvstore.get_wait(get_tasks, layer_names=[name]):
                raise RuntimeError("GET did not complete")
    get_time = time.perf_counter() - start

    verified = all(
        torch.equal(tensor.index_select(BLOCK_DIM, index), expected[name])
        for name, tensor in kv_caches.items()
    )

    print("\nResults")
    print("-" * 80)
    print_result("PUT", put_time, total_blocks, total_bytes)
    print_result("GET", get_time, total_blocks, total_bytes)
    print(f"Verification: {'passed' if verified else 'FAILED'}")

    status = cache_status_rest(args.metrics_url)
    print(
        f"Compression ratio: {status['compression_ratio (unzip/zip, higher=better)']:.3f}x"
    )
    print(
        f"Native cache size: {format_bytes(status['current_bytes'])} "
        f"({status['cache_entries']} chunks in {status['group_count']} groups)"
    )
    print(
        f"Cache hits/misses/puts: {status['hits']}/{status['misses']}/{status['puts']}"
    )
    print(
        f"Compressed/uncompressed bytes: {format_bytes(status['total_zip_bytes'])} / "
        f"{format_bytes(status['total_unzip_bytes'])}"
    )

    metrics = cache_metrics_rest(args.metrics_url)
    print(
        f"Compress throughput:   {metrics['compress_gbps']:.3f} GB/s "
        f"({format_bytes(metrics['compress_bytes'])} in {metrics['compress_ns'] / 1e6:.1f} ms)"
    )
    print(
        f"Decompress throughput: {metrics['decompress_gbps']:.3f} GB/s "
        f"({format_bytes(metrics['decompress_bytes'])} in {metrics['decompress_ns'] / 1e6:.1f} ms)"
    )
    return verified


def main() -> int:
    args = parse_args()
    try:
        return 0 if run_benchmark(args) else 1
    except Exception as error:
        print(f"\nBenchmark FAILED: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
