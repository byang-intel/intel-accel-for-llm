#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""用 Hunyuan INT4 MLA KV cache dump 驱动的 KVStore PUT/GET 测试。

一、数据来源
----------
dump 目录(格式见同目录下 HUNYUAN_INT4_KV_CACHE_DUMP_FORMAT.md)中每层一个 ``layer_NNN.pt``，
保存量化后 KV cache 的**原始字节**(torch.uint8)：

- MLA KV：``[tokens, 448]``，每行一个 token(78 层都有)；
- indexer cache：``[N]`` 一维字节流，FP8 key + FP32 scale，按 64 token 一页
  (8192 B key + 256 B scale)排布(仅 21 层有)。

脚本把每个 cache 看成 ``[num_blocks, block_size * 448]`` 字节(indexer 也按同样
的字节数切块，尾部不足一块的丢弃)，用 block_dim=0 注册成 KVStore 的一个 layer，
与 kvshrink connector 的做法一致。然后对所有 block 做 PUT，把 GPU tensor 清零后
GET 回来，逐字节与原始数据比对。

二、--mla-shuffle：block 内字节重排
--------------------------------
kv_zip 以 block 为单位做 DEFLATE，所以任何预处理都必须限制在 block 内部，
才能在不改 C++ 代码的前提下验证效果；GET 回来的仍是重排后的字节，由 Python 侧
的逆变换还原(加载时已对每层做 ``unshuffle(shuffle(x)) == x`` 校验)。这是纯粹的
**字节置换，无损**：不改变任何字节的值，也不增减字节数。

shuffle 前(dump 原始布局，一个 block = block_size 个 token 行，token-major)：

    token 0 : [ int4 nibbles 256 B | scale bf16 x32 = 64 B | rope bf16 x64 = 128 B ]
    token 1 : [ int4 nibbles 256 B | scale bf16 x32 = 64 B | rope bf16 x64 = 128 B ]
    ...
    token 63: [ int4 nibbles 256 B | scale bf16 x32 = 64 B | rope bf16 x64 = 128 B ]

    其中每个 bf16(1 符号 + 8 指数 + 7 尾数)为小端两个字节：
        lo = 指数最低 1 位 + 7 位尾数(接近随机)，hi = 符号 + 指数高 7 位(分布集中)

shuffle 后(``planes``，五个"字节平面"首尾相接，总长不变)：

    [ int4 平面      : 所有 token 的 256 B nibble，共 bs*256 B ]
    [ scale hi 平面  : 所有 token 的 32 个 scale 高字节，共 bs*32 B ]
    [ scale lo 平面  : 所有 token 的 32 个 scale 低字节，共 bs*32 B ]
    [ rope  hi 平面  : 所有 token 的 64 个 rope  高字节，共 bs*64 B ]
    [ rope  lo 平面  : 所有 token 的 64 个 rope  低字节，共 bs*64 B ]

``planes_cm`` 与 ``planes`` 相同，但 bf16 的四个平面内部改为 channel-major：

    planes    (token-major)  : t0c0 t0c1 ... t0c31 | t1c0 t1c1 ... t1c31 | ...
    planes_cm (channel-major): t0c0 t1c0 ... t63c0 | t0c1 t1c1 ... t63c1 | ...

三、为什么能提高压缩比
--------------------
严格说整个 block 的零阶熵并没有变——置换不改变字节直方图。变的是**局部统计**：
DEFLATE = LZ77(在 32 KiB 窗口里找重复串) + Huffman(对一段数据用一张码表)。
它对"同一段数据里符号分布是否集中"非常敏感。

1. 原始布局把三种统计特性完全不同的字节交错在一起：
   - int4 nibble 几乎均匀分布(每 4 bit 约 3.9 bit 熵)，接近不可压缩；
   - bf16 低字节(1 位指数 + 7 位尾数)接近随机噪声；
   - bf16 高字节(符号 + 指数高 7 位)在同一层内高度集中：相邻 token、同一通道的
     数值量级相近，指数只取少数几个值。
   混在一起时，Huffman 只能用一张"平均"的码表，高字节的集中分布被 nibble 和
   尾数字节"稀释"，LZ77 也很难找到长匹配(每 2 字节就插入一个随机尾数字节)。

2. 分平面后，高字节平面的条件熵远低于 8 bit，可以用很短的 Huffman 码；
   低字节和 nibble 平面虽然仍不可压缩，但不再拖累其它平面。
   实测(layer_001，zlib)：scales 1.44x -> 1.59x，rope 1.27x -> 1.37x。

3. channel-major 进一步把"同一通道跨 token"的字节排在一起。rope 是位置编码，
   同一维度随位置缓慢变化，相邻 token 的高字节经常相同，形成 LZ77 可以匹配的
   重复串：rope 1.37x -> 1.51x。

4. int4 区占 57% 字节且本身接近满熵，所以整体上限有限：
   zlib  level 1：1.16x -> 1.30x；QAT DEFLATE：1.12x -> 1.22x(8 层实测)。
   再想提升只能对 bf16 部分做有损(截断尾数)，那就不是无损了。

四、运行方式
----------
在 dev 容器内、仓库根目录下运行(dump 目录需在容器内可见)：

    source setvars.sh
    LD_PRELOAD=$IAXL_LD_PRELOAD python3 benchmark/kvstore/kvstore_hunyuan_test.py \\
        --dump-dir hunyuan/int4/merged --num-layers 4 --mla-shuffle planes_cm

压缩后端由环境变量决定，例如 IAXL_QAT_ZIP_ENABLE=1 或 IAXL_KV_COMPRESSION=0，
与 kvstore_benchmark.sh 相同。
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kvstore_benchmark as kb  # sets IAXL_KV_LOSSY_TRUNC / IAXL_KV_DATA_SHUFFLE

import torch

from iaxl.kvstore import KVStore, get_accelerator_device

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DUMP_DIR = os.path.join(REPO_ROOT, "hunyuan", "int4", "merged")
DEFAULT_BLOCK_SIZE = 64  # tokens; the indexer page is also 64 tokens
MLA_INT4_BYTES = 256  # 512 int4 values
MLA_SCALE_BYTES = 64  # 32 bf16
MLA_ROPE_BYTES = 128  # 64 bf16
MLA_ROW_BYTES = MLA_INT4_BYTES + MLA_SCALE_BYTES + MLA_ROPE_BYTES  # 448
MLA_SHUFFLES = ("none", "planes", "planes_cm")
WARMUP_BLOCKS = 8
BLOCK_DIM = 0
CACHE_KINDS = ("attn", "indexer")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test KVStore PUT/GET with the Hunyuan INT4 KV cache dump.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dump-dir",
        default=DEFAULT_DUMP_DIR,
        help="Directory holding manifest.json and layer_NNN.pt files.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layer files to load (default: all layers in the manifest).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Tokens per KV block; one block is BLOCK_SIZE * 448 bytes for every cache.",
    )
    parser.add_argument(
        "--caches",
        choices=("all",) + CACHE_KINDS,
        default="all",
        help="Which caches to test: the MLA KV ('attn'), the FP8 indexer, or both.",
    )
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Check the sha256 recorded in each .pt entry while loading.",
    )
    parser.add_argument(
        "--mla-shuffle",
        choices=MLA_SHUFFLES,
        default="none",
        help="Byte-plane layout applied inside every MLA block before PUT (see module "
        "docstring) to measure its effect on the compression ratio.",
    )
    kb.add_env_arguments(parser)
    args = parser.parse_args(argv)
    if args.block_size <= 0:
        parser.error("--block-size must be greater than 0")
    if args.num_layers is not None and args.num_layers <= 0:
        parser.error("--num-layers must be greater than 0")
    return args


def cache_kind(name: str) -> str:
    return "indexer" if name.endswith(".indexer.k_cache") else "attn"


def view_as_blocks(tensor: torch.Tensor, block_bytes: int) -> torch.Tensor:
    """Flat uint8 bytes -> [num_blocks, block_bytes], dropping the partial tail."""
    flat = tensor.contiguous().view(-1)
    num_blocks = flat.numel() // block_bytes
    if num_blocks == 0:
        raise ValueError(
            f"cache of {flat.numel()} bytes is smaller than one block ({block_bytes} bytes)"
        )
    return flat[: num_blocks * block_bytes].view(num_blocks, block_bytes)


def _bf16_planes(region: torch.Tensor, channel_major: bool) -> List[torch.Tensor]:
    """[nb, bs, 2*C] little-endian bf16 bytes -> [high-byte plane, low-byte plane], each [nb, bs*C]."""
    nb = region.shape[0]
    pairs = region.reshape(nb, region.shape[1], -1, 2)
    hi, lo = pairs[..., 1], pairs[..., 0]
    if channel_major:
        hi, lo = hi.transpose(1, 2), lo.transpose(1, 2)
    return [hi.reshape(nb, -1), lo.reshape(nb, -1)]


def _bf16_unplanes(hi: torch.Tensor, lo: torch.Tensor, bs: int, channel_major: bool) -> torch.Tensor:
    nb = hi.shape[0]
    if channel_major:
        hi, lo = hi.reshape(nb, -1, bs).transpose(1, 2), lo.reshape(nb, -1, bs).transpose(1, 2)
    else:
        hi, lo = hi.reshape(nb, bs, -1), lo.reshape(nb, bs, -1)
    return torch.stack([lo, hi], dim=-1).reshape(nb, bs, -1)


def shuffle_mla_blocks(blocks: torch.Tensor, block_size: int, mode: str) -> torch.Tensor:
    """[nb, bs*448] -> [nb, bs*448]: int4 | scale hi | scale lo | rope hi | rope lo per block."""
    nb = blocks.shape[0]
    rows = blocks.view(nb, block_size, MLA_ROW_BYTES)
    q = rows[:, :, :MLA_INT4_BYTES].reshape(nb, -1)
    sc = rows[:, :, MLA_INT4_BYTES : MLA_INT4_BYTES + MLA_SCALE_BYTES]
    rp = rows[:, :, MLA_INT4_BYTES + MLA_SCALE_BYTES :]
    cm = mode == "planes_cm"
    return torch.cat([q, *_bf16_planes(sc, cm), *_bf16_planes(rp, cm)], dim=1).contiguous()


def unshuffle_mla_blocks(blocks: torch.Tensor, block_size: int, mode: str) -> torch.Tensor:
    nb, bs = blocks.shape[0], block_size
    sizes = [bs * MLA_INT4_BYTES] + [bs * MLA_SCALE_BYTES // 2] * 2 + [bs * MLA_ROPE_BYTES // 2] * 2
    q, sc_hi, sc_lo, rp_hi, rp_lo = torch.split(blocks, sizes, dim=1)
    cm = mode == "planes_cm"
    rows = torch.cat(
        [
            q.reshape(nb, bs, MLA_INT4_BYTES),
            _bf16_unplanes(sc_hi, sc_lo, bs, cm),
            _bf16_unplanes(rp_hi, rp_lo, bs, cm),
        ],
        dim=2,
    )
    return rows.reshape(nb, -1)


def load_dump(
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Return {layer_key: CPU uint8 [num_blocks, block_bytes]} in PUT order plus the manifest."""
    manifest_path = os.path.join(args.dump_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    layers = manifest["layers"]
    if args.num_layers is not None:
        layers = layers[: args.num_layers]
    block_bytes = args.block_size * MLA_ROW_BYTES

    caches: Dict[str, torch.Tensor] = {}
    for layer in layers:
        path = os.path.join(args.dump_dir, layer["file"])
        data = torch.load(path, map_location="cpu", weights_only=True)
        index = data["layer"]
        # Indexer before attn so the last registered layer is an MLA layer,
        # which has the most blocks and thus completes the record for all hashes.
        names = sorted(data["caches"], key=lambda n: cache_kind(n) != "indexer")
        for name in names:
            kind = cache_kind(name)
            if args.caches != "all" and kind != args.caches:
                continue
            entry = data["caches"][name]
            tensor = entry["tensor"]
            if tensor.dtype != torch.uint8 or tensor.numel() != entry["nbytes"]:
                raise ValueError(f"{path}: unexpected tensor for {name}")
            if args.verify_sha256:
                import hashlib

                digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                if digest != entry["sha256"]:
                    raise ValueError(f"{path}: sha256 mismatch for {name}")
            blocks = view_as_blocks(tensor, block_bytes)
            if kind == "attn" and args.mla_shuffle != "none":
                shuffled = shuffle_mla_blocks(blocks, args.block_size, args.mla_shuffle)
                restored = unshuffle_mla_blocks(shuffled, args.block_size, args.mla_shuffle)
                if not torch.equal(restored, blocks):
                    raise AssertionError(f"--mla-shuffle {args.mla_shuffle} does not round-trip")
                blocks = shuffled
            caches[f"{index}.{kind}"] = blocks
        print(f"[dump] loaded {layer['file']} ({len(names)} caches)")

    if not caches:
        raise ValueError(f"no '{args.caches}' caches found in {args.dump_dir}")
    return caches, manifest


class HunyuanTest(kb.Benchmark):
    """All blocks of every dumped cache are PUT, GET and fully verified."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rank = 0
        self.tp_size = 1
        self.block_bytes = args.block_size * MLA_ROW_BYTES
        self.cpu_caches, self.manifest = load_dump(args)
        self.layer_names = list(self.cpu_caches)
        self.block_indices = {
            name: list(range(t.shape[0])) for name, t in self.cpu_caches.items()
        }
        self.block_hashes = {
            name: kb.make_block_hashes(len(idx)) for name, idx in self.block_indices.items()
        }
        self.total_blocks = sum(len(idx) for idx in self.block_indices.values())
        self.total_bytes = self.total_blocks * self.block_bytes

    def print_config(self) -> None:
        kinds = {kind: 0 for kind in CACHE_KINDS}
        for name in self.layer_names:
            kinds[name.split(".", 1)[1]] += 1
        attn = next((t for n, t in self.cpu_caches.items() if n.endswith(".attn")), None)
        print("=" * 80)
        print("KVStore Hunyuan INT4 dump test")
        print("=" * 80)
        print(f"Dump dir:       {self.args.dump_dir}")
        print(
            f"Dump:           {self.manifest['kv_cache_dtype']}, "
            f"{self.manifest['valid_tokens']} tokens, "
            f"{self.manifest['num_hidden_layers']} hidden layers"
        )
        print(f"Caches tested:  {kinds['attn']} MLA + {kinds['indexer']} indexer")
        print(
            f"Block:          {self.args.block_size} tokens x {MLA_ROW_BYTES} B = "
            f"{kb.format_bytes(self.block_bytes)}"
        )
        if attn is not None:
            print(f"MLA layer:      {list(attn.shape)} uint8 blocks")
        print(f"Blocks tested:  {self.total_blocks} (all blocks of every cache)")
        print(f"MLA shuffle:    {self.args.mla_shuffle}")
        print(f"Quant/shuffle:  {self.args.quant}/{self.args.data_shuffle}")
        print(f"Transfer size:  {kb.format_bytes(self.total_bytes)}")

    def setup(self) -> None:
        print("\nUploading KV caches to GPU...")
        self.kv_caches = {name: t.to("cuda") for name, t in self.cpu_caches.items()}
        torch.cuda.synchronize()
        self.kvstore = KVStore(
            model_name="kvstore_hunyuan_test",
            kv_caches=self.kv_caches,
            block_dim=BLOCK_DIM,
            rank=self.rank,
            tp_size=self.tp_size,
        )

    def metrics(self, **params) -> dict:
        return self.kvstore.metrics(params)

    def status(self) -> dict:
        return self.kvstore.status()

    def warmup(self) -> None:
        self.metrics(enable=1, reset=1)
        print("\nWarming up...")
        put_tasks = {}
        for name in self.layer_names:
            indices = self.block_indices[name][:WARMUP_BLOCKS]
            hashes = [f"warmup_{h}" for h in self.block_hashes[name][:WARMUP_BLOCKS]]
            put_tasks.update(
                self.kvstore.put(indices, hashes, layer_names=[name], description="warmup PUT")
            )
        if not self.kvstore.put_wait(put_tasks):
            raise RuntimeError("warm-up PUT did not complete")
        get_tasks = {}
        for name in self.layer_names:
            indices = self.block_indices[name][:WARMUP_BLOCKS]
            hashes = [f"warmup_{h}" for h in self.block_hashes[name][:WARMUP_BLOCKS]]
            get_tasks.update(
                self.kvstore.get(indices, hashes, layer_names=[name], description="warmup GET")
            )
        if not self.kvstore.get_wait(get_tasks):
            raise RuntimeError("warm-up GET did not complete")
        torch.cuda.synchronize()
        self.metrics(reset=1)

    def put(self) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.cuda.nvtx.range("hunyuan PUT"):
            put_tasks = {}
            for name in self.layer_names:
                put_tasks.update(
                    self.kvstore.put(
                        self.block_indices[name],
                        self.block_hashes[name],
                        layer_names=[name],
                        description="hunyuan PUT",
                    )
                )
            if not self.kvstore.put_wait(put_tasks):
                raise RuntimeError("PUT did not complete")
        return time.perf_counter() - start

    def get(self) -> float:
        for tensor in self.kv_caches.values():
            tensor.zero_()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.cuda.nvtx.range("hunyuan GET"):
            get_tasks = {}
            for name in self.layer_names:
                get_tasks.update(
                    self.kvstore.get(
                        self.block_indices[name],
                        self.block_hashes[name],
                        layer_names=[name],
                        description="hunyuan GET",
                    )
                )
            for name in self.layer_names:
                if not self.kvstore.get_wait(get_tasks, layer_names=[name]):
                    raise RuntimeError("GET did not complete")
        return time.perf_counter() - start

    def verify(self) -> bool:
        ok = True
        for name, tensor in self.kv_caches.items():
            got = tensor.cpu()
            if not torch.equal(got, self.cpu_caches[name]):
                bad = (got != self.cpu_caches[name]).any(dim=1).nonzero().flatten()
                print(f"[verify] {name}: {bad.numel()} blocks differ, first {bad[:8].tolist()}")
                ok = False
        return ok


def run_test(args: argparse.Namespace) -> bool:
    if get_accelerator_device() != "cuda":
        print("No CUDA device available; KVStore test cannot run.")
        return False

    test = HunyuanTest(args)
    test.print_config()
    test.setup()
    test.warmup()
    put_time = test.put()
    get_time = test.get()
    verified = test.verify()
    kb.print_report(
        put_time,
        get_time,
        verified,
        test.status(),
        test.metrics(),
        test.total_blocks,
        test.total_bytes,
    )
    test.kvstore.stop()
    return verified


def main() -> int:
    return 0 if run_test(parse_args()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
