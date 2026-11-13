#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Hunyuan INT4 MLA KV cache 的 kv_zip 压缩/解压测试。

直接调用 torch_ext.compress_tensors / decompress_tensors(用法见 tests/iaxl_unittest.py)，
不经过 KVStore。数据来自 Hunyuan INT4 dump(格式见
benchmark/kvstore/HUNYUAN_INT4_KV_CACHE_DUMP_FORMAT.md)，每层一个 ``layer_NNN.pt``，
MLA KV 为 ``[tokens, 448]`` 的 torch.uint8 原始字节。

在 dev 容器内、仓库根目录下运行：

    python3 tests/hunyuan-kv-zip-int4.py [--dump-dir DIR] [--layer N] [--block-size N]
                                         [--max-blocks N] [--mla-shuffle MODE ...]

命令行参数：
    --dump-dir      dump 目录(默认 hunyuan/int4/merged)
    --layer         测试的层号(默认 1)
    --block-size    每个 block 的 token 数(默认 64)
    --max-blocks    最多测试的 block 数(默认 0 = 全部)
    --mla-shuffle   要测试的重排模式，可多选(默认 none planes planes_cm)
压缩后端由 IAXL_*_ZIP_ENABLE 选择，默认开启 CPU 后端；回环要求逐字节一致，所以关闭有损截断。
任何一处数据不一致都会以非零退出码结束。

一、block 内字节重排(--mla-shuffle；代码与说明拷贝自 benchmark/kvstore/kvstore_hunyuan_test.py)
--------------------------------------------------------------------------------
kv_zip 以 block 为单位做 DEFLATE，所以任何预处理都必须限制在 block 内部，
才能在不改 C++ 代码的前提下验证效果；解压回来的仍是重排后的字节，由 Python 侧
的逆变换还原。这是纯粹的**字节置换，无损**：不改变任何字节的值，也不增减字节数。

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

二、为什么能提高压缩比
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
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

os.environ.setdefault("IAXL_CPU_ZIP_ENABLE", "1")
os.environ["IAXL_KV_LOSSY_TRUNC"] = "0"

import torch  # noqa: E402
from iaxl import torch_ext  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DUMP_DIR = os.path.join(REPO_ROOT, "hunyuan", "int4", "merged")

MLA_INT4_BYTES = 256  # 512 int4 values
MLA_SCALE_BYTES = 64  # 32 bf16
MLA_ROPE_BYTES = 128  # 64 bf16
MLA_ROW_BYTES = MLA_INT4_BYTES + MLA_SCALE_BYTES + MLA_ROPE_BYTES  # 448
MLA_SHUFFLES = ("none", "planes", "planes_cm")
WARMUP_BLOCKS = 64


# --- 数据格式调整(拷贝自 benchmark/kvstore/kvstore_hunyuan_test.py) -----------------


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
    if mode == "none":
        return blocks
    nb = blocks.shape[0]
    rows = blocks.view(nb, block_size, MLA_ROW_BYTES)
    q = rows[:, :, :MLA_INT4_BYTES].reshape(nb, -1)
    sc = rows[:, :, MLA_INT4_BYTES : MLA_INT4_BYTES + MLA_SCALE_BYTES]
    rp = rows[:, :, MLA_INT4_BYTES + MLA_SCALE_BYTES :]
    cm = mode == "planes_cm"
    return torch.cat([q, *_bf16_planes(sc, cm), *_bf16_planes(rp, cm)], dim=1).contiguous()


def unshuffle_mla_blocks(blocks: torch.Tensor, block_size: int, mode: str) -> torch.Tensor:
    if mode == "none":
        return blocks
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


# --- 测试 -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kv_zip round trip on the Hunyuan INT4 MLA dump")
    parser.add_argument("--dump-dir", default=DEFAULT_DUMP_DIR, help="directory with layer_NNN.pt")
    parser.add_argument("--layer", type=int, default=1, help="layer index to load")
    parser.add_argument("--block-size", type=int, default=64, help="tokens per block")
    parser.add_argument("--max-blocks", type=int, default=0, help="limit the number of blocks (0 = all)")
    parser.add_argument(
        "--mla-shuffle",
        nargs="+",
        choices=MLA_SHUFFLES,
        default=list(MLA_SHUFFLES),
        help="in-block byte shuffle modes to test",
    )
    return parser.parse_args()


def load_mla_blocks(dump_dir: str, layer: int, block_size: int, max_blocks: int) -> torch.Tensor:
    """layer_NNN.pt 中的 MLA cache -> CPU uint8 [num_blocks, block_size*448]."""
    path = os.path.join(dump_dir, f"layer_{layer:03d}.pt")
    data = torch.load(path, map_location="cpu", weights_only=True)
    tensor = next(e["tensor"] for e in data["caches"].values() if e["tensor"].dim() == 2)
    assert tensor.dtype == torch.uint8 and tensor.shape[1] == MLA_ROW_BYTES, tensor.shape
    blocks = view_as_blocks(tensor, block_size * MLA_ROW_BYTES)
    if max_blocks > 0:
        blocks = blocks[:max_blocks]
    return blocks


def check_shuffle_roundtrip(blocks: torch.Tensor, block_size: int, mode: str) -> None:
    shuffled = shuffle_mla_blocks(blocks, block_size, mode)
    if shuffled.shape != blocks.shape:
        raise RuntimeError(f"{mode}: shuffle changed shape {tuple(blocks.shape)} -> {tuple(shuffled.shape)}")
    restored = unshuffle_mla_blocks(shuffled, block_size, mode)
    if not torch.equal(restored, blocks):
        raise RuntimeError(f"{mode}: shuffle/unshuffle round trip mismatch")


def zip_roundtrip(blocks: torch.Tensor, block_size: int, mode: str) -> Tuple[float, float, float]:
    """shuffle -> compress_tensors -> decompress_tensors -> unshuffle.

    Returns (ratio, zip_seconds, unzip_seconds); the timings cover only the torch_ext calls.
    """
    shuffled = shuffle_mla_blocks(blocks, block_size, mode)
    # Inputs: contiguous CPU tensors of one dtype and size, one per KV block.
    # Segmented compression uses the inputs as scratch, so hand over copies.
    tensors = [b.clone() for b in shuffled]

    # Compress: one uint8 tensor (the block) per input.
    t0 = time.perf_counter()
    zipped = torch_ext.compress_tensors(tensors)
    zip_s = time.perf_counter() - t0
    if len(zipped) != len(tensors):
        raise RuntimeError(f"{mode}: compress_tensors returned {len(zipped)} blocks for {len(tensors)} inputs")
    if any(z.dtype != torch.uint8 for z in zipped):
        raise RuntimeError(f"{mode}: compressed blocks are not uint8")

    # Decompress into preallocated tensors of the original shape and dtype.
    outs = [torch.empty_like(b) for b in shuffled]
    t0 = time.perf_counter()
    torch_ext.decompress_tensors(zipped, outs)
    unzip_s = time.perf_counter() - t0
    restored = unshuffle_mla_blocks(torch.stack(outs), block_size, mode)
    if not torch.equal(restored, blocks):
        raise RuntimeError(f"{mode}: data mismatch after compress/decompress round trip")

    zip_bytes = sum(z.numel() for z in zipped)
    return blocks.nbytes / zip_bytes, zip_s, unzip_s


def main() -> int:
    args = parse_args()
    path = os.path.join(args.dump_dir, f"layer_{args.layer:03d}.pt")
    if not os.path.exists(path):
        print(f"error: dump not found: {path} (use --dump-dir)", file=sys.stderr)
        return 2

    blocks = load_mla_blocks(args.dump_dir, args.layer, args.block_size, args.max_blocks)
    print(
        f"[dump] layer {args.layer}: {blocks.shape[0]} blocks x {args.block_size} tokens x "
        f"{MLA_ROW_BYTES} B = {blocks.nbytes / 2**20:.1f} MiB"
    )

    try:
        for mode in args.mla_shuffle:
            check_shuffle_roundtrip(blocks, args.block_size, mode)
        print(f"[shuffle] round trip OK: {' '.join(args.mla_shuffle)}")

        # One untimed pass so backend init / thread pool spin-up stay out of the numbers.
        zip_roundtrip(blocks[: min(WARMUP_BLOCKS, blocks.shape[0])], args.block_size, args.mla_shuffle[0])

        gb = blocks.nbytes / 1e9
        for mode in args.mla_shuffle:
            ratio, zip_s, unzip_s = zip_roundtrip(blocks, args.block_size, mode)
            print(
                f"[zip] {mode:<10} ratio {ratio:.3f}x ({1 - 1 / ratio:.1%} saved)  "
                f"zip {gb / zip_s:6.2f} GB/s ({zip_s * 1e3:.1f} ms)  "
                f"unzip {gb / unzip_s:6.2f} GB/s ({unzip_s * 1e3:.1f} ms)  verify OK"
            )
    except RuntimeError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
