#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Hunyuan FP8 MLA KV cache 的 kv_zip 压缩/解压测试。

与 tests/hunyuan-kv-zip-int4.py 同一套流程(加载 -> block 内重排 -> compress_tensors ->
decompress_tensors -> 逆重排 -> 逐字节比对 -> 汇总压缩比与吞吐)，复用其中的加载、
回环校验、计时和命令行解析；本文件只提供 FP8 行布局对应的重排/逆重排。

数据来自 Hunyuan FP8 dump(格式见 hunyuan/fp8/KV_CACHE_DUMP_FORMAT.md)，每层一个
``layer_NNN.pt``，MLA KV 为 ``[tokens, 656]`` 的 torch.uint8 原始字节。

在 dev 容器内、仓库根目录下运行：

    python3 tests/hunyuan-kv-zip-fp8.py [--dump-dir DIR] [--layers N ...] [--block-size N]
                                        [--max-blocks N] [--iters N] [--mla-shuffle MODE ...]

参数含义与 INT4 脚本相同，--dump-dir 默认 hunyuan/fp8/merged。

block 内字节重排(--mla-shuffle)
------------------------------
shuffle 前(dump 原始布局，一个 block = block_size 个 token 行，token-major)：

    token 0 : [ fp8 e4m3 x512 = 512 B | scale fp32 x4 = 16 B | rope bf16 x64 = 128 B ]
    token 1 : [ fp8 e4m3 x512 = 512 B | scale fp32 x4 = 16 B | rope bf16 x64 = 128 B ]
    ...
    token 63: [ fp8 e4m3 x512 = 512 B | scale fp32 x4 = 16 B | rope bf16 x64 = 128 B ]

    fp8 e4m3 = 1 符号 + 4 指数 + 3 尾数，单字节，无法再按字节拆分。
    fp32(小端四字节) b3 = 符号 + 指数高 7 位，b2 = 指数最低 1 位 + 尾数高 7 位，
    b1/b0 = 尾数低 16 位(接近随机)。
    bf16(小端两字节) hi = 符号 + 指数高 7 位(分布集中)，lo = 指数最低 1 位 + 7 位尾数。

shuffle 后(``planes``，七个字节平面首尾相接，总长不变)：

    [ fp8 平面       : 所有 token 的 512 B fp8，共 bs*512 B ]
    [ scale b3 平面  : 所有 token 的 4 个 scale 最高字节，共 bs*4 B ]
    [ scale b2 平面  :                                   共 bs*4 B ]
    [ scale b1 平面  :                                   共 bs*4 B ]
    [ scale b0 平面  :                                   共 bs*4 B ]
    [ rope  hi 平面  : 所有 token 的 64 个 rope 高字节，共 bs*64 B ]
    [ rope  lo 平面  : 所有 token 的 64 个 rope 低字节，共 bs*64 B ]

``planes_cm`` 与 ``planes`` 相同，但每个平面内部改为 channel-major
(t0c0 t1c0 ... t63c0 | t0c1 t1c1 ... )，fp8 平面同样按通道排列。

为什么能提高压缩比的分析见 INT4 脚本的模块注释；FP8 的区别在于：
  * fp8 区占 78% 字节，每个字节里指数(4 bit)占比更高、分布更集中，所以
    channel-major 把同一通道跨 token 的 fp8 排在一起对 LZ77 更有帮助；
  * scale 只有 4 个 fp32/token(2.4%)，字节平面化主要是避免其随机尾数污染其它平面。
"""

import importlib.util
import os
import sys
from typing import List

os.environ.setdefault("IAXL_CPU_ZIP_ENABLE", "1")
os.environ["IAXL_KV_LOSSY_TRUNC"] = "0"

import torch  # noqa: E402


def _import_int4_module():
    """Load the sibling hunyuan-kv-zip-int4.py (hyphenated name, not importable directly)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hunyuan-kv-zip-int4.py")
    spec = importlib.util.spec_from_file_location("hunyuan_kv_zip_int4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


int4 = _import_int4_module()

DEFAULT_DUMP_DIR = os.path.join(int4.REPO_ROOT, "hunyuan", "fp8", "merged")

FP8_BYTES = 512  # 512 fp8 e4m3 values
FP8_SCALE_BYTES = 16  # 4 fp32
FP8_ROPE_BYTES = 128  # 64 bf16
FP8_ROW_BYTES = FP8_BYTES + FP8_SCALE_BYTES + FP8_ROPE_BYTES  # 656
FP8_SHUFFLES = ("none", "planes", "planes_cm")


def _fp32_planes(region: torch.Tensor, channel_major: bool) -> List[torch.Tensor]:
    """[nb, bs, 4*C] little-endian fp32 bytes -> [b3, b2, b1, b0] planes, each [nb, bs*C]."""
    nb = region.shape[0]
    quads = region.reshape(nb, region.shape[1], -1, 4)
    planes = [quads[..., i] for i in (3, 2, 1, 0)]
    if channel_major:
        planes = [p.transpose(1, 2) for p in planes]
    return [p.reshape(nb, -1) for p in planes]


def _fp32_unplanes(planes: List[torch.Tensor], bs: int, channel_major: bool) -> torch.Tensor:
    nb = planes[0].shape[0]
    if channel_major:
        planes = [p.reshape(nb, -1, bs).transpose(1, 2) for p in planes]
    else:
        planes = [p.reshape(nb, bs, -1) for p in planes]
    b3, b2, b1, b0 = planes
    return torch.stack([b0, b1, b2, b3], dim=-1).reshape(nb, bs, -1)


def shuffle_fp8_blocks(blocks: torch.Tensor, block_size: int, mode: str) -> torch.Tensor:
    """[nb, bs*656] -> [nb, bs*656]: fp8 | scale b3..b0 | rope hi | rope lo per block."""
    if mode == "none":
        return blocks
    nb = blocks.shape[0]
    rows = blocks.view(nb, block_size, FP8_ROW_BYTES)
    cm = mode == "planes_cm"
    q = rows[:, :, :FP8_BYTES]
    if cm:
        q = q.transpose(1, 2)
    sc = rows[:, :, FP8_BYTES : FP8_BYTES + FP8_SCALE_BYTES]
    rp = rows[:, :, FP8_BYTES + FP8_SCALE_BYTES :]
    return torch.cat(
        [q.reshape(nb, -1), *_fp32_planes(sc, cm), *int4._bf16_planes(rp, cm)], dim=1
    ).contiguous()


def unshuffle_fp8_blocks(blocks: torch.Tensor, block_size: int, mode: str) -> torch.Tensor:
    if mode == "none":
        return blocks
    nb, bs = blocks.shape[0], block_size
    sizes = [bs * FP8_BYTES] + [bs * FP8_SCALE_BYTES // 4] * 4 + [bs * FP8_ROPE_BYTES // 2] * 2
    q, b3, b2, b1, b0, rp_hi, rp_lo = torch.split(blocks, sizes, dim=1)
    cm = mode == "planes_cm"
    q = q.reshape(nb, FP8_BYTES, bs).transpose(1, 2) if cm else q.reshape(nb, bs, FP8_BYTES)
    rows = torch.cat(
        [q, _fp32_unplanes([b3, b2, b1, b0], bs, cm), int4._bf16_unplanes(rp_hi, rp_lo, bs, cm)],
        dim=2,
    )
    return rows.reshape(nb, -1)


def main() -> int:
    args = int4.parse_args("kv_zip round trip on the Hunyuan FP8 MLA dump", DEFAULT_DUMP_DIR, FP8_SHUFFLES)
    return int4.run(args, FP8_ROW_BYTES, shuffle_fp8_blocks, unshuffle_fp8_blocks)


if __name__ == "__main__":
    sys.exit(main())
