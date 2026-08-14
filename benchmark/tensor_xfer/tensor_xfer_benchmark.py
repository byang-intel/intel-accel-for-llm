#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare fragmented CUDA, Triton, and IAXL DSA transfers.

Example:
    IAXL_DSA_GD_ENABLE=1 \
    python3 benchmark/tensor_xfer/tensor_xfer_benchmark.py
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any

import cupy  # pyright: ignore[reportMissingImports]
import matplotlib
import torch
import triton
import triton.language as tl
from cuda.bindings import runtime as cuda_runtime  # pyright: ignore[reportAttributeAccessIssue]

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iaxl.tensor_ops import SliceCopier


DEFAULT_FRAG_SIZES = "4K,16K,32K,64K,128K,256K,512K,1M,2M"
METHOD_COLORS = {
    "CUDA copy_": "#d62728",
    "cudaMemcpy3DBatchAsync": "#ff7f0e",
    "Triton kernel": "#2ca02c",
    "IAXL DSA": "#1f77b4",
}


def parse_size(value: str) -> int:
    value = value.strip().upper()
    if value.endswith("B"):
        value = value[:-1]
    multiplier = 1
    if value.endswith("K"):
        multiplier, value = 1 << 10, value[:-1]
    elif value.endswith("M"):
        multiplier, value = 1 << 20, value[:-1]
    size = int(float(value) * multiplier)
    if size <= 0 or size % 2:
        raise argparse.ArgumentTypeError("fragment sizes must be positive and even")
    return size


def parse_sizes(value: str) -> list[int]:
    return [parse_size(item) for item in value.split(",")]


def format_size(size: int) -> str:
    if size >= 1 << 20:
        return f"{size / (1 << 20):g}M"
    if size >= 1 << 10:
        return f"{size / (1 << 10):g}K"
    return f"{size}B"


def host_cuda_view(tensor: torch.Tensor) -> torch.Tensor:
    owner = cupy.cuda.UnownedMemory(tensor.data_ptr(), tensor.nbytes, tensor)
    pointer = cupy.cuda.MemoryPointer(owner, 0)
    array = cupy.ndarray((tensor.numel(),), dtype=cupy.int16, memptr=pointer)
    return torch.from_dlpack(array)


def build_3d_batch_ops(
    src_base: int,
    dst_base: int,
    src_offsets: list[int],
    dst_offsets: list[int],
    frag_bytes: int,
) -> list[Any]:
    order = cuda_runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
    pointer_type = cuda_runtime.cudaMemcpy3DOperandType.cudaMemcpyOperandTypePointer
    operations = []
    for src_offset, dst_offset in zip(src_offsets, dst_offsets):
        operation = cuda_runtime.cudaMemcpy3DBatchOp()

        src = cuda_runtime.cudaMemcpy3DOperand()
        src.type = pointer_type
        src_union = src.op
        src_pointer = src_union.ptr
        src_pointer.ptr = src_base + src_offset
        src_pointer.rowLength = 0
        src_pointer.layerHeight = 0
        src_union.ptr = src_pointer
        src.op = src_union
        operation.src = src

        dst = cuda_runtime.cudaMemcpy3DOperand()
        dst.type = pointer_type
        dst_union = dst.op
        dst_pointer = dst_union.ptr
        dst_pointer.ptr = dst_base + dst_offset
        dst_pointer.rowLength = 0
        dst_pointer.layerHeight = 0
        dst_union.ptr = dst_pointer
        dst.op = dst_union
        operation.dst = dst

        extent = cuda_runtime.cudaExtent()
        extent.width = frag_bytes
        extent.height = 1
        extent.depth = 1
        operation.extent = extent
        operation.srcAccessOrder = order
        operation.flags = 0
        operations.append(operation)
    return operations


@triton.jit
def copy_kernel(
    src,
    dst,
    src_offsets,
    dst_offsets,
    FRAG: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    BLOCK: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
):
    fragment = tl.program_id(0)
    src_start = tl.load(src_offsets + fragment)
    dst_start = tl.load(dst_offsets + fragment)
    for start in range(0, FRAG, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < FRAG
        values = tl.load(src + src_start + offsets, mask=mask)
        tl.store(dst + dst_start + offsets, values, mask=mask)


@dataclass
class Result:
    method: str
    direction: str
    milliseconds: float
    total_bytes: int
    valid: bool

    @property
    def gbps(self) -> float:
        return self.total_bytes / (self.milliseconds / 1000) / 1e9


def measure(operation, *, cuda_timing: bool, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    best = float("inf")
    for _ in range(iterations):
        if cuda_timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
        else:
            start_time = time.perf_counter()
            operation()
            elapsed = (time.perf_counter() - start_time) * 1000
        best = min(best, elapsed)
    return best


def run(frag_bytes: int, args: argparse.Namespace) -> list[Result]:
    total_target = int(args.total_gib * (1 << 30))
    fragments = max(1, total_target // frag_bytes)
    frag_elements = frag_bytes // 2
    total_bytes = fragments * frag_bytes
    elements = fragments * frag_elements

    source = torch.randint(-32768, 32767, (elements,), dtype=torch.int16, pin_memory=True)
    permutation = torch.randperm(fragments)
    src_offsets = torch.arange(fragments, dtype=torch.int64) * frag_elements
    dst_offsets = permutation * frag_elements
    src_indices = (src_offsets // frag_elements).tolist()
    dst_indices = (dst_offsets // frag_elements).tolist()

    source_fragments = source.view(fragments, frag_elements)
    expected = torch.empty_like(source_fragments)
    expected[permutation] = source_fragments
    expected = expected.flatten()

    results = []
    directions = ("H2D", "D2H") if args.direction == "both" else (args.direction.upper(),)
    for direction in directions:
        h2d = direction == "H2D"
        host = source.clone().pin_memory()
        gpu = torch.empty(elements, dtype=torch.int16, device="cuda")
        if not h2d:
            gpu.copy_(source)
        torch.cuda.synchronize()

        src_tensor, dst_tensor = (host, gpu) if h2d else (gpu, host)

        def clear_destination():
            dst_tensor.zero_()
            torch.cuda.synchronize()

        def valid() -> bool:
            torch.cuda.synchronize()
            actual = dst_tensor.cpu() if dst_tensor.is_cuda else dst_tensor
            return torch.equal(actual, expected)

        def cuda_copy():
            for src_index, dst_index in zip(src_indices, dst_indices):
                src_start = src_index * frag_elements
                dst_start = dst_index * frag_elements
                dst_tensor[dst_start : dst_start + frag_elements].copy_(
                    src_tensor[src_start : src_start + frag_elements], non_blocking=True
                )

        clear_destination()
        elapsed = measure(
            cuda_copy, cuda_timing=True, warmup=args.warmup, iterations=args.iterations
        )
        results.append(Result("CUDA copy_", direction, elapsed, total_bytes, valid()))

        src_byte_offsets = [index * frag_bytes for index in src_indices]
        dst_byte_offsets = [index * frag_bytes for index in dst_indices]
        batch_operations = build_3d_batch_ops(
            src_tensor.data_ptr(),
            dst_tensor.data_ptr(),
            src_byte_offsets,
            dst_byte_offsets,
            frag_bytes,
        )
        batch_stream = torch.cuda.Stream()

        def cuda_3d_batch_copy():
            result = cuda_runtime.cudaMemcpy3DBatchAsync(
                len(batch_operations), batch_operations, 0, batch_stream.cuda_stream
            )
            if result[0] != cuda_runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpy3DBatchAsync failed: {result[0]}")

        clear_destination()
        with torch.cuda.stream(batch_stream):
            elapsed = measure(
                cuda_3d_batch_copy,
                cuda_timing=True,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        results.append(
            Result("cudaMemcpy3DBatchAsync", direction, elapsed, total_bytes, valid())
        )

        src_view = src_tensor if src_tensor.is_cuda else host_cuda_view(src_tensor)
        dst_view = dst_tensor if dst_tensor.is_cuda else host_cuda_view(dst_tensor)
        src_offsets_gpu = src_offsets.cuda()
        dst_offsets_gpu = dst_offsets.cuda()

        def triton_copy():
            copy_kernel[(fragments,)](
                src_view,
                dst_view,
                src_offsets_gpu,
                dst_offsets_gpu,
                FRAG=frag_elements,  # pyright: ignore[reportArgumentType]
                BLOCK=args.block,
            )

        clear_destination()
        elapsed = measure(
            triton_copy, cuda_timing=True, warmup=args.warmup, iterations=args.iterations
        )
        results.append(Result("Triton kernel", direction, elapsed, total_bytes, valid()))

        if h2d:
            cpu_tensors = list(source_fragments.unbind())
            iaxl_copier = SliceCopier(
                cpu_tensors, 0, dst_indices, out=gpu.view(fragments, frag_elements)
            )

        else:
            host_fragments = host.view(fragments, frag_elements)
            cpu_tensors = [host_fragments[index] for index in dst_indices]
            iaxl_copier = SliceCopier(
                gpu.view(fragments, frag_elements), 0, src_indices, out=cpu_tensors
            )

        clear_destination()
        elapsed = measure(
            iaxl_copier.copy,
            cuda_timing=False,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        results.append(Result("IAXL DSA", direction, elapsed, total_bytes, valid()))

    return results


def plot_results(
    results: dict[str, dict[int, list[Result]]], frag_sizes: list[int], path: str
) -> None:
    directions = [direction for direction in ("H2D", "D2H") if direction in results]
    figure, axes = plt.subplots(
        len(directions), 1, figsize=(8, 5 * len(directions)), squeeze=False
    )

    for axis, direction in zip(axes[:, 0], directions):
        by_fragment = results[direction]
        methods = list(METHOD_COLORS)
        for method in methods:
            points = [
                (size, result.gbps)
                for size in frag_sizes
                for result in by_fragment.get(size, [])
                if result.method == method
            ]
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    color=METHOD_COLORS[method],
                    label=method,
                )

        axis.axvspan(4 << 10, 32 << 10, color="gold", alpha=0.15, label="KV 4K-32K")
        axis.set_xscale("log", base=2)
        axis.set_xticks(frag_sizes)
        axis.set_xticklabels([format_size(size) for size in frag_sizes], rotation=45)
        axis.set_xlabel("Fragment size")
        axis.set_ylabel("GB/s")
        axis.set_title(direction)
        axis.grid(True, which="both", linestyle=":", alpha=0.5)
        axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fragmented H2D/D2H copies using CUDA, Triton, and IAXL DSA."
    )
    parser.add_argument("--total-gib", type=float, default=0.1)
    parser.add_argument("--frag-sizes", type=parse_sizes, default=DEFAULT_FRAG_SIZES)
    parser.add_argument("--direction", choices=("h2d", "d2h", "both"), default="both")
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--plot",
        default="tensor_xfer.png",
        metavar="PATH",
        help="output plot path; use an empty string to disable plotting",
    )
    args = parser.parse_args()
    if isinstance(args.frag_sizes, str):
        args.frag_sizes = parse_sizes(args.frag_sizes)
    if not torch.cuda.is_available():
        parser.error("CUDA is not available")
#    if os.environ.get("IAXL_DSA_GD_ENABLE", "0").lower() not in ("1", "true", "yes", "on"):
#        parser.error("set IAXL_DSA_GD_ENABLE=1 to benchmark IAXL DSA")

    torch.manual_seed(42)
    print(f"{'Direction':9} {'Fragment':>10} {'Method':24} {'ms':>10} {'GB/s':>10} Check")
    all_results: dict[str, dict[int, list[Result]]] = {}
    for frag_bytes in args.frag_sizes:
        fragment_results = run(frag_bytes, args)
        for result in fragment_results:
            all_results.setdefault(result.direction, {}).setdefault(frag_bytes, []).append(result)
            print(
                f"{result.direction:9} {frag_bytes / 1024:9g}K "
                f"{result.method:24} {result.milliseconds:10.3f} "
                f"{result.gbps:10.2f} {'PASS' if result.valid else 'FAIL'}"
            )
    if args.plot:
        plot_results(all_results, args.frag_sizes, args.plot)


if __name__ == "__main__":
    main()
