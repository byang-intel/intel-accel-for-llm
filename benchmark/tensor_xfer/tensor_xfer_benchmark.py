#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare fragmented CUDA, Triton, and IAXL DSA transfers.

Example:
    IAXL_DSA_GD_ENABLE=1 \
    python3 benchmark/tensor_xfer/tensor_xfer_benchmark.py
"""

import argparse
import contextlib
import os
import threading
import time
from collections import deque
from collections.abc import Callable
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
from iaxl.utils.flamegraph import PerfRecorder


DEFAULT_FRAG_SIZES = "1K,4K,8K,16K,32K,64K,128K,256K,512K,1M,2M"
DEFAULT_OUTPUT_DIR = "/_data/tensor_xfer_benchmark"
# BIDIR issues H2D and D2H at the same time, one thread per direction (PCIe full duplex),
# and reports each direction separately as "BIDIR H2D" / "BIDIR D2H".
DIRECTIONS = {
    "h2d": ("H2D",),
    "d2h": ("D2H",),
    "both": ("H2D", "D2H"),
    "bidir": ("BIDIR",),
    "all": ("H2D", "D2H", "BIDIR"),
}
BIDIR_METHODS = ("cuda", "batch", "iaxl")
# Passes queued ahead per stream; keeps both directions busy without a long drain tail.
BIDIR_LOOKAHEAD = 2
PLOT_DIRECTIONS = ("H2D", "D2H", "BIDIR H2D", "BIDIR D2H")
METHODS = {
    "cuda": "CUDA Copy",
    "batch": "CUDA Copy Batch",
    "triton": "CUDA Kernel",
    "iaxl": "IAXL DSA",
}
METHOD_COLORS = {
    "CUDA Copy": "#d62728",
    "CUDA Copy Batch": "#ff7f0e",
    "CUDA Kernel": "#2ca02c",
    "IAXL DSA": "#1f77b4",
}
# Same color per method; the line style tells the variants apart.
VARIANT_STYLES = {"": ("-", "o"), "max": ("--", None), "min": (":", None)}
RECORDER: PerfRecorder | None = None
# Called before every timed iteration; multi-rank runs use it to align all ranks.
SYNC: Callable[[], None] | None = None


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


def parse_methods(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"invalid choice: {', '.join(unknown)} (choose from {', '.join(METHODS)})"
        )
    return names


def format_size(size: int) -> str:
    if size >= 1 << 20:
        return f"{size / (1 << 20):g}M"
    if size >= 1 << 10:
        return f"{size / (1 << 10):g}K"
    return f"{size}B"


def output_path(directory: str, name: str) -> str:
    return name if os.path.isabs(name) else os.path.join(directory, name)


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


def batch_copy(operations: list[Any]) -> None:
    result = cuda_runtime.cudaMemcpy3DBatchAsync(
        len(operations), operations, 0, torch.cuda.current_stream().cuda_stream
    )
    if result[0] != cuda_runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy3DBatchAsync failed: {result[0]}")


def slice_copy(
    src_tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    src_indices: list[int],
    dst_indices: list[int],
    frag_elements: int,
) -> None:
    for src_index, dst_index in zip(src_indices, dst_indices):
        src_start = src_index * frag_elements
        dst_start = dst_index * frag_elements
        dst_tensor[dst_start : dst_start + frag_elements].copy_(
            src_tensor[src_start : src_start + frag_elements], non_blocking=True
        )


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
    variant: str = ""  # "" plots as a solid line; "min"/"max" as dotted/dashed

    @property
    def gbps(self) -> float:
        return self.total_bytes / (self.milliseconds / 1000) / 1e9


def measure(
    operation,
    *,
    cuda_timing: bool,
    warmup: int,
    iterations: int,
    label: tuple[str, ...] = (),
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    # Sampling starts after warmup so the flame graph only covers timed work.
    if RECORDER is not None:
        RECORDER.start(label)
    best = float("inf")
    for _ in range(iterations):
        if SYNC is not None:
            SYNC()
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
    if RECORDER is not None:
        RECORDER.stop()
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
    for direction in DIRECTIONS[args.direction]:
        if direction == "BIDIR":
            results.extend(
                run_bidir(source, expected, src_indices, dst_indices, frag_bytes, args)
            )
            continue
        h2d = direction == "H2D"
        label = (direction, format_size(frag_bytes))
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

        if "cuda" in args.methods:

            def cuda_copy():
                slice_copy(src_tensor, dst_tensor, src_indices, dst_indices, frag_elements)

            clear_destination()
            elapsed = measure(
                cuda_copy,
                cuda_timing=True,
                warmup=args.warmup,
                iterations=args.iterations,
                label=label + ("CUDA Copy",),
            )
            results.append(Result("CUDA Copy", direction, elapsed, total_bytes, valid()))

        if "batch" in args.methods:
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
                batch_copy(batch_operations)

            clear_destination()
            with torch.cuda.stream(batch_stream):
                elapsed = measure(
                    cuda_3d_batch_copy,
                    cuda_timing=True,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    label=label + ("cudaMemcpy3DBatchAsync",),
                )
            results.append(
                Result("CUDA Copy Batch", direction, elapsed, total_bytes, valid())
            )

        if "triton" in args.methods:
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
                triton_copy,
                cuda_timing=True,
                warmup=args.warmup,
                iterations=args.iterations,
                label=label + ("CUDA Kernel",),
            )
            results.append(Result("CUDA Kernel", direction, elapsed, total_bytes, valid()))

        if "iaxl" in args.methods:
            gpu_fragments = gpu.view(fragments, frag_elements)
            if h2d:
                iaxl_copier = SliceCopier(source_fragments, 0, dst_indices, out=gpu_fragments)
            else:
                iaxl_copier = SliceCopier(
                    gpu_fragments, 0, src_indices,
                    out=host.view(fragments, frag_elements), cpu_index=dst_indices,
                )

            clear_destination()
            elapsed = measure(
                iaxl_copier.copy,
                cuda_timing=False,
                warmup=args.warmup,
                iterations=args.iterations,
                label=label + ("IAXL DSA",),
            )
            iaxl_copier.close()
            results.append(Result("IAXL DSA", direction, elapsed, total_bytes, valid()))

    return results


@dataclass
class BidirPass:
    """One direction of a bidir run: issue() enqueues a pass, wait() blocks on its handle."""

    issue: Callable[[], Any]
    wait: Callable[[Any], None]
    lookahead: int = BIDIR_LOOKAHEAD
    stream: torch.cuda.Stream | None = None  # made current on the direction's thread


def stream_pass(run: Callable[[], None], stream: torch.cuda.Stream) -> BidirPass:
    def issue() -> torch.cuda.Event:
        run()
        event = torch.cuda.Event()
        event.record(stream)
        return event

    return BidirPass(issue, lambda event: event.synchronize(), stream=stream)


def measure_bidir(
    passes: tuple[BidirPass, BidirPass],
    *,
    seconds: float,
    warmup: int,
    label: tuple[str, ...] = (),
) -> list[tuple[float, int]]:
    """Run both passes concurrently for `seconds`, each direction on its own thread.

    Returns (milliseconds, passes completed) per direction, so each direction's
    bandwidth is measured while the other is active.
    """
    barrier = threading.Barrier(len(passes) + 1)
    timings: list[tuple[float, int]] = [(0.0, 0)] * len(passes)
    errors: list[BaseException] = []

    def worker(index: int, bidir_pass: BidirPass) -> None:
        pending: deque[Any] = deque()
        count = 0
        try:
            with (
                torch.cuda.stream(bidir_pass.stream)
                if bidir_pass.stream is not None
                else contextlib.nullcontext()
            ):
                for _ in range(warmup):
                    bidir_pass.wait(bidir_pass.issue())
                barrier.wait()
                start = time.perf_counter()
                deadline = start + seconds
                while time.perf_counter() < deadline:
                    if len(pending) >= bidir_pass.lookahead:
                        bidir_pass.wait(pending.popleft())
                    pending.append(bidir_pass.issue())
                    count += 1
                while pending:
                    bidir_pass.wait(pending.popleft())
                timings[index] = ((time.perf_counter() - start) * 1000, count)
        except BaseException as error:  # noqa: BLE001 - re-raised on the main thread
            errors.append(error)
            barrier.abort()

    threads = [
        threading.Thread(target=worker, args=(index, bidir_pass), daemon=True)
        for index, bidir_pass in enumerate(passes)
    ]
    for thread in threads:
        thread.start()
    if RECORDER is not None:
        RECORDER.start(label)
    if SYNC is not None:
        SYNC()
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass
    for thread in threads:
        thread.join()
    if RECORDER is not None:
        RECORDER.stop()
    if errors:
        raise errors[0]
    return timings


def run_bidir(
    source: torch.Tensor,
    expected: torch.Tensor,
    src_indices: list[int],
    dst_indices: list[int],
    frag_bytes: int,
    args: argparse.Namespace,
) -> list[Result]:
    frag_elements = frag_bytes // 2
    fragments = len(src_indices)
    pass_bytes = source.nbytes
    label = ("BIDIR", format_size(frag_bytes))
    host_src = source.clone().pin_memory()
    host_dst = torch.empty(source.numel(), dtype=torch.int16, pin_memory=True)
    gpu_src = source.cuda()
    gpu_dst = torch.empty_like(gpu_src)
    torch.cuda.synchronize()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())

    def clear_destinations():
        gpu_dst.zero_()
        host_dst.zero_()
        torch.cuda.synchronize()

    def valid() -> list[bool]:
        torch.cuda.synchronize()
        return [torch.equal(gpu_dst.cpu(), expected), torch.equal(host_dst, expected)]

    def record(method: str, timings: list[tuple[float, int]]) -> list[Result]:
        checks = valid()
        return [
            Result(method, direction, elapsed, count * pass_bytes, ok)
            for direction, (elapsed, count), ok in zip(("BIDIR H2D", "BIDIR D2H"), timings, checks)
        ]

    results = []
    if "cuda" in args.methods:
        clear_destinations()
        timings = measure_bidir(
            (
                stream_pass(
                    lambda: slice_copy(host_src, gpu_dst, src_indices, dst_indices, frag_elements),
                    streams[0],
                ),
                stream_pass(
                    lambda: slice_copy(gpu_src, host_dst, src_indices, dst_indices, frag_elements),
                    streams[1],
                ),
            ),
            seconds=args.bidir_seconds,
            warmup=args.warmup,
            label=label + ("CUDA Copy",),
        )
        results.extend(record("CUDA Copy", timings))

    if "batch" in args.methods:
        src_byte_offsets = [index * frag_bytes for index in src_indices]
        dst_byte_offsets = [index * frag_bytes for index in dst_indices]
        h2d_operations = build_3d_batch_ops(
            host_src.data_ptr(), gpu_dst.data_ptr(), src_byte_offsets, dst_byte_offsets, frag_bytes
        )
        d2h_operations = build_3d_batch_ops(
            gpu_src.data_ptr(), host_dst.data_ptr(), src_byte_offsets, dst_byte_offsets, frag_bytes
        )

        clear_destinations()
        timings = measure_bidir(
            (
                stream_pass(lambda: batch_copy(h2d_operations), streams[0]),
                stream_pass(lambda: batch_copy(d2h_operations), streams[1]),
            ),
            seconds=args.bidir_seconds,
            warmup=args.warmup,
            label=label + ("cudaMemcpy3DBatchAsync",),
        )
        results.extend(record("CUDA Copy Batch", timings))

    if "iaxl" in args.methods:
        h2d_copier = SliceCopier(
            host_src.view(fragments, frag_elements), 0, dst_indices,
            out=gpu_dst.view(fragments, frag_elements),
        )
        d2h_copier = SliceCopier(
            gpu_src.view(fragments, frag_elements), 0, src_indices,
            out=host_dst.view(fragments, frag_elements), cpu_index=dst_indices,
        )

        # A SliceCopier tracks one transfer at a time, so no lookahead.
        def dsa_pass(copier: SliceCopier) -> BidirPass:
            return BidirPass(copier.start, lambda _: copier.wait(), lookahead=1)

        clear_destinations()
        timings = measure_bidir(
            (dsa_pass(h2d_copier), dsa_pass(d2h_copier)),
            seconds=args.bidir_seconds,
            warmup=args.warmup,
            label=label + ("IAXL DSA",),
        )
        h2d_copier.close()
        d2h_copier.close()
        results.extend(record("IAXL DSA", timings))

    return results


def plot_results(
    results: dict[str, dict[int, list[Result]]], frag_sizes: list[int], path: str
) -> None:
    directions = [direction for direction in PLOT_DIRECTIONS if direction in results]
    figure, axes = plt.subplots(
        len(directions), 1, figsize=(8, 5 * len(directions)), squeeze=False
    )

    for axis, direction in zip(axes[:, 0], directions):
        by_fragment = results[direction]
        methods = list(METHOD_COLORS)
        for method in methods:
            for variant, (style, marker) in VARIANT_STYLES.items():
                points = [
                    (size, result.gbps)
                    for size in frag_sizes
                    for result in by_fragment.get(size, [])
                    if result.method == method and result.variant == variant
                ]
                if points:
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker=marker,
                        linestyle=style,
                        color=METHOD_COLORS[method],
                        label=f"{method} {variant}".strip(),
                    )

        axis.axvspan(4 << 10, 64 << 10, color="gold", alpha=0.15, label="KV <= 64K")
        axis.set_xscale("log", base=2)
        axis.set_xticks(frag_sizes)
        axis.set_xticklabels([format_size(size) for size in frag_sizes], rotation=45)
        axis.set_xlabel("Fragment size")
        axis.set_ylabel("GB/s")
        axis.set_title(
            f"{direction} (concurrent with the other direction)"
            if direction.startswith("BIDIR")
            else direction
        )
        axis.grid(True, which="both", linestyle=":", alpha=0.5)
        axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare fragmented H2D/D2H copies using CUDA, Triton, and IAXL DSA."
    )
    parser.add_argument("--total-gib", type=float, default=0.1)
    parser.add_argument("--frag-sizes", type=parse_sizes, default=DEFAULT_FRAG_SIZES)
    parser.add_argument(
        "--direction",
        choices=tuple(DIRECTIONS),
        default="both",
        help="bidir runs H2D and D2H concurrently (only "
        f"{', '.join(BIDIR_METHODS)}); all runs h2d, d2h and bidir",
    )
    parser.add_argument(
        "--bidir-seconds",
        type=float,
        default=1.0,
        metavar="SEC",
        help="how long both directions run concurrently in bidir mode; the ms column "
        "then shows this window and GB/s is per direction",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        type=parse_methods,
        metavar="NAME",
        help=f"transfer methods to run, space or comma separated: {', '.join(METHODS)}",
    )
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        metavar="PATH",
        help="directory for the plot, flame graph and perf data",
    )
    parser.add_argument(
        "--plot",
        default="tensor_xfer.png",
        metavar="PATH",
        help="plot file name inside the output directory; empty disables plotting",
    )
    parser.add_argument(
        "--flamegraph",
        default="",
        metavar="PATH",
        help="record the timed iterations (warmup excluded) with perf and write this "
        "flame graph SVG into the output directory, together with .folded and perf.data",
    )
    parser.add_argument(
        "--flamegraph-freq",
        type=int,
        default=999,
        metavar="HZ",
        help="perf sampling frequency",
    )
    parser.add_argument(
        "--flamegraph-call-graph",
        choices=("fp", "dwarf", "lbr"),
        default="fp",
        help="perf call-graph unwinding mode",
    )
    return parser


def parse_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> argparse.Namespace:
    args = parser.parse_args(argv)
    if isinstance(args.frag_sizes, str):
        args.frag_sizes = parse_sizes(args.frag_sizes)
    args.methods = [name for group in args.methods for name in group] if args.methods else list(METHODS)
    if args.direction == "bidir" and not any(name in BIDIR_METHODS for name in args.methods):
        parser.error(f"--direction bidir supports only: {', '.join(BIDIR_METHODS)}")
    if not torch.cuda.is_available():
        parser.error("CUDA is not available")
#    if os.environ.get("IAXL_DSA_GD_ENABLE", "0").lower() not in ("1", "true", "yes", "on"):
#        parser.error("set IAXL_DSA_GD_ENABLE=1 to benchmark IAXL DSA")
    return args


def setup(args: argparse.Namespace) -> None:
    global RECORDER
    torch.manual_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.flamegraph:
        RECORDER = PerfRecorder(
            output_path(args.output_dir, "perf.data"),
            frequency=args.flamegraph_freq,
            call_graph=args.flamegraph_call_graph,
        )


def print_header() -> None:
    print(f"{'Direction':9} {'Fragment':>10} {'Method':24} {'ms':>10} {'GB/s':>10} Check")


def print_result(frag_bytes: int, result: Result, extra: str = "") -> None:
    print(
        f"{result.direction:9} {frag_bytes / 1024:9g}K "
        f"{result.method:24} {result.milliseconds:10.3f} "
        f"{result.gbps:10.2f} {'PASS' if result.valid else 'FAIL'}{extra}"
    )


def run_all(
    args: argparse.Namespace, on_fragment: Callable[[int, list[Result]], None]
) -> dict[str, dict[int, list[Result]]]:
    all_results: dict[str, dict[int, list[Result]]] = {}
    for frag_bytes in args.frag_sizes:
        fragment_results = run(frag_bytes, args)
        for result in fragment_results:
            all_results.setdefault(result.direction, {}).setdefault(frag_bytes, []).append(result)
        on_fragment(frag_bytes, fragment_results)
    return all_results


def finish(args: argparse.Namespace, all_results: dict[str, dict[int, list[Result]]]) -> None:
    if args.plot:
        plot_results(all_results, args.frag_sizes, output_path(args.output_dir, args.plot))
    if RECORDER is not None:
        svg = output_path(args.output_dir, args.flamegraph)
        folded = f"{os.path.splitext(svg)[0]}.folded"
        RECORDER.write_folded(folded)
        RECORDER.write_svg(svg, title="tensor_xfer measured iterations")
        print(f"Flame graph saved to {svg} ({RECORDER.samples} samples)")
        print(f"Folded stacks saved to {folded}")
        print(f"perf data kept at {RECORDER.data_path}")


def main() -> None:
    parser = build_parser()
    args = parse_args(parser)
    setup(args)
    print_header()

    def on_fragment(frag_bytes: int, results: list[Result]) -> None:
        for result in results:
            print_result(frag_bytes, result)

    finish(args, run_all(args, on_fragment))


if __name__ == "__main__":
    main()
