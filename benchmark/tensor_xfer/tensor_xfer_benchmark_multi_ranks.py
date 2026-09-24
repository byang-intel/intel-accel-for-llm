#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run tensor_xfer_benchmark.py with one process per TP rank, mirroring vLLM.

Rank r is bound to its CPUs / QAT / DSA via iaxl.utils.affinity (using the
per-rank env from setvars.sh) and gets its own GPU through CUDA_VISIBLE_DEVICES.
Every timed iteration starts simultaneously on all ranks; rank 0 gathers the
per-rank results and reports (and plots) the per-rank average.
--flamegraph only records rank 0.

Example:
    TP_SIZE=2 source setvars.sh
    python3 benchmark/tensor_xfer/tensor_xfer_benchmark_multi_ranks.py
"""

import logging
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPIN_SLACK_S = 0.002  # covers barrier wake-up jitter before the shared start time
logger = logging.getLogger(__name__)


def worker(rank, tp_size, gpu, argv, barrier, deadline, queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # must precede the torch import
    from iaxl.utils.affinity import bind_cpu_affinity, bind_intel_accel
    from iaxl.utils.logger import setup_root_logger

    setup_root_logger(f"rank{rank}")
    bind_cpu_affinity(rank, tp_size, os.getenv("VLLM_CPU_OMP_THREADS_BIND"))
    bind_intel_accel(rank)

    import tensor_xfer_benchmark as bench
    import torch

    logger.info("Bound rank %d to GPU %s (%s)", rank, gpu, torch.cuda.get_device_name(0))

    late = [0, 0.0]  # missed starts, worst lateness in ms

    def sync() -> None:
        # The first barrier waits for every rank to finish the previous iteration, so
        # the deadline only has to cover the second barrier's wake-up jitter.
        barrier.wait()
        if rank == 0:
            deadline.value = time.monotonic() + SPIN_SLACK_S
        barrier.wait()
        target = deadline.value
        now = time.monotonic()
        if now > target:
            late[0] += 1
            late[1] = max(late[1], (now - target) * 1000)
        while time.monotonic() < target:
            pass

    bench.SYNC = sync
    parser = bench.build_parser()
    parser.set_defaults(plot=f"tensor_xfer_tp{tp_size}.png")
    args = bench.parse_args(parser, argv)
    if rank != 0:
        args.flamegraph = ""
    bench.setup(args)

    try:
        if rank != 0:
            bench.run_all(args, lambda frag_bytes, results: queue.put((rank, frag_bytes, results)))
            return

        aggregated: dict[str, dict[int, list[bench.Result]]] = {}

        def on_fragment(frag_bytes: int, results: list[bench.Result]) -> None:
            per_rank = {0: results}
            while len(per_rank) < tp_size:
                other_rank, other_frag, other_results = queue.get()
                assert other_frag == frag_bytes, "ranks out of step"
                per_rank[other_rank] = other_results
            for index, result in enumerate(results):
                group = [per_rank[r][index] for r in range(tp_size)]
                times = [r.milliseconds for r in group]
                rank_bytes = sum(r.total_bytes for r in group) // tp_size
                # Harmonic mean of the times, so GB/s is the plain mean of the ranks'.
                average = bench.Result(
                    result.method,
                    result.direction,
                    tp_size / sum(1 / t for t in times),
                    rank_bytes,
                    all(r.valid for r in group),
                )
                series = [
                    average,
                    # Fastest / slowest rank, i.e. the shortest / longest time.
                    bench.Result(result.method, result.direction, min(times), rank_bytes, True, "max"),
                    bench.Result(result.method, result.direction, max(times), rank_bytes, True, "min"),
                ]
                aggregated.setdefault(result.direction, {}).setdefault(frag_bytes, []).extend(series)
                ranks = " ".join(f"{r.gbps:.2f}" for r in group)
                bench.print_result(frag_bytes, average, f"  [{ranks}]")

        bench.print_header()
        print(f"Average per rank over {tp_size} ranks; per-rank GB/s in brackets")
        bench.run_all(args, on_fragment)
        bench.finish(args, aggregated)
    finally:
        if late[0]:
            logger.warning(
                "rank %d started late %d times (worst %.2f ms); results are less "
                "aligned than intended, raise SPIN_SLACK_S",
                rank,
                late[0],
                late[1],
            )


def main() -> None:
    tp_size = int(os.environ.get("TP_SIZE", "1"))
    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count < tp_size:
        sys.exit(f"TP_SIZE={tp_size} needs {tp_size} GPUs, but only {gpu_count} are visible")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = visible.split(",") if visible else [str(i) for i in range(gpu_count)]

    ctx = multiprocessing.get_context("spawn")  # each rank must initialize CUDA itself
    barrier = ctx.Barrier(tp_size, timeout=600)
    deadline = ctx.Value("d", 0.0, lock=False)
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=worker,
            args=(rank, tp_size, gpus[rank], sys.argv[1:], barrier, deadline, queue),
            name=f"rank{rank}",
        )
        for rank in range(tp_size)
    ]
    for proc in procs:
        proc.start()
    while any(proc.is_alive() for proc in procs):
        if any(proc.exitcode for proc in procs):  # one rank failed: stop the rest
            for proc in procs:
                proc.terminate()
            break
        time.sleep(0.2)
    failed = False
    for proc in procs:
        proc.join()
        if proc.exitcode:
            print(f"{proc.name} exited with code {proc.exitcode}", file=sys.stderr)
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
