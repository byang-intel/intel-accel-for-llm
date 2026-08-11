# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import threading
import logging
from typing import Dict, List, Tuple
from ..utils.profiler import profile_scope, profile_cross_scope, profile_func
from ..envs import envs

logger = logging.getLogger(__name__)


class ScratchPool:
    def __init__(self, cache_size_gb: float, min_create_count: int = 2048):
        self._cache_size_gb = cache_size_gb
        self._min_create_count = min_create_count

        prealloc_limit = envs.IAXL_PREALLOC_LIMIT
        if prealloc_limit > 0:
            self._min_create_count = min(self._min_create_count, prealloc_limit)

        self._pools: Dict[Tuple[Tuple[int, ...], torch.dtype], List[torch.Tensor]] = {}
        self._lock = threading.Lock()

        self._total_bytes = 0
        self._total_tensors = 0

        self._allocate_count = 0
        self._release_count = 0

        self._debug = False

        self._pinned_check_warned = False

        logger.info(
            "ScratchPool initialized (on-demand allocation, min_create=%d, debug=%s)",
            self._min_create_count,
            self._debug,
        )

    def _get_pool_key(
        self, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> Tuple[Tuple[int, ...], torch.dtype]:
        return (tuple(shape), dtype)

    def _create_tensor(
        self, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

        if not self._pinned_check_warned:
            logger.warning(
                "ScratchPool._create_tensor: verifying pinned per newly created "
                "tensor here (checks each one individually); this adds a small "
                "one-time cost at allocation/warmup so the DSA transfer path can skip "
                "the per-transfer all_pinned() scan."
            )
            self._pinned_check_warned = True
        assert tensor.is_pinned(), (
            "ScratchPool._create_tensor: tensor is not pinned "
            "(pin_memory=True did not take effect); the DSA backend requires "
            "pinned host memory"
        )

        tensor_bytes = tensor.numel() * tensor.element_size()
        self._total_bytes += tensor_bytes
        self._total_tensors += 1

        return tensor

    @profile_func(lambda self, count, *_: f"(count={count})")
    def allocate(
        self, count: int, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> List[torch.Tensor]:
        with self._lock:
            key = self._get_pool_key(shape, dtype)
            available_before = len(self._pools.get(key, []))

            if key not in self._pools:
                self._pools[key] = []
                available = self._pools[key]

                chunk_numel = 1
                for dim in shape:
                    chunk_numel *= dim
                chunk_bytes = chunk_numel * torch.tensor([], dtype=dtype).element_size()
                preallocate_bytes = int(self._cache_size_gb * 0.1 * 1024**3)
                preallocate_count = max(preallocate_bytes // chunk_bytes, count)
                preallocate_count = min(preallocate_count, self._min_create_count * 100)

                max_alloc = envs.IAXL_PREALLOC_LIMIT
                if max_alloc > 0:
                    preallocate_count = min(preallocate_count, max_alloc)

                logger.info(
                    "ScratchPool: pre-allocating %d tensors for shape=%s, dtype=%s (%.2f MB)",
                    preallocate_count,
                    shape,
                    dtype,
                    preallocate_count * chunk_bytes / (1024**2),
                )

                for _ in range(preallocate_count):
                    tensor = self._create_tensor(shape, dtype)
                    available.append(tensor)
            else:
                available = self._pools[key]

            tensors = []
            available_count = len(available)

            if available_count >= count:
                tensors = available[-count:]
                del available[-count:]
            elif available_count > 0:
                tensors = available[:]
                available.clear()
                need_count = count - available_count

                create_count = max(need_count, self._min_create_count)
                for _ in range(create_count):
                    available.append(self._create_tensor(shape, dtype))

                tensors.extend(available[:need_count])
                del available[:need_count]
            else:
                create_count = max(count, self._min_create_count)
                for _ in range(create_count):
                    available.append(self._create_tensor(shape, dtype))

                tensors = available[:count]
                del available[:count]

            self._allocate_count += count

            if self._debug:
                created_count = count - min(available_before, count)
                available_after = len(available)
                in_flight = self._allocate_count - self._release_count
                logger.info(
                    "ScratchPool.allocate: key=%s, requested=%d, created=%d, available: %d -> %d, in_flight=%d",
                    key,
                    count,
                    created_count,
                    available_before,
                    available_after,
                    in_flight,
                )

            return tensors

    @profile_func(lambda self, tensors, *_: f"(count={len(tensors)})")
    def release(self, tensors: List[torch.Tensor]):
        with self._lock:
            first = tensors[0]
            key = self._get_pool_key(tuple(first.shape), first.dtype)

            assert key in self._pools, (
                f"ScratchPool.release: unknown key={key} (shape={first.shape}, dtype={first.dtype})"
            )

            available_before = len(self._pools[key])
            self._pools[key].extend(tensors)
            available_after = len(self._pools[key])

            self._release_count += len(tensors)

            if self._debug:
                in_flight = self._allocate_count - self._release_count
                logger.info(
                    "ScratchPool.release: key=%s, released=%d, available: %d -> %d, in_flight=%d",
                    key,
                    len(tensors),
                    available_before,
                    available_after,
                    in_flight,
                )

    def available_count(
        self, shape: Tuple[int, ...] = None, dtype: torch.dtype = None
    ) -> int:
        with self._lock:
            if shape is not None and dtype is not None:
                key = self._get_pool_key(shape, dtype)
                return len(self._pools.get(key, []))
            else:
                return sum(len(p) for p in self._pools.values())

    def total_count(
        self, shape: Tuple[int, ...] = None, dtype: torch.dtype = None
    ) -> int:
        with self._lock:
            return self._total_tensors

    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def status(self) -> str:
        with self._lock:
            total_available = sum(len(p) for p in self._pools.values())
            in_use = self._allocate_count - self._release_count
            lines = [
                f"ScratchPool: {self._total_tensors} created, {total_available} available, {in_use} in-use, alloc/release={self._allocate_count}/{self._release_count}, {self._total_bytes / (1024**2):.2f} MB"
            ]
            for key, available in self._pools.items():
                shape, dtype = key
                lines.append(f"  {shape} {dtype}: {len(available)} available")
            return "\n".join(lines)
