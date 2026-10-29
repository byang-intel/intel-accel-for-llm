# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import math
import os
import threading
import time
import torch
import numpy as np
from typing import Dict, List, Mapping, Optional, Set, TYPE_CHECKING, Union, cast
import psutil
from ..envs import envs
from ..kvflow import KVFlow, Task, get_accelerator_device
from ..remote_pool.remote_tensor import RemoteTensor
from ..utils.profiler import (
    profile_scope,
    profile_cross_scope,
    profile_func,
    start_profiling,
    stop_profiling,
)
from .api import (
    ControllerServer,
    register as mgmt_register,
    start_mgmt_server,
    stop_mgmt_server,
)

logger = logging.getLogger(__name__)


def _get_default_cache_size_gb() -> float:
    available_bytes = psutil.virtual_memory().available
    return available_bytes / (10 * 1024**3)


def _abort_on_error(fn):
    # An exception would only end the evict thread, leaving ranks diverged and has() possibly stuck.
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BaseException:
            logger.critical(
                "DDR pool evict failed, ranks may diverge; aborting", exc_info=True
            )
            os.abort()

    return wrapper


class KVStoreLocal:
    LABEL = "kv"

    def __init__(
        self,
        model_name: str,
        block_dim: Optional[int] = None,
        kv_caches: Optional[Mapping[str, Union[torch.Tensor, RemoteTensor]]] = None,
        layer_names: Optional[List[str]] = None,
        rank: int = 0,
        tp_size: int = 1,
    ):

        if kv_caches is None and layer_names is None:
            raise ValueError(
                "At least one of kv_caches or layer_names must be provided"
            )

        if kv_caches is not None and block_dim is None:
            raise ValueError("block_dim is required when kv_caches is provided")

        if kv_caches is not None and layer_names is not None:
            kv_keys = set(kv_caches.keys())
            layer_set = set(layer_names)
            if kv_keys != layer_set:
                raise ValueError(
                    f"kv_caches keys {kv_keys} must match layer_names {layer_set}"
                )

        self.kv_caches = kv_caches
        self.block_dim = block_dim
        self.rank = rank
        self.tp_size = tp_size

        if kv_caches is not None:
            self.layer_names = list(kv_caches.keys())
        else:
            self.layer_names = layer_names
        # Layer index of each layer: the sort key of its load tasks.
        self._layer_order: Dict[str, int] = {}
        for index, name in enumerate(self.layer_names):
            self._layer_order[name] = index

        self.skip_compression_count = min(
            envs.IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS, len(self.layer_names)
        )

        self.has_only_mode = kv_caches is None

        if kv_caches:
            first_tensor = next(iter(kv_caches.values()))
            self.kvcache_shape = list(first_tensor.shape)

            self.block_shape = list(first_tensor.shape)
            self.block_shape[block_dim] = 1
            self.block_shape = tuple(
                self.block_shape[i]
                for i in range(len(self.block_shape))
                if i != block_dim
            )
        else:
            self.kvcache_shape = None
            self.block_shape = None

        # Cache directory keyed by the caller-supplied rank (KVStore is DP-agnostic).
        final_persist_dir = f"{model_name}_rank{rank}"

        if self.has_only_mode:
            pool_size_gb = 0.0
        else:
            env_size = envs.IAXL_DDR_POOL_SIZE_GB
            if env_size is not None:
                pool_size_gb = env_size
            else:
                pool_size_gb = _get_default_cache_size_gb()

        self.tensorzip = KVFlow(
            persist_dir=final_persist_dir,
            cache_size_gb=pool_size_gb,
            rank=rank,
        )
        start_profiling()

        logger.info(
            "KVStore initialized successfully: "
            "model_name=%s, rank=%d, has_only_mode=%s, "
            "num_layers=%d, block_dim=%s, block_shape=%s, "
            "pool_size_gb=%.2f, persist_dir=%s",
            model_name,
            self.rank,
            self.has_only_mode,
            len(self.layer_names),
            self.block_dim,
            self.block_shape,
            pool_size_gb,
            final_persist_dir,
        )

        role = "controller" if self.has_only_mode else "worker"
        self._evict_cond = threading.Condition()  # has() vs. the scheduler's evict decision
        self._evicting: Set[str] = set()  # group keys the workers are still dropping
        self._evict_enabled = True  # auto-evict switch, POST /v1/cache/evict
        if role == "worker":
            mgmt_register("GET", "/v1/cache/status", lambda params, s=self: s.status())
            mgmt_register(
                "GET",
                "/v1/health",
                lambda params, s=self: {"status": "ok", "rank": s.rank},
            )
            mgmt_register(
                "POST",
                "/v1/cache/persist",
                lambda body, s=self: {
                    "rank": s.rank,
                    "result": s.persist(int(body.get("count", 10))),
                },
            )

            def evict_labels(body, s=self):
                s.tensorzip.mem.evict_labels(body["labels"])  # raises -> HTTP 500 -> scheduler aborts
                return {"rank": s.rank}

            mgmt_register("POST", "/v1/cache/evict_labels", evict_labels)
            mgmt_register(
                "GET", "/v1/cache/metrics", lambda params, s=self: s.metrics(params)
            )
        self._mgmt_server = start_mgmt_server(
            role=role, rank=self.rank, num_workers=self.tp_size
        )
        if role == "controller":
            self._controller = cast(ControllerServer, self._mgmt_server)
            # Workers start with an empty DDR pool: every row already in the DB is disk-only.
            self._evicted_until = time.time()

            def set_auto_evict(body, s=self):
                enable = body.get("enable")
                if enable is not None:
                    if not isinstance(enable, bool):
                        raise ValueError(
                            f"enable must be true or false, got {enable!r}"
                        )
                    with s._evict_cond:
                        s._evict_enabled = enable
                        s._evict_cond.wait_for(lambda: not s._evicting)  # off: in-flight eviction done
                    logger.info("DDR pool auto evict %s", "on" if enable else "off")
                return {"auto_evict": s._evict_enabled}

            self._controller.post_routes["/v1/cache/evict"] = set_auto_evict
            self._evict_stop = threading.Event()
            threading.Thread(
                target=self._evict_loop, name="iaxl-evict", daemon=True
            ).start()

    def put(
        self,
        block_indices: List[int],
        block_hashs: List[str],
        layer_names: Optional[List[str]] = None,
        description: str = "",
        label: Optional[str] = None,
    ) -> Dict[str, Task]:
        # A non-default label is its own namespace: each block hash gets one
        # more EntryGroup and record row.

        if self.has_only_mode:
            raise RuntimeError(
                "put() not available in has-only mode (kv_caches not provided)"
            )

        if layer_names is None:
            layer_names = self.layer_names

        tensors = {name: self.kv_caches[name] for name in layer_names}

        # skip_compression_count counts GLOBAL layers, but put() may be called with
        # a single layer / subset (layerwise async), where flow compares against the
        # per-call tensor index. Rebase onto this call's first layer so "skip the
        # first N layers" holds regardless of how puts are batched.
        base = self.layer_names.index(layer_names[0])
        local_skip = max(0, self.skip_compression_count - base)

        result = self.tensorzip.put(
            label=label or self.LABEL,
            tensors=tensors,
            chunk_dim=self.block_dim,
            chunk_indices=block_indices,
            chunk_labels=block_hashs,
            description=description,
            skip_compression_count=local_skip,
        )

        if label is not None or self.layer_names[-1] in layer_names:
            self.tensorzip.put_finish(label or self.LABEL, block_hashs)
            self.tensorzip.record_flush()

        return result

    def put_wait(
        self,
        put_results: Dict[str, Task],
        layer_names: Optional[List[str]] = None,
        wait: bool = True,
    ) -> bool:

        if self.has_only_mode:
            raise RuntimeError(
                "put_wait() not available in has-only mode (kv_caches not provided)"
            )

        return self.tensorzip.put_wait(
            put_results,
            tensor_dict_keys=layer_names,
            wait=wait,
        )

    def get(
        self,
        block_indices: List[int],
        block_hashs: List[str],
        layer_names: Optional[List[str]] = None,
        description: str = "",
        label: Optional[str] = None,
        priorities: Optional[Dict[str, int]] = None,
        bundle_id: int = 0,
    ) -> Dict[str, Task]:
        # priorities: OMP-Main level per layer name (None: 0); bundle_id: for set_bundles_priority.
        if self.has_only_mode:
            raise RuntimeError(
                "get() not available in has-only mode (kv_caches not provided)"
            )

        if layer_names is None:
            layer_names = self.layer_names

        tensors = {name: self.kv_caches[name] for name in layer_names}
        sort_keys: Dict[str, int] = {}
        for name in layer_names:
            sort_keys[name] = self._layer_order[name]

        return self.tensorzip.get(
            label=label or self.LABEL,
            tensors=tensors,
            chunk_dim=self.block_dim,
            chunk_indices=block_indices,
            chunk_labels=block_hashs,
            description=description,
            priorities=priorities,
            bundle_id=bundle_id,
            sort_keys=sort_keys,
        )

    def set_bundles_priority(self, bundle_ids: List[int], priority: int) -> int:
        # Moves waiting load tasks of these bundles from less urgent levels to `priority`.
        return self.tensorzip.set_bundles_priority(bundle_ids, priority)

    def sort_priority_level(self, priority: int) -> None:
        # Sorts the waiting load tasks of level `priority` by layer.
        self.tensorzip.sort_priority_level(priority)

    def get_wait(
        self,
        get_results: Dict[str, Task],
        layer_names: Optional[List[str]] = None,
        wait: bool = True,
    ) -> bool:

        if self.has_only_mode:
            raise RuntimeError(
                "get_wait() not available in has-only mode (kv_caches not provided)"
            )

        return self.tensorzip.get_wait(
            get_results,
            tensor_dict_keys=layer_names,
            wait=wait,
        )

    def has(self, block_hashs: Optional[List[str]] = None,
            label: Optional[str] = None,
            truncate: bool = True) -> List[bool]:
        if not block_hashs:
            self.tensorzip.record_flush()
            return []

        label = label or self.LABEL
        with self._evict_cond:
            if self._evicting:
                keys = {f"{label}:{h}" for h in block_hashs}  # group keys, as Record.evict returns
                self._evict_cond.wait_for(lambda: self._evicting.isdisjoint(keys))
            results = self.tensorzip.has(label=label, chunk_labels=block_hashs)
            if self.has_only_mode:
                # Every hit, including those past the truncation point that the connector re-saves.
                self.tensorzip.record.touch(label, block_hashs)
        if not truncate:
            return results

        mask = np.array(results, dtype=np.bool_)
        idx = np.argmin(mask)
        if not mask[idx]:
            mask[idx + 1 :] = False
            results = mask.tolist()

        return results

    def stop(self):
        if self.has_only_mode:
            self._evict_stop.set()
        stop_mgmt_server(self)
        self.tensorzip.stop()

    def evict_lru(self, max_count: int) -> None:
        with self._evict_cond:
            self._evict_cond.wait_for(lambda: not self._evicting)  # one eviction at a time
            if not self._evict_enabled:  # switched off after _evict_loop checked
                return
            labels, self._evicted_until = self.tensorzip.record.evict(
                self._evicted_until,
                time.time() - envs.IAXL_DDR_POOL_EVICT_LEASE_S,
                max_count,
            )
            if not labels:
                return
            # has() on these waits until every rank has dropped them.
            self._evicting.update(labels)
        workers = self._controller.broadcast(
            lambda c: c.post("/v1/cache/evict_labels", {"labels": labels})
        )
        failed = {r: w for r, w in workers.items() if "error" in w}
        if failed:
            raise RuntimeError(f"evict_labels failed: {failed}")
        with self._evict_cond:
            self._evicting.clear()
            self._evict_cond.notify_all()

    @_abort_on_error
    def _evict_loop(self) -> None:
        while not self._evict_stop.wait(envs.IAXL_DDR_POOL_EVICT_INTERVAL_S):
            if not self._evict_enabled:
                continue
            st = self._controller.broadcast(
                lambda c: c.get("/v1/cache/status"), ranks=[0]
            )["0"]
            if "error" in st:
                raise RuntimeError(f"DDR pool status failed: {st}")
            if not st["capacity_bytes"]:  # 0 = unlimited pool
                continue
            usage = st["current_bytes"] / st["capacity_bytes"]
            if usage >= envs.IAXL_DDR_POOL_EVICT_HIGH_WATERMARK:
                # Groups are about the same size, so evicting this share lands near the low watermark.
                share = 1 - envs.IAXL_DDR_POOL_EVICT_LOW_WATERMARK / usage
                max_count = math.ceil(st["group_count"] * share)
                logger.info(
                    "DDR pool usage %.2f, evicting up to %d LRU groups", usage, max_count
                )
                self.evict_lru(max_count)

    def status(self) -> dict:
        status = self.tensorzip.status()
        status["rank"] = self.rank
        status["num_layers"] = len(self.layer_names)
        status["kvcache_shape"] = self.kvcache_shape
        return status

    def metrics(self, params: Optional[dict] = None) -> dict:
        from ..torch_ext import (
            metrics_set_enabled,
            metrics_reset,
            metrics_read,
        )

        params = params or {}
        if "enable" in params:
            metrics_set_enabled(
                str(params["enable"]).lower() in ("1", "true", "yes", "on")
            )
        if str(params.get("reset", "")).lower() in ("1", "true", "yes", "on"):
            metrics_reset()

        result = metrics_read()
        result["rank"] = self.rank
        return result

    def persist(self, max_count: int) -> dict:
        if self.has_only_mode:
            return {"error": "persist() not available in has-only mode (scheduler)"}
        return self.tensorzip.persist(max_count)

    def evict(self, max_count: int) -> dict:
        if self.has_only_mode:
            return {"error": "evict() not available in has-only mode (scheduler)"}
        return self.tensorzip.evict(max_count)

    def get_persist_candidates(self, max_count: int) -> List[str]:
        return self.tensorzip.get_persist_candidates(max_count)

    def get_evict_candidates(self, max_count: int) -> List[str]:
        return self.tensorzip.get_evict_candidates(max_count)


if TYPE_CHECKING:  # the two shells share one interface; pick the local one for typing
    KVStore = KVStoreLocal
elif envs.IAXL_RDMA_ENABLE:
    from ..remote_pool.kvstore_remote import KVStoreRemote

    KVStore = KVStoreRemote
else:
    KVStore = KVStoreLocal
