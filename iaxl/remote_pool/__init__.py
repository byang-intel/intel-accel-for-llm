# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Remote memory pool: the whole KVStore (KVFlow, scratch pool, compression, DRAM
pool, persistence) runs in a daemon on another node; vLLM workers keep only the
`KVStoreRemote` shell (iaxl.kvstore.kvstore_remote). The daemon moves KV blocks
itself over RDMA (NIXL/UCX): put = RDMA READ of client VRAM, get = RDMA WRITE
into client VRAM.

  nixl_impl       rdma_xfer / rdma_xfer_cpp - NIXL agent wrappers (register / connect / notif)
  rpc             RpcChannel (client), KVStoreService + serve (daemon), wire codec
  daemon          `python3 -m iaxl.remote_pool.daemon [--ip IP] [--port P] [--tp-size N]`
"""

from .nixl_impl import DEFAULT_PORT, rdma_xfer, rdma_xfer_cpp
from .rpc import KVStoreService, rank_port, serve

__all__ = ["DEFAULT_PORT", "rdma_xfer", "rdma_xfer_cpp", "KVStoreService", "rank_port", "serve"]
