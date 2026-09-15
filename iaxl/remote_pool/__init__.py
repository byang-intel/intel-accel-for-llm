# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Remote memory-pool service: a daemon mirrors a client's GPU KV cache in pinned
CPU memory on another node and moves blocks over RDMA (NIXL/UCX).

  nixl_impl  rdma_xfer - thin NIXL agent wrapper (register / connect / xfer / notif)
  rpc        KVClient (client stubs) and KVService (daemon handlers) over NIXL notifs
  daemon     `python3 -m iaxl.remote_pool.daemon`
  client     `python3 -m iaxl.remote_pool.client` - end-to-end put/get verification
"""

from .nixl_impl import DEFAULT_PORT, RDMA_NIC, rdma_xfer
from .rpc import KVClient, KVService

__all__ = ["DEFAULT_PORT", "RDMA_NIC", "rdma_xfer", "KVClient", "KVService"]
