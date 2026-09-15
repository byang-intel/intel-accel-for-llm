#!/usr/bin/env python3
"""KV-block store daemon: serves rpc.KVService over NIXL notifications.

  register(base, shape, dtype, dev_id) -> mirrors the client's GPU KV cache as a pinned CPU tensor
  put(blocks)  -> job     RDMA READ  client GPU blocks -> CPU blocks
  get(blocks)  -> job     RDMA WRITE CPU blocks -> client GPU blocks
  put_wait(job) / get_wait(job) -> {seconds, GBps, bytes}
  checksum(blocks) -> float64 sum of the CPU-side blocks
  unregister()

Run:  python3 -m iaxl.remote_pool.daemon [--port 5555]
"""

import argparse

from . import rpc
from .nixl_impl import DEFAULT_PORT, RDMA_NIC, rdma_xfer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    xfer = rdma_xfer("daemon", listen_port=args.port)
    print(f"[daemon] UCX {xfer.ibdev} ({RDMA_NIC}), metadata listener on port {args.port}")
    try:
        rpc.serve(xfer, rpc.KVService(xfer))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
