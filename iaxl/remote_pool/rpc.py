"""KV-block RPC on top of `rdma_xfer`. Control messages ride NIXL notifications
(UCX active messages); bulk data moves by RDMA READ/WRITE between the client's
GPU blocks and the daemon's pinned-CPU mirror, initiated by the daemon.

  KVClient   client-side stubs: register_kv_caches / put / get / put_wait / get_wait / checksum / unregister
  KVService  daemon-side implementation of the same methods, driven by serve()
"""

import json
import time

import torch

from .nixl_impl import block_descs, desc_indices, rdma_xfer


class RpcChannel:
    """Request/response RPC to one peer, carried by NIXL notifications."""

    def __init__(self, xfer: rdma_xfer, peer: str):
        self.xfer, self.peer, self.seq = xfer, peer, 0

    def call(self, method: str, **args):
        self.seq += 1
        self.xfer.send_notif(self.peer, json.dumps({"id": self.seq, "m": method, "args": args}).encode())
        while True:
            for peer, msg in self.xfer.iter_notifs():
                r = json.loads(msg)
                if peer == self.peer and r.get("id") == self.seq:
                    if "error" in r:
                        raise RuntimeError(f"{method}: {r['error']}")
                    return r.get("result")
            time.sleep(1e-5)


def serve(xfer: rdma_xfer, handler, idle_sleep: float = 1e-5):
    """Dispatch incoming requests to handler.<method>(peer, **args) forever."""
    while True:
        for peer, msg in xfer.iter_notifs():
            req = json.loads(msg)
            try:
                resp = {"id": req["id"], "result": getattr(handler, req["m"])(peer, **req.get("args", {}))}
            except Exception as e:  # report to caller instead of killing the daemon
                resp = {"id": req.get("id"), "error": repr(e)}
            xfer.send_notif(peer, json.dumps(resp).encode())
        time.sleep(idle_sleep)


class KVClient:
    """Client side: owns the GPU KV-cache registration and the daemon connection."""

    def __init__(self, xfer: rdma_xfer, daemon_ip: str, daemon_port: int, peer: str = "daemon"):
        self.xfer, self.peer = xfer, peer
        self.daemon_ip, self.daemon_port = daemon_ip, daemon_port
        self.rpc = RpcChannel(xfer, peer)
        self.reg = None

    def register_kv_caches(self, kv: torch.Tensor) -> dict:
        """Register `kv` ([2, num_blocks, ...], contiguous GPU tensor) for RDMA and
        have the daemon allocate its CPU mirror. Must precede put/get."""
        self.reg = self.xfer.register_memory(kv)  # before connect: rkeys ride along with our metadata
        self.xfer.connect(self.peer, self.daemon_ip, self.daemon_port)
        return self.rpc.call("register", base=kv.data_ptr(), shape=list(kv.shape),
                             dtype=str(kv.dtype).split(".")[-1], dev_id=kv.get_device())

    def put(self, blocks) -> int:
        """Start GPU -> daemon copy of `blocks`; returns a job id for put_wait."""
        return self.rpc.call("put", blocks=list(blocks))

    def get(self, blocks) -> int:
        """Start daemon -> GPU copy of `blocks`; returns a job id for get_wait."""
        return self.rpc.call("get", blocks=list(blocks))

    def put_wait(self, job: int) -> dict:
        """Block until the put job completes; returns {seconds, GBps, bytes}."""
        return self.rpc.call("put_wait", job=job)

    def get_wait(self, job: int) -> dict:
        return self.rpc.call("get_wait", job=job)

    def checksum(self, blocks) -> float:
        return self.rpc.call("checksum", blocks=list(blocks))

    def unregister(self):
        self.rpc.call("unregister")
        self.xfer.deregister_memory(self.reg)
        self.xfer.disconnect(self.peer)
        self.reg = None


class KVService:
    """Daemon side: mirrors each peer's GPU KV cache in pinned CPU memory and
    moves blocks with RDMA READ (put) / WRITE (get)."""

    def __init__(self, xfer: rdma_xfer):
        self.xfer = xfer
        self.peers = {}  # peer -> state dict

    def register(self, peer, base, shape, dtype, dev_id):
        self.xfer.wait_peer(peer)  # client's metadata arrives via the listen thread
        shape = tuple(shape)
        cpu = torch.empty(shape, dtype=getattr(torch, dtype)).pin_memory()
        remote_descs, blk = block_descs(base, shape, cpu.element_size(), dev_id)
        local_descs, _ = block_descs(cpu.data_ptr(), shape, cpu.element_size(), 0)
        self.peers[peer] = dict(
            cpu=cpu,
            reg=self.xfer.register_memory(cpu),
            local_h=self.xfer.prep_dlist(local_descs, "DRAM"),
            remote_h=self.xfer.prep_dlist(remote_descs, "VRAM", peer),
            num_blocks=shape[1], blk=blk, jobs={}, next_job=0,
        )
        print(f"[daemon] {peer}: registered {shape} {dtype}, {len(local_descs)} descs x {blk} B")
        return {"descs": len(local_descs), "block_bytes": blk}

    def _xfer(self, peer, op, blocks):
        st = self.peers[peer]
        idx = desc_indices(st["num_blocks"], blocks)
        h = self.xfer.start_xfer(op, st["local_h"], idx, st["remote_h"], idx)
        job = st["next_job"]
        st["next_job"] += 1
        st["jobs"][job] = (h, time.perf_counter(), len(idx) * st["blk"])
        return job

    def _wait(self, peer, job):
        h, t0, nbytes = self.peers[peer]["jobs"].pop(job)
        self.xfer.wait_xfer(h)
        self.xfer.release_xfer(h)
        el = time.perf_counter() - t0
        return {"seconds": el, "GBps": nbytes / el / 1e9, "bytes": nbytes}

    def put(self, peer, blocks):
        return self._xfer(peer, "READ", blocks)

    def get(self, peer, blocks):
        return self._xfer(peer, "WRITE", blocks)

    put_wait = _wait
    get_wait = _wait

    def checksum(self, peer, blocks):
        return float(self.peers[peer]["cpu"][:, blocks].double().sum().item())

    def unregister(self, peer):
        st = self.peers.pop(peer)
        self.xfer.release_dlist(st["local_h"])
        self.xfer.release_dlist(st["remote_h"])
        self.xfer.deregister_memory(st["reg"])
        # keep the peer connection: the RPC reply still has to reach it
        print(f"[daemon] {peer}: unregistered")
        return True
