"""NIXL wrapper. `rdma_xfer` owns one NIXL agent (UCX backend pinned to the RDMA
NIC) and exposes memory registration, peer metadata exchange, notifications and
prepped block transfers. It knows nothing about KV caches or RPC (see rpc.py)."""

import math
import os
import time

RDMA_NIC = "ens34f0np0"
DEFAULT_PORT = 5555
LOCAL_AGENT = "NIXL_INIT_AGENT"


def configure_ucx_env(nic: str = RDMA_NIC) -> str:
    """Force UCX onto the RDMA NIC. Must run before `import nixl`."""
    ibdev = sorted(os.listdir(f"/sys/class/net/{nic}/device/infiniband"))[0]
    os.environ.setdefault("UCX_NET_DEVICES", f"{ibdev}:1")
    os.environ.setdefault("UCX_TLS", "rc,cuda_copy,cuda_ipc")
    return ibdev


def block_descs(base: int, shape, elem_size: int, dev_id: int):
    """One (addr, len, dev_id) per (kv, block); desc index = kv * num_blocks + block."""
    kv_count, num_blocks = shape[0], shape[1]
    blk = math.prod(shape[2:]) * elem_size
    return [(base + i * blk, blk, dev_id) for i in range(kv_count * num_blocks)], blk


def desc_indices(num_blocks: int, blocks):
    return [kv * num_blocks + b for kv in (0, 1) for b in blocks]


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


class rdma_xfer:
    """One NIXL agent. `listen_port` enables the metadata listener so peers can
    connect to us; None gives a connect-only agent."""

    def __init__(self, name: str, listen_port: int | None = None, nic: str = RDMA_NIC):
        self.ibdev = configure_ucx_env(nic)  # must precede `import nixl`
        from nixl._api import nixl_agent, nixl_agent_config

        cfg = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=listen_port is not None,
            backends=["UCX"],
            listen_port=listen_port or 0,
        )
        self.name = name
        self.agent = nixl_agent(name, cfg)

    # -- peers ---------------------------------------------------------------
    def connect(self, peer: str, ip: str, port: int, timeout_s: float | None = None):
        """Exchange metadata with a listening peer. Register local memory first:
        rkeys of already-registered buffers ride along with our metadata."""
        self.agent.fetch_remote_metadata(peer, ip, port)
        self.agent.send_local_metadata(ip, port)
        self.wait_peer(peer, timeout_s)

    def wait_peer(self, peer: str, timeout_s: float | None = None):
        t0 = time.perf_counter()
        while not self.agent.check_remote_metadata(peer):
            if timeout_s is not None and time.perf_counter() - t0 > timeout_s:
                raise TimeoutError(f"no metadata from {peer}")
            time.sleep(1e-3)

    def disconnect(self, peer: str):
        self.agent.remove_remote_agent(peer)

    # -- memory --------------------------------------------------------------
    def register_memory(self, tensor):
        return self.agent.register_memory(tensor)

    def deregister_memory(self, handle):
        self.agent.deregister_memory(handle)

    # -- descriptor lists ----------------------------------------------------
    def prep_dlist(self, descs, mem_type: str, peer: str | None = None):
        """mem_type "DRAM" | "VRAM"; peer None means our own (local) side."""
        return self.agent.prep_xfer_dlist(peer or LOCAL_AGENT, descs, mem_type)

    def release_dlist(self, handle):
        self.agent.release_dlist_handle(handle)

    # -- transfers -----------------------------------------------------------
    def start_xfer(self, op: str, local_h, local_idx, remote_h, remote_idx):
        """op "READ" (remote -> local) | "WRITE" (local -> remote). Returns the xfer handle."""
        h = self.agent.make_prepped_xfer(op, local_h, local_idx, remote_h, remote_idx)
        self.agent.transfer(h)
        return h

    def xfer_state(self, handle) -> str:
        return self.agent.check_xfer_state(handle)

    def wait_xfer(self, handle, timeout_s: float = 60.0):
        t0 = time.perf_counter()
        while (state := self.agent.check_xfer_state(handle)) != "DONE":
            if state == "ERR":
                raise RuntimeError("NIXL transfer failed")
            if time.perf_counter() - t0 > timeout_s:
                raise TimeoutError("NIXL transfer timed out")
            time.sleep(1e-5)

    def release_xfer(self, handle):
        self.agent.release_xfer_handle(handle)

    # -- notifications -------------------------------------------------------
    def send_notif(self, peer: str, payload: bytes):
        self.agent.send_notif(peer, payload)

    def iter_notifs(self):
        """Yield (peer_name, payload_bytes) for every pending notification."""
        for peer, msgs in self.agent.get_new_notifs().items():
            for m in msgs:
                yield _s(peer), m
