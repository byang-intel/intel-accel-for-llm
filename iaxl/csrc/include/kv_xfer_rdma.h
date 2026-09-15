// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Daemon-side RDMA backend control API (DEVICE=rdma). One nixlAgent per process;
// the data plane (kv_xfer.h) reads/writes the client's GPU memory through it.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace kv_xfer {

void rdma_init(const std::string &name, int listen_port);
void rdma_wait_peer(const std::string &peer, double timeout_s);
void rdma_remove_peer(const std::string &peer);

// Register a pinned DRAM buffer so peers can RDMA into it (RPC control buffer).
void rdma_register_mem(uintptr_t base, size_t bytes);

// Register the ScratchPool; the local descriptor list is built lazily from the
// first context geometry (outer_dims x inner_size sub-descriptors per block).
void rdma_register_local(uintptr_t base, size_t bytes, int64_t block_bytes);

// Prep a remote (VRAM) descriptor list for a client kv_cache tensor laid out
// contiguously as `shape`; desc[o * num_blocks + b] covers block b of outer dim o.
// Returns `base`, which is the gpu_base_ptr to pass to context_create.
uintptr_t rdma_register_remote(const std::string &peer, uintptr_t base,
                               const std::vector<int64_t> &shape, int64_t elem_size, int dev_id,
                               int chunk_dim);
void rdma_unregister_remote(uintptr_t base);

void rdma_send_notif(const std::string &peer, const std::string &msg);
std::vector<std::pair<std::string, std::string>> rdma_get_notifs();

} // namespace kv_xfer
