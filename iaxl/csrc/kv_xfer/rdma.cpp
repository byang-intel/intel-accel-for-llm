// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// kv_xfer backend for the remote_pool daemon: the "GPU" is a client's VRAM
// reached over RDMA through NIXL. put = RDMA READ (d2h), get = RDMA WRITE (h2d).
// Transfers are posted and polled synchronously on the TaskQueue worker, so
// copy_chunks_batch returning means the data has landed; events are plain flags.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <nixl.h>

#include "iaxl_common.h"
#include "kv_xfer.h"
#include "kv_xfer_rdma.h"

#define NIXL_CHECK(call)                                                                           \
    do {                                                                                           \
        nixl_status_t st_ = (call);                                                                \
        if (st_ != NIXL_SUCCESS) {                                                                 \
            fprintf(stderr, "NIXL error %s:%d: %s returned %s\n", __FILE__, __LINE__, #call,       \
                    nixlEnumStrings::statusStr(st_).c_str());                                      \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

namespace kv_xfer {

namespace {

struct RemoteReg {
    std::string peer;
    nixlDlistH *dlist = nullptr;
    int64_t num_blocks = 0;
    int64_t outer_dims = 0;
    int64_t inner_size = 0;
    int64_t chunk_stride = 0;
    int64_t outer_block_size = 0;
};

struct LocalPool {
    uintptr_t base = 0;
    size_t bytes = 0;
    int64_t block_bytes = 0;
    int64_t outer_dims = 0;
    int64_t inner_size = 0;
    nixlDlistH *dlist = nullptr;
};

struct Rdma {
    // Heap-allocated and never freed: destroying nixlAgent during static teardown
    // (after UCX / plugin statics are gone) segfaults; the daemon lives as long as the process.
    static Rdma &get() {
        static Rdma *r = new Rdma;
        return *r;
    }

    nixlAgent &agent() {
        IAXL_CHECK(agent_ != nullptr, "rdma_init must be called first");
        return *agent_;
    }

    std::unique_ptr<nixlAgent> agent_;
    nixl_opt_args_t opt;                      // pins every call to the UCX backend
    std::mutex mu;                            // remotes / pool bookkeeping
    std::unordered_map<uintptr_t, RemoteReg> remotes;
    std::vector<nixl_reg_dlist_t> regs;       // registrations stay alive for the process lifetime
    LocalPool pool;
};

struct XferContext {
    const RemoteReg *reg;
    int64_t chunk_stride, outer_dims, inner_size, outer_block_size;
};

inline XferContext *as_ctx(context_t c) { return static_cast<XferContext *>(c); }
inline std::atomic<bool> *as_ev(event_t e) { return static_cast<std::atomic<bool> *>(e); }

// Build the pool descriptor list once the sub-block geometry is known.
void ensure_pool_dlist(Rdma &r, int64_t outer_dims, int64_t inner_size) {
    std::lock_guard<std::mutex> lock(r.mu);
    LocalPool &p = r.pool;
    IAXL_CHECK(p.base != 0, "rdma_register_local must be called before the first transfer");
    if (p.dlist) {
        IAXL_CHECK(p.outer_dims == outer_dims && p.inner_size == inner_size,
                   "scratch pool geometry differs between layers");
        return;
    }
    IAXL_CHECK(outer_dims * inner_size == p.block_bytes, "pool block_bytes != outer_dims * inner_size");
    const int64_t nblocks = p.bytes / p.block_bytes;
    nixl_xfer_dlist_t descs(DRAM_SEG, nblocks * outer_dims);
    for (int64_t b = 0; b < nblocks; b++)
        for (int64_t o = 0; o < outer_dims; o++)
            descs[b * outer_dims + o] = nixlBasicDesc(p.base + b * p.block_bytes + o * inner_size, inner_size, 0);
    NIXL_CHECK(r.agent().prepXferDlist(NIXL_INIT_AGENT, descs, p.dlist, &r.opt));
    p.outer_dims = outer_dims;
    p.inner_size = inner_size;
    fprintf(stderr, "[kv_xfer/rdma] pool: %ld blocks x %ld descs of %ld B\n", nblocks, outer_dims, inner_size);
}

} // namespace

// ---- kv_xfer_rdma.h --------------------------------------------------------

void rdma_init(const std::string &name, int listen_port) {
    Rdma &r = Rdma::get();
    IAXL_CHECK(r.agent_ == nullptr, "rdma_init called twice");
    // Listen thread serves metadata; STRICT lets the main thread poll notifs while
    // TaskQueue workers post/poll transfers.
    nixlAgentConfig cfg(false, true, listen_port, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT);
    r.agent_ = std::make_unique<nixlAgent>(name, cfg);
    nixl_mem_list_t mems;
    nixl_b_params_t params;
    NIXL_CHECK(r.agent_->getPluginParams("UCX", mems, params));
    nixlBackendH *ucx = nullptr;
    NIXL_CHECK(r.agent_->createBackend("UCX", params, ucx));
    r.opt.backends.push_back(ucx);
    fprintf(stderr, "[kv_xfer/rdma] agent %s listening on port %d\n", name.c_str(), listen_port);
}

void rdma_wait_peer(const std::string &peer, double timeout_s) {
    nixlAgent &a = Rdma::get().agent();
    nixl_xfer_dlist_t empty(DRAM_SEG);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
    while (a.checkRemoteMD(peer, empty) != NIXL_SUCCESS) {
        IAXL_CHECK(std::chrono::steady_clock::now() < deadline, "timed out waiting for peer metadata");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void rdma_remove_peer(const std::string &peer) { NIXL_CHECK(Rdma::get().agent().invalidateRemoteMD(peer)); }

void rdma_register_mem(uintptr_t base, size_t bytes) {
    Rdma &r = Rdma::get();
    nixl_reg_dlist_t regs(DRAM_SEG);
    regs.addDesc(nixlBlobDesc(base, bytes, 0, ""));
    NIXL_CHECK(r.agent().registerMem(regs, &r.opt));
    std::lock_guard<std::mutex> lock(r.mu);
    r.regs.push_back(std::move(regs));
}

void rdma_register_local(uintptr_t base, size_t bytes, int64_t block_bytes) {
    Rdma &r = Rdma::get();
    IAXL_CHECK(r.pool.base == 0, "rdma_register_local called twice");
    rdma_register_mem(base, bytes);
    std::lock_guard<std::mutex> lock(r.mu);
    r.pool.base = base;
    r.pool.bytes = bytes;
    r.pool.block_bytes = block_bytes;
}

uintptr_t rdma_register_remote(const std::string &peer, uintptr_t base, const std::vector<int64_t> &shape,
                               int64_t elem_size, int dev_id, int chunk_dim) {
    Rdma &r = Rdma::get();
    RemoteReg reg;
    reg.peer = peer;
    reg.outer_dims = 1;
    for (int d = 0; d < chunk_dim; d++)
        reg.outer_dims *= shape[d];
    reg.inner_size = elem_size;
    for (size_t d = chunk_dim + 1; d < shape.size(); d++)
        reg.inner_size *= shape[d];
    reg.num_blocks = shape[chunk_dim];
    reg.chunk_stride = reg.inner_size; // contiguous
    reg.outer_block_size = reg.num_blocks * reg.inner_size;

    nixl_xfer_dlist_t descs(VRAM_SEG, reg.outer_dims * reg.num_blocks);
    for (int64_t o = 0; o < reg.outer_dims; o++)
        for (int64_t b = 0; b < reg.num_blocks; b++)
            descs[o * reg.num_blocks + b] =
                nixlBasicDesc(base + o * reg.outer_block_size + b * reg.chunk_stride, reg.inner_size, dev_id);
    NIXL_CHECK(r.agent().prepXferDlist(peer, descs, reg.dlist, &r.opt));

    std::lock_guard<std::mutex> lock(r.mu);
    IAXL_CHECK(r.remotes.emplace(base, std::move(reg)).second, "remote tensor registered twice");
    return base;
}

void rdma_unregister_remote(uintptr_t base) {
    Rdma &r = Rdma::get();
    std::lock_guard<std::mutex> lock(r.mu);
    auto it = r.remotes.find(base);
    IAXL_CHECK(it != r.remotes.end(), "rdma_unregister_remote: unknown base");
    NIXL_CHECK(r.agent().releasedDlistH(it->second.dlist));
    r.remotes.erase(it);
}

void rdma_send_notif(const std::string &peer, const std::string &msg) {
    Rdma &r = Rdma::get();
    NIXL_CHECK(r.agent().genNotif(peer, msg, &r.opt));
}

std::vector<std::pair<std::string, std::string>> rdma_get_notifs() {
    nixl_notifs_t map;
    NIXL_CHECK(Rdma::get().agent().getNotifs(map));
    std::vector<std::pair<std::string, std::string>> out;
    for (auto &[peer, msgs] : map)
        for (auto &m : msgs)
            out.emplace_back(peer, std::move(m));
    return out;
}

// ---- kv_xfer.h ---------------------------------------------------------------

event_t event_acquire() { return new std::atomic<bool>(false); }
void event_release(event_t event) { delete as_ev(event); }
event_t event_create() { return event_acquire(); }
void event_destroy(event_t event) { event_release(event); }

void event_synchronize(event_t event) {
    auto *f = as_ev(event);
    while (!f->load(std::memory_order_acquire))
        std::this_thread::yield();
}

stream_t extract_stream(pybind11::object) { return nullptr; }
event_t wait_stream_from_py(pybind11::object) { return nullptr; }

context_t context_create(char *gpu_base_ptr, int /*device_index*/, int64_t chunk_stride, int64_t outer_dims,
                         int64_t inner_size, int64_t outer_block_size, stream_t /*work_stream*/) {
    Rdma &r = Rdma::get();
    XferContext *x = new XferContext();
    {
        std::lock_guard<std::mutex> lock(r.mu);
        auto it = r.remotes.find(reinterpret_cast<uintptr_t>(gpu_base_ptr));
        IAXL_CHECK(it != r.remotes.end(), "context_create: tensor not registered with rdma_register_remote");
        x->reg = &it->second; // unordered_map nodes are address-stable until erased
    }
    IAXL_CHECK(x->reg->chunk_stride == chunk_stride && x->reg->outer_dims == outer_dims &&
                   x->reg->inner_size == inner_size && x->reg->outer_block_size == outer_block_size,
               "context_create: geometry differs from rdma_register_remote");
    x->chunk_stride = chunk_stride;
    x->outer_dims = outer_dims;
    x->inner_size = inner_size;
    x->outer_block_size = outer_block_size;
    ensure_pool_dlist(r, outer_dims, inner_size);
    return x;
}

void context_destroy(context_t ctx) { delete as_ctx(ctx); }

unsigned long long context_stream_id(context_t) { return 0; }
bool context_same_stream(context_t) { return true; }

void context_record_event(context_t, event_t event) { as_ev(event)->store(true, std::memory_order_release); }
void context_work_wait_event(context_t, event_t event) { event_synchronize(event); }
void context_cur_wait_event(context_t, event_t) {}
void context_work_wait_cur(context_t) {}
void context_sync_cur(context_t) {}

void copy_chunks_batch(context_t ctx, const std::vector<int64_t> &chunk_indices, const std::vector<char *> &cpu_ptrs,
                       bool h2d) {
    Rdma &r = Rdma::get();
    XferContext *x = as_ctx(ctx);
    const RemoteReg &reg = *x->reg;
    const LocalPool &p = r.pool;
    const size_t n = chunk_indices.size();
    if (n == 0)
        return;

    std::vector<int> local_idx(n * x->outer_dims), remote_idx(n * x->outer_dims);
    for (size_t i = 0, k = 0; i < n; i++) {
        const int64_t pool_block = (reinterpret_cast<uintptr_t>(cpu_ptrs[i]) - p.base) / p.block_bytes;
        for (int64_t o = 0; o < x->outer_dims; o++, k++) {
            local_idx[k] = static_cast<int>(pool_block * x->outer_dims + o);
            remote_idx[k] = static_cast<int>(o * reg.num_blocks + chunk_indices[i]);
        }
    }

    static std::once_flag noted;
    std::call_once(noted, [&] {
        fprintf(stderr, "[kv_xfer] copy_chunks_batch: using RDMA backend (%ld B x %ld per block)\n",
                x->inner_size, x->outer_dims);
    });

    nixlAgent &a = r.agent();
    nixlXferReqH *req = nullptr;
    NIXL_CHECK(a.makeXferReq(h2d ? NIXL_WRITE : NIXL_READ, p.dlist, local_idx, reg.dlist, remote_idx, req, &r.opt));
    nixl_status_t st = a.postXferReq(req);
    while (st == NIXL_IN_PROG)
        st = a.getXferStatus(req);
    NIXL_CHECK(st);
    NIXL_CHECK(a.releaseXferReq(req));
}

void copy_chunk(context_t ctx, char *cpu_base, int64_t chunk_index, bool h2d) {
    copy_chunks_batch(ctx, {chunk_index}, {cpu_base}, h2d);
}

} // namespace kv_xfer
