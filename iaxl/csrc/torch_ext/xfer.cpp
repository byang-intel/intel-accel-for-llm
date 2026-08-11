// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "context.h"

#include <chrono>
#include <vector>

void Context::xfer_chunk(const torch::Tensor &cpu_tensor, int64_t chunk_idx) {
    PROFILE_SCOPE_FMT("xfer_chunk(%s,stream=%llu,idx=%ld)", name_.c_str(), stream_id_, chunk_idx);
    char *cpu_base = (char *)cpu_tensor.data_ptr();
    bool h2d = (direction_ == GpuTransferDirection::H2D);
    auto x = xctx_;

    queue_->submit([=]() {
        PROFILE_SCOPE_FMT("xfer_chunk(%s,stream=%llu,idx=%ld)", name_.c_str(), stream_id_,
                          chunk_idx);
        kv_xfer::copy_chunk(x, cpu_base, chunk_idx, h2d);
    });
}

void Context::xfer_chunks_batch(const std::vector<int64_t> &chunk_indices,
                                const std::vector<torch::Tensor> &cpu_tensors) {
    PROFILE_SCOPE_FMT("xfer_chunks_batch(%s,stream=%llu,n=%zu,i0=%ld)", name_.c_str(), stream_id_,
                      chunk_indices.size(), chunk_indices[0]);

    auto chunk_indices_copy = chunk_indices;
    bool h2d = (direction_ == GpuTransferDirection::H2D);
    auto x = xctx_;

    std::vector<char *> cpu_ptrs;
    cpu_ptrs.reserve(cpu_tensors.size());
    for (const auto &tensor : cpu_tensors) {
        cpu_ptrs.push_back((char *)tensor.data_ptr());
    }

    queue_->submit([=, chunk_indices = std::move(chunk_indices_copy)]() {
        PROFILE_SCOPE_FMT("xfer_chunks_batch(%s,stream=%llu,n=%zu,i0=%ld)", name_.c_str(),
                          stream_id_, chunk_indices.size(), chunk_indices[0]);
        kv_xfer::copy_chunks_batch(x, chunk_indices, cpu_ptrs, h2d);
    });
}

void Context::xfer_finish() {
    PROFILE_SCOPE_FMT("xfer_finish(%s,stream=%llu)", name_.c_str(), stream_id_);
    auto event = event_;
    auto x = xctx_;
    auto *ev_flag = &event_recorded_;

    xfer_last_future_ = queue_->submit([=]() {
        PROFILE_SCOPE_FMT("xfer_finish(%s,stream=%llu)", name_.c_str(), stream_id_);
        kv_xfer::context_record_event(x, event);
        ev_flag->store(true, std::memory_order_release);
    });
}

void Context::xfer_wait() {
    IAXL_CHECK(xfer_last_future_.valid(), "xfer_wait: xfer_finish must be called first");
    {
        PROFILE_SCOPE_FMT("xfer_wait(%s,stream=%llu)", name_.c_str(), stream_id_);
        xfer_last_future_.get();
        xfer_last_future_ = std::future<void>();
    }
    {
        PROFILE_SCOPE_FMT("event_wait(%s,stream=%llu)", name_.c_str(), stream_id_);
        kv_xfer::event_synchronize(event_);

        if (direction_ == GpuTransferDirection::H2D && !kv_xfer::context_same_stream(xctx_)) {
            kv_xfer::context_cur_wait_event(xctx_, event_);
        }
    }
}

bool Context::xfer_is_complete() {
    if (xfer_last_future_.valid() &&
        xfer_last_future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        if (envs.IAXL_DEBUG_LOG)
            fprintf(stderr, "[xfer_is_complete] %s: future not ready\n", name_.c_str());
        return false;
    }
    bool recorded = event_recorded_.load(std::memory_order_acquire);
    if (!recorded && envs.IAXL_DEBUG_LOG)
        fprintf(stderr, "[xfer_is_complete] %s: future=%s but event_recorded=false\n",
                name_.c_str(), xfer_last_future_.valid() ? "valid+ready" : "invalid");
    return recorded;
}

void Context::xfer_wait_cur_stream(bool sync_cur_stream) {
    if (kv_xfer::context_same_stream(xctx_) && !sync_cur_stream)
        return;
    if (sync_cur_stream) {
        PROFILE_SCOPE("sync_cur_stream");

        kv_xfer::context_sync_cur(xctx_);
    }
    auto x = xctx_;
    queue_->submit([x]() {
        PROFILE_SCOPE("wait_cur_stream");
        kv_xfer::context_work_wait_cur(x);
    });
}

void Context::xfer_wait_stream(kv_xfer::event_t wait_event) {
    auto x = xctx_;
    queue_->submit([x, wait_event]() {
        PROFILE_SCOPE("wait_stream");
        kv_xfer::context_work_wait_event(x, wait_event);
        kv_xfer::event_destroy(wait_event);
    });
}

bool Context::xfer_wait_stream(pybind11::object cur_stream) {
    PROFILE_SCOPE_FMT("wait_stream(%s)", name_.c_str());
    kv_xfer::event_t wait_event = kv_xfer::wait_stream_from_py(cur_stream);
    if (wait_event) {
        xfer_wait_stream(wait_event);
        return true;
    }
    return false;
}

void h2d_xfer_chunks_batch(const torch::Tensor &gpu_tensor, int chunk_dim,
                           const std::vector<int64_t> &chunk_indices,
                           const std::vector<torch::Tensor> &cpu_tensors) {
    auto ctx = Context::create(gpu_tensor, chunk_dim, GpuTransferDirection::H2D);
    ctx.xfer_chunks_batch(chunk_indices, cpu_tensors);
}

void d2h_xfer_chunks_batch(const torch::Tensor &gpu_tensor, int chunk_dim,
                           const std::vector<int64_t> &chunk_indices,
                           const std::vector<torch::Tensor> &cpu_tensors) {
    auto ctx = Context::create(gpu_tensor, chunk_dim, GpuTransferDirection::D2H);
    ctx.xfer_chunks_batch(chunk_indices, cpu_tensors);
}
