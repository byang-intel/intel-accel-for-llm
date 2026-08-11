// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <queue>
#include <vector>

#include <pybind11/pybind11.h>
#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include "kv_xfer.h"

namespace py = pybind11;

#define GPU_CHECK(call)                                                                            \
    do {                                                                                           \
        try {                                                                                      \
            (call);                                                                                \
        } catch (const sycl::exception &e) {                                                       \
            fprintf(stderr, "SYCL error %s:%d: %s threw %s\n", __FILE__, __LINE__, #call,          \
                    e.what());                                                                     \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

using GpuStream_t = sycl::queue *;
using GpuEvent_t = sycl::event *;

class GpuEventPool {
  public:
    static GpuEventPool &instance() {
        static GpuEventPool pool;
        return pool;
    }

    GpuEvent_t acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_.empty()) {
            GpuEvent_t event = pool_.front();
            pool_.pop();

            *event = sycl::event();
            return event;
        }
        return new sycl::event();
    }

    void release(GpuEvent_t event) {
        if (event) {
            std::lock_guard<std::mutex> lock(mutex_);
            pool_.push(event);
        }
    }

    ~GpuEventPool() {
        while (!pool_.empty()) {
            delete pool_.front();
            pool_.pop();
        }
    }

  private:
    GpuEventPool() = default;
    std::mutex mutex_;
    std::queue<GpuEvent_t> pool_;
};

static inline void gpu_set_device(int device_index) { (void)device_index; }

static inline GpuStream_t gpu_get_current_stream(int device_index) {
    return &(at::xpu::getCurrentXPUStream(device_index).queue());
}

static inline unsigned long long gpu_get_stream_id(GpuStream_t stream) {
    return reinterpret_cast<unsigned long long>(stream);
}

static inline void gpu_event_record(GpuEvent_t event, GpuStream_t stream) {
    *event = stream->ext_oneapi_submit_barrier();
}

static inline void gpu_event_synchronize(GpuEvent_t event) { event->wait(); }

static inline void gpu_stream_wait_event(GpuStream_t stream, GpuEvent_t event) {
    stream->ext_oneapi_submit_barrier({*event});
}

static inline void gpu_stream_synchronize(GpuStream_t stream) { stream->wait(); }

static inline GpuEvent_t gpu_event_create() { return new sycl::event(); }

static inline void gpu_event_destroy(GpuEvent_t event) { delete event; }

static inline GpuStream_t gpu_extract_stream(py::object work_stream) {
    uintptr_t stream_ptr = work_stream.attr("sycl_queue").cast<uintptr_t>();
    return reinterpret_cast<GpuStream_t>(stream_ptr);
}

static inline GpuEvent_t gpu_wait_stream_from_py(py::object cur_stream) {
    cur_stream.attr("synchronize")();
    return nullptr;
}

namespace kv_xfer {

struct XferContext {
    int device_index;
    GpuStream_t cur_stream;
    GpuStream_t work_stream;
    unsigned long long stream_id;
    char *gpu_base_ptr;
    int64_t chunk_stride;
    int64_t outer_dims;
    int64_t inner_size;
    int64_t outer_block_size;
};

static inline XferContext *as_ctx(context_t c) { return static_cast<XferContext *>(c); }

event_t event_acquire() { return GpuEventPool::instance().acquire(); }

void event_release(event_t event) {
    GpuEventPool::instance().release(static_cast<GpuEvent_t>(event));
}

event_t event_create() { return gpu_event_create(); }

void event_destroy(event_t event) { gpu_event_destroy(static_cast<GpuEvent_t>(event)); }

void event_synchronize(event_t event) { gpu_event_synchronize(static_cast<GpuEvent_t>(event)); }

stream_t extract_stream(pybind11::object stream) { return gpu_extract_stream(stream); }

event_t wait_stream_from_py(pybind11::object stream) { return gpu_wait_stream_from_py(stream); }

context_t context_create(char *gpu_base_ptr, int device_index, int64_t chunk_stride,
                         int64_t outer_dims, int64_t inner_size, int64_t outer_block_size,
                         stream_t work_stream) {
    XferContext *x = new XferContext();
    x->device_index = device_index;
    x->cur_stream = gpu_get_current_stream(device_index);
    x->work_stream = work_stream ? static_cast<GpuStream_t>(work_stream) : x->cur_stream;
    x->stream_id = gpu_get_stream_id(x->work_stream);
    x->gpu_base_ptr = gpu_base_ptr;
    x->chunk_stride = chunk_stride;
    x->outer_dims = outer_dims;
    x->inner_size = inner_size;
    x->outer_block_size = outer_block_size;
    return x;
}

void context_destroy(context_t ctx) { delete as_ctx(ctx); }

unsigned long long context_stream_id(context_t ctx) { return as_ctx(ctx)->stream_id; }

bool context_same_stream(context_t ctx) {
    XferContext *x = as_ctx(ctx);
    return x->cur_stream == x->work_stream;
}

void context_record_event(context_t ctx, event_t event) {
    XferContext *x = as_ctx(ctx);
    gpu_set_device(x->device_index);
    gpu_event_record(static_cast<GpuEvent_t>(event), x->work_stream);
}

void context_work_wait_event(context_t ctx, event_t event) {
    gpu_stream_wait_event(as_ctx(ctx)->work_stream, static_cast<GpuEvent_t>(event));
}

void context_cur_wait_event(context_t ctx, event_t event) {
    gpu_stream_wait_event(as_ctx(ctx)->cur_stream, static_cast<GpuEvent_t>(event));
}

void context_work_wait_cur(context_t ctx) {
    XferContext *x = as_ctx(ctx);
    if (x->cur_stream == x->work_stream)
        return;
    gpu_set_device(x->device_index);
    GpuEvent_t ev = gpu_event_create();
    gpu_event_record(ev, x->cur_stream);
    gpu_stream_wait_event(x->work_stream, ev);
    gpu_event_destroy(ev);
}

void context_sync_cur(context_t ctx) { gpu_stream_synchronize(as_ctx(ctx)->cur_stream); }

void copy_chunk(context_t ctx, char *cpu_base, int64_t chunk_index, bool h2d) {
    XferContext *x = as_ctx(ctx);
    gpu_set_device(x->device_index);
    char *gpu_base = x->gpu_base_ptr + chunk_index * x->chunk_stride;
    GpuStream_t stream = x->work_stream;
    for (int64_t outer = 0; outer < x->outer_dims; outer++) {
        char *gpu_ptr = gpu_base + outer * x->outer_block_size;
        char *cpu_ptr = cpu_base + outer * x->inner_size;
        if (h2d) {
            stream->memcpy(gpu_ptr, cpu_ptr, x->inner_size);
        } else {
            stream->memcpy(cpu_ptr, gpu_ptr, x->inner_size);
        }
    }
}

void copy_chunks_batch(context_t ctx, const std::vector<int64_t> &chunk_indices,
                       const std::vector<char *> &cpu_ptrs, bool h2d) {
    XferContext *x = as_ctx(ctx);
    gpu_set_device(x->device_index);
    GpuStream_t stream = x->work_stream;
    for (size_t i = 0; i < chunk_indices.size(); i++) {
        char *gpu_base = x->gpu_base_ptr + chunk_indices[i] * x->chunk_stride;
        char *cpu_base = cpu_ptrs[i];

        for (int64_t outer = 0; outer < x->outer_dims; outer++) {
            char *gpu_ptr = gpu_base + outer * x->outer_block_size;
            char *cpu_ptr = cpu_base + outer * x->inner_size;

            if (h2d) {
                stream->memcpy(gpu_ptr, cpu_ptr, x->inner_size);
            } else {
                stream->memcpy(cpu_ptr, gpu_ptr, x->inner_size);
            }
        }
    }
}

} // namespace kv_xfer
