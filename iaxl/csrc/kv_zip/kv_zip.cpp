// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <omp.h>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "env.h"
#include "iaxl_common.h"
#include "qat_zip.h"
#include "data_shuffle.h"
#include "lossy.h"
#include "kv_zip.h"

#define OMP_SCHEDULE dynamic

namespace kv_zip {

static void ensure_qat_zip_init() {
    static std::once_flag flag;
    std::call_once(flag, [] { IAXL_CHECK(qat_zip_init() == 0, "kv_zip: qat_zip_init failed"); });
}

template <class Submit, class Complete>
static void qat_pipeline(size_t n, Submit &&submit, Complete &&complete) {
    ensure_qat_zip_init();
    const int qd = qat_zip_queue_depth();
    const int inst = qat_zip_num_slots() / qd;

#pragma omp parallel num_threads(inst)
    {
        const int t = omp_get_thread_num();
        const int base = t * qd;

        std::vector<size_t> items;
        for (size_t i = t; i < n; i += inst)
            items.push_back(i);

        const int m = static_cast<int>(items.size());
        const int depth = m < qd ? m : qd;
        std::vector<size_t> slot_item(depth);

        for (int k = 0; k < depth; k++) {
            submit(base + k, items[k]);
            slot_item[k] = items[k];
        }
        for (int completed = 0, next = depth, s = 0; completed < m;
             completed++, s = (s + 1) % depth) {
            void *out;
            int out_len;
            IAXL_CHECK(qat_zip_wait(base + s, &out, &out_len) == 0, "kv_zip: qat_zip_wait failed");
            complete(slot_item[s], out, out_len);
            if (next < m) {
                submit(base + s, items[next]);
                slot_item[s] = items[next++];
            }
        }
    }
}

void kv_zip_compress_batch(const std::vector<torch::Tensor> &tensors, std::vector<char *> &out_bufs,
                           std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes) {
    const size_t n = tensors.size();

    auto prep = [&](size_t i, char **data, size_t *nbytes) {
        const auto &t = tensors[i];
        IAXL_CHECK(t.is_contiguous() && t.device().type() == c10::DeviceType::CPU,
                   "kv_zip: tensor must be a contiguous CPU tensor");
        size_t nb = t.numel() * t.element_size();
        char *p = static_cast<char *>(t.data_ptr());
        if (envs.IAXL_KV_COMPRESSION) {
            lossy_trunc(p, nb, t.element_size());
            data_shuffle(p, nb, t.dtype() == torch::kBFloat16, data_shuffle_enabled());
        }
        orig_sizes[i] = nb;
        *data = p;
        *nbytes = nb;
    };

    auto pack = [&](size_t i, const void *payload, int payload_len) {
        char *buf = static_cast<char *>(malloc(sizeof(int) * 2 + payload_len));
        IAXL_CHECK(buf != nullptr, "kv_zip: cache buffer allocation failed");
        reinterpret_cast<int *>(buf)[0] = payload_len;
        reinterpret_cast<int *>(buf)[1] = static_cast<int>(orig_sizes[i]);
        memcpy(buf + sizeof(int) * 2, payload, payload_len);
        out_bufs[i] = buf;
        out_sizes[i] = sizeof(int) * 2 + payload_len;
    };

    if (!envs.IAXL_KV_COMPRESSION) {
#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(envs.IAXL_QAT_INSTANCE_NUM)
        for (size_t i = 0; i < n; i++) {
            char *data;
            size_t nb;
            prep(i, &data, &nb);
            pack(i, data, static_cast<int>(nb));
        }
        return;
    }

    qat_pipeline(
        n,
        [&](int slot, size_t i) {
            char *data;
            size_t nb;
            prep(i, &data, &nb);
            IAXL_CHECK(nb <= static_cast<size_t>(INT_MAX),
                       "kv_zip: tensor byte size exceeds QAT integer length range");
            IAXL_CHECK(nb <= static_cast<size_t>(qat_zip_src_cap()),
                       "kv_zip: tensor byte size exceeds QAT source capacity");
            IAXL_CHECK(qat_zip_compress(slot, data, static_cast<int>(nb)) == 0,
                       "kv_zip: qat_zip_compress failed");
        },
        [&](size_t i, void *out, int out_len) { pack(i, out, out_len); });
}

void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                             const std::vector<torch::Tensor> &tensors) {
    const size_t n = tensors.size();

    auto finish = [&](size_t i, const void *out, int out_len) {
        const auto &t = tensors[i];
        size_t nb = t.numel() * t.element_size();
        IAXL_CHECK(out_len >= 0 && static_cast<size_t>(out_len) == nb,
                   "kv_zip: decompressed size does not match tensor byte size");
        char *dst = static_cast<char *>(t.data_ptr());
        memcpy(dst, out, nb);
        if (envs.IAXL_KV_COMPRESSION)
            data_shuffle(dst, nb, t.dtype() == torch::kBFloat16, data_shuffle_enabled());
    };

    if (!envs.IAXL_KV_COMPRESSION) {
#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(envs.IAXL_QAT_INSTANCE_NUM)
        for (size_t i = 0; i < n; i++) {
            const int *hdr = reinterpret_cast<const int *>(data_ptrs[i]);
            finish(i, data_ptrs[i] + sizeof(int) * 2, hdr[1]);
        }
        return;
    }

    qat_pipeline(
        n,
        [&](int slot, size_t i) {
            const int *hdr = reinterpret_cast<const int *>(data_ptrs[i]);
            const char *payload = data_ptrs[i] + sizeof(int) * 2;
            IAXL_CHECK(qat_zip_decompress(slot, const_cast<char *>(payload), hdr[0]) == 0,
                       "kv_zip: qat_zip_decompress failed");
        },
        [&](size_t i, void *out, int out_len) { finish(i, out, out_len); });
}

} // namespace kv_zip
