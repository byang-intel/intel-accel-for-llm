// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <utility>

#include "env.h"

namespace kv_zip {

inline bool data_shuffle_enabled() { return envs.IAXL_KV_DATA_SHUFFLE; }

inline void data_shuffle(char *data, size_t size, bool is_bf16, bool enabled) {
    if (!enabled || !is_bf16)
        return;

    char *high_bytes = data;
    char *low_bytes = data + size / 2;
    const size_t n = size / 4;

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __builtin_prefetch(high_bytes + (i + 16) * 2, 1, 1);
        __builtin_prefetch(low_bytes + (i + 16) * 2, 1, 1);

        char t0 = high_bytes[(i + 0) * 2];
        char t1 = high_bytes[(i + 1) * 2];
        char t2 = high_bytes[(i + 2) * 2];
        char t3 = high_bytes[(i + 3) * 2];
        char t4 = high_bytes[(i + 4) * 2];
        char t5 = high_bytes[(i + 5) * 2];
        char t6 = high_bytes[(i + 6) * 2];
        char t7 = high_bytes[(i + 7) * 2];

        high_bytes[(i + 0) * 2] = low_bytes[(i + 0) * 2 + 1];
        high_bytes[(i + 1) * 2] = low_bytes[(i + 1) * 2 + 1];
        high_bytes[(i + 2) * 2] = low_bytes[(i + 2) * 2 + 1];
        high_bytes[(i + 3) * 2] = low_bytes[(i + 3) * 2 + 1];
        high_bytes[(i + 4) * 2] = low_bytes[(i + 4) * 2 + 1];
        high_bytes[(i + 5) * 2] = low_bytes[(i + 5) * 2 + 1];
        high_bytes[(i + 6) * 2] = low_bytes[(i + 6) * 2 + 1];
        high_bytes[(i + 7) * 2] = low_bytes[(i + 7) * 2 + 1];

        low_bytes[(i + 0) * 2 + 1] = t0;
        low_bytes[(i + 1) * 2 + 1] = t1;
        low_bytes[(i + 2) * 2 + 1] = t2;
        low_bytes[(i + 3) * 2 + 1] = t3;
        low_bytes[(i + 4) * 2 + 1] = t4;
        low_bytes[(i + 5) * 2 + 1] = t5;
        low_bytes[(i + 6) * 2 + 1] = t6;
        low_bytes[(i + 7) * 2 + 1] = t7;
    }

    for (; i < n; i++) {
        std::swap(high_bytes[i * 2], low_bytes[i * 2 + 1]);
    }
}

} // namespace kv_zip
