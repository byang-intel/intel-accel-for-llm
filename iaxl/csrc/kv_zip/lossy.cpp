// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "lossy.h"

#include <cstddef>
#include <cstdint>

#include "env.h"

namespace kv_zip {

void lossy_trunc(char *data, size_t size, int element_size) {
    const int truncate_bits = envs.IAXL_KV_LOSSY_TRUNC;
    if (truncate_bits <= 0)
        return;

    if (element_size == 2) {

        const uint16_t mask = static_cast<uint16_t>(0xFFFF << truncate_bits);
        uint16_t *ptr = reinterpret_cast<uint16_t *>(data);
        const size_t n = size / 2;

        size_t i = 0;

        for (; i + 8 <= n; i += 8) {
            __builtin_prefetch(ptr + i + 16, 1, 1);

            ptr[i + 0] &= mask;
            ptr[i + 1] &= mask;
            ptr[i + 2] &= mask;
            ptr[i + 3] &= mask;
            ptr[i + 4] &= mask;
            ptr[i + 5] &= mask;
            ptr[i + 6] &= mask;
            ptr[i + 7] &= mask;
        }

        for (; i < n; i++) {
            ptr[i] &= mask;
        }

    } else if (element_size == 1) {

        const uint8_t mask = static_cast<uint8_t>(0xFF << truncate_bits);
        uint8_t *ptr = reinterpret_cast<uint8_t *>(data);
        const size_t n = size;

        size_t i = 0;

        for (; i + 8 <= n; i += 8) {
            __builtin_prefetch(ptr + i + 64, 1, 1);

            ptr[i + 0] &= mask;
            ptr[i + 1] &= mask;
            ptr[i + 2] &= mask;
            ptr[i + 3] &= mask;
            ptr[i + 4] &= mask;
            ptr[i + 5] &= mask;
            ptr[i + 6] &= mask;
            ptr[i + 7] &= mask;
        }

        for (; i < n; i++) {
            ptr[i] &= mask;
        }
    }
}

} // namespace kv_zip
