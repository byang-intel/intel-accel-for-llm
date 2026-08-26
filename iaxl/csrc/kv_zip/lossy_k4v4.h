// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include <torch/extension.h>

#include "env.h"

namespace kv_zip {

// Feature switch for the TurboQuant-style 4-bit lossy MSE KV codec (K4/V4).
inline bool lossy_k4v4_enabled() { return envs.IAXL_KV_LOSSY_K4V4; }

// True when the codec applies to this tensor: switch on, bf16, and the trailing
// (head) dimension is a multiple of 16 in (0, 1024]. Evaluated identically on the
// compress and decompress side so both agree per tensor without any stored flag.
bool lossy_k4v4_applicable(const torch::Tensor &t);

// Compress side: quantize `t` (bf16, contiguous) and serialize the
// (indices || norms) payload into a freshly malloc'd buffer. Returns the buffer
// and writes its byte length to *out_size. Ownership passes to the caller (free()).
char *lossy_k4v4_serialize(const torch::Tensor &t, size_t *out_size);

// Decompress side: reinterpret `src` as the (indices || norms) payload in place
// (no copy) and reconstruct straight into `t`'s storage via the inverse rotation.
void lossy_k4v4_deserialize(const void *src, size_t src_size, const torch::Tensor &t);

} // namespace kv_zip
