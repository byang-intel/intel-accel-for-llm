# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import os
import hashlib
from typing import Union, List, Tuple, Optional, Any, Callable
import numpy as np
import xxhash


def compute_hash(*args, **kwargs) -> int:
    parts = [repr(arg) for arg in args]
    parts += [f"{k}={repr(v)}" for k, v in sorted(kwargs.items())]
    text = "|".join(parts)
    return xxhash.xxh64(text.encode("utf-8")).intdigest() & 0x7FFFFFFFFFFFFFFF


def hash_block_tokens(
    is_first_block,
    prev_block_hash,
    cur_block_token_ids,
) -> int:
    return compute_hash(is_first_block, prev_block_hash, cur_block_token_ids)


def generate_block_hashs(
    tokens,
    block_size,
):
    aligned_len = (len(tokens) // block_size) * block_size
    prev_block_hash = None
    hashs = []
    for i in range(0, aligned_len, block_size):
        block_tokens = tokens[i : i + block_size]
        is_first_block = i == 0
        cur_hash = hash_block_tokens(
            is_first_block=is_first_block,
            prev_block_hash=prev_block_hash,
            cur_block_token_ids=block_tokens,
        )
        hashs.append(cur_hash)
        prev_block_hash = cur_hash
    return hashs
