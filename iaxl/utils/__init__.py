# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import warnings

import torch


def cuda_available() -> bool:
    """Probe CUDA without torch's noisy warning when no usable device exists."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CUDA initialization.*",
                                category=UserWarning)
        try:
            return torch.cuda.is_available()
        except Exception:
            return False
