# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Handle for a client GPU tensor that the daemon reaches over RDMA. Carries just
enough of the torch.Tensor surface for KVStore/KVFlow (shape, dtype, contiguity)."""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class RemoteTensor:
    peer: str
    base: int  # remote VRAM virtual address
    shape: Tuple[int, ...]
    dtype: torch.dtype
    dev_id: int

    is_remote = True
    is_cuda = False
    is_xpu = False

    @property
    def device(self) -> str:
        return f"remote:{self.peer}:{self.dev_id}"

    def element_size(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()

    def dim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return True

    def stride(self, d: int) -> int:
        s = 1
        for n in self.shape[d + 1 :]:
            s *= n
        return s
