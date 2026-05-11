# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from pylibraft.common.cuda import Stream

class HasCudaStream(Protocol):
    def __cuda_stream__(self) -> tuple[int, int]: ...

CudaStreamLike = Stream | HasCudaStream | int
