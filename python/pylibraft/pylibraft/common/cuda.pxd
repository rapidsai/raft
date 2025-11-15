#
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from cuda.bindings.cyruntime cimport cudaStream_t


cdef class Stream:
    cdef cudaStream_t s

    cdef cudaStream_t getStream(self)
