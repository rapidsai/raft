#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.memory cimport shared_ptr

from cuda.bindings.cyruntime cimport cudaStream_t



cdef extern from "raft/core/interruptible.hpp" namespace "raft" nogil:
    cdef cppclass interruptible:
        void cancel()

cdef extern from "raft/core/interruptible.hpp" \
        namespace "raft::interruptible" nogil:
    cdef void inter_synchronize \
        "raft::interruptible::synchronize"(cudaStream_t stream) except+
    cdef void inter_yield "raft::interruptible::yield"() except+
    cdef shared_ptr[interruptible] get_token() except+
