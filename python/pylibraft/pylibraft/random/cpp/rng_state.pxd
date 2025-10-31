#
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stdint cimport uint64_t


cdef extern from "raft/random/rng_state.hpp" namespace "raft::random" nogil:

    ctypedef enum GeneratorType:
        GenPhilox "raft::random::GeneratorType::GenPhilox"
        GenPC "raft::random::GeneratorType::GenPC"

    cdef cppclass RngState:
        RngState(uint64_t seed) except +
        uint64_t seed
        uint64_t base_subsequence
        GeneratorType type
