#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
