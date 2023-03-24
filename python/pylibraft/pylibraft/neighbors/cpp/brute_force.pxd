#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

import numpy as np

import pylibraft.common.handle

from cython.operator cimport dereference as deref
from libc.stdint cimport int8_t, int64_t, uint8_t, uint64_t, uintptr_t
from libcpp cimport bool, nullptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from rmm._lib.memory_resource cimport device_memory_resource

from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    host_matrix_view,
    make_device_matrix_view,
    make_host_matrix_view,
    row_major,
)
from pylibraft.common.cpp.optional cimport optional
from pylibraft.common.handle cimport device_resources
from pylibraft.distance.distance_type cimport DistanceType


cdef extern from "raft_runtime/neighbors/brute_force.hpp" \
        namespace "raft::runtime::neighbors::brute_force" nogil:

    cdef void knn(const device_resources & handle,
                  device_matrix_view[float, int64_t, row_major] index,
                  device_matrix_view[float, int64_t, row_major] search,
                  device_matrix_view[int64_t, int64_t, row_major] indices,
                  device_matrix_view[float, int64_t, row_major] distances,
                  DistanceType metric,
                  optional[float] metric_arg,
                  optional[int64_t] global_id_offset) except +
