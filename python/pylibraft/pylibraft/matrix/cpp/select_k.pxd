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

from libc.stdint cimport int64_t
from libcpp cimport bool

from pylibraft.common.cpp.mdspan cimport device_matrix_view, row_major
from pylibraft.common.cpp.optional cimport optional
from pylibraft.common.handle cimport device_resources


cdef extern from "raft_runtime/matrix/select_k.hpp" \
        namespace "raft::runtime::matrix" nogil:

    cdef void select_k(const device_resources & handle,
                       device_matrix_view[float, int64_t, row_major],
                       optional[device_matrix_view[int64_t,
                                                   int64_t,
                                                   row_major]],
                       device_matrix_view[float, int64_t, row_major],
                       device_matrix_view[int64_t, int64_t, row_major],
                       bool) except +
