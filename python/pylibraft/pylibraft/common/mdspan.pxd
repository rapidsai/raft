#
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

from libc.stdint cimport int8_t, uint8_t, uint64_t
from libcpp.string cimport string

from pylibraft.common.cpp.mdspan cimport device_matrix_view, row_major
from pylibraft.common.handle cimport device_resources


cdef device_matrix_view[float, uint64_t, row_major] get_dmv_float(
    array, check_shape) except *

cdef device_matrix_view[uint8_t, uint64_t, row_major] get_dmv_uint8(
    array, check_shape) except *

cdef device_matrix_view[int8_t, uint64_t, row_major] get_dmv_int8(
    array, check_shape) except *

cdef device_matrix_view[uint64_t, uint64_t, row_major] get_dmv_uint64(
    array, check_shape) except *
