#
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

from libc.stdint cimport int8_t, int64_t, uint8_t, uint32_t, uint64_t
from libcpp cimport bool
from libcpp.string cimport string

from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    host_matrix_view,
    row_major,
)
from pylibraft.common.handle cimport device_resources
from pylibraft.common.optional cimport make_optional, optional

# Cython doesn't like `const float` inside template parameters
# hack around this with using typedefs
ctypedef const float const_float
ctypedef const int8_t const_int8_t
ctypedef const uint8_t const_uint8_t


cdef device_matrix_view[float, int64_t, row_major] get_dmv_float(
    array, check_shape) except *

cdef device_matrix_view[bool, int64_t, row_major] get_dmv_bool(
    array, check_shape) except *

cdef device_matrix_view[uint8_t, int64_t, row_major] get_dmv_uint8(
    array, check_shape) except *

cdef device_matrix_view[int8_t, int64_t, row_major] get_dmv_int8(
    array, check_shape) except *

cdef device_matrix_view[int64_t, int64_t, row_major] get_dmv_int64(
    array, check_shape) except *

cdef optional[device_matrix_view[int64_t, int64_t, row_major]] make_optional_view_int64(  # noqa: E501
    device_matrix_view[int64_t, int64_t, row_major]& dmv) except *

cdef device_matrix_view[uint32_t, int64_t, row_major] get_dmv_uint32(
    array, check_shape) except *

cdef device_matrix_view[const_float, int64_t, row_major] get_const_dmv_float(
    array, check_shape) except *

cdef device_matrix_view[const_uint8_t, int64_t, row_major] get_const_dmv_uint8(
    array, check_shape) except *

cdef device_matrix_view[const_int8_t, int64_t, row_major] get_const_dmv_int8(
    array, check_shape) except *

cdef host_matrix_view[float, int64_t, row_major] get_hmv_float(
    array, check_shape) except *

cdef host_matrix_view[uint8_t, int64_t, row_major] get_hmv_uint8(
    array, check_shape) except *

cdef host_matrix_view[int8_t, int64_t, row_major] get_hmv_int8(
    array, check_shape) except *

cdef host_matrix_view[int64_t, int64_t, row_major] get_hmv_int64(
    array, check_shape) except *

cdef host_matrix_view[uint32_t, int64_t, row_major] get_hmv_uint32(
    array, check_shape) except *

cdef host_matrix_view[uint64_t, int64_t, row_major] get_hmv_uint64(
    array, check_shape) except *

cdef host_matrix_view[const_float, int64_t, row_major] get_const_hmv_float(
    array, check_shape) except *

cdef host_matrix_view[const_uint8_t, int64_t, row_major] get_const_hmv_uint8(
    array, check_shape) except *

cdef host_matrix_view[const_int8_t, int64_t, row_major] get_const_hmv_int8(
    array, check_shape) except *
