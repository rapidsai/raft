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

from libc.stdint cimport int8_t, uint8_t, uint32_t
from libcpp cimport bool
from libcpp.string cimport string

cimport pylibraft.neighbors.cagra.cpp.c_cagra as c_cagra


cdef class Index:
    cdef readonly bool trained
    cdef str active_index_type

cdef class IndexFloat(Index):
    cdef c_cagra.index[float, uint32_t] * index

cdef class IndexInt8(Index):
    cdef c_cagra.index[int8_t, uint32_t] * index

cdef class IndexUint8(Index):
    cdef c_cagra.index[uint8_t, uint32_t] * index
