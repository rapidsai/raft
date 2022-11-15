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


cdef extern from "raft/core/device_mdspan.hpp" namespace "raft" nogil:
    cdef cppclass device_vector_view[ElementType, IndexType]:
        pass

    cdef cppclass device_matrix_view[ElementType, IndexType]:
        pass

    cdef cppclass host_scalar_view[ElementType, IndexType]:
        pass

    cdef device_vector_view[ElementType, IndexType] make_device_vector_view[ElementType, IndexType](ElementType * ptr, IndexType n) except +
    
    cdef device_matrix_view[ElementType, IndexType] make_device_matrix_view[ElementType, IndexType](ElementType * ptr, IndexType rows, IndexType cols) except +

    cdef host_scalar_view[ElementType, IndexType] make_host_scalar_view[ElementType, IndexType](ElementType * ptr) except +


ctypedef fused ElementType:
    float
    double


cdef device_matrix_view[ElementType, int] device_matrix_view_from_array(arr, ElementType * p) except +
cdef device_matrix_view[const ElementType, int] const_device_matrix_view_from_array(arr, ElementType * p) except +
