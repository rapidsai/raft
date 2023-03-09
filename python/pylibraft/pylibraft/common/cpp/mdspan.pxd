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

from libc.stdint cimport int8_t, int64_t, uint8_t, uint64_t
from libcpp.string cimport string

from pylibraft.common.handle cimport device_resources


cdef extern from "raft/thirdparty/mdspan/include/experimental/__p0009_bits/layout_stride.hpp" namespace "std::experimental":  # noqa: E501
    cdef cppclass layout_right:
        pass

    cdef cppclass layout_left:
        pass


cdef extern from "raft/core/mdspan_types.hpp" \
        namespace "raft":
    ctypedef layout_right row_major
    ctypedef layout_left col_major
    cdef cppclass matrix_extent[IndexType]:
        pass


cdef extern from "raft/core/device_mdspan.hpp" namespace "raft" nogil:

    cdef cppclass device_vector_view[ElementType, IndexType]:
        pass

    cdef cppclass device_scalar_view[ElementType, IndexType]:
        pass

    cdef cppclass device_matrix_view[ElementType, IndexType, LayoutType]:
        pass

    cdef device_matrix_view[ElementType, IndexType, LayoutPolicy] \
        make_device_matrix_view[ElementType, IndexType, LayoutPolicy](
            ElementType* ptr, IndexType n_rows, IndexType n_cols) except +

    cdef device_vector_view[ElementType, IndexType] \
        make_device_vector_view[ElementType, IndexType](
            ElementType* ptr, IndexType n) except +

    cdef device_scalar_view[ElementType, IndexType] \
        make_device_vector_view[ElementType, IndexType](
            ElementType* ptr) except +


cdef extern from "raft/core/host_mdspan.hpp" \
        namespace "raft" nogil:

    cdef cppclass host_matrix_view[ElementType, IndexType, LayoutPolicy]:
        pass

    cdef cppclass host_vector_view[ElementType, IndexType]:
        pass

    cdef cppclass host_scalar_view[ElementType, IndexType]:
        pass

    cdef cppclass host_mdspan[ElementType, Extents, LayoutPolicy]:
        pass

    cdef host_matrix_view[ElementType, IndexType, LayoutPolicy] \
        make_host_matrix_view[ElementType, IndexType, LayoutPolicy](
            ElementType* ptr, IndexType n_rows, IndexType n_cols) except +

    cdef host_vector_view[ElementType, IndexType] \
        make_host_vector_view[ElementType, IndexType](
            ElementType* ptr, IndexType n) except +

    cdef host_scalar_view[ElementType, IndexType] \
        make_host_scalar_view[ElementType, IndexType](
            ElementType *ptr) except +

cdef extern from "<sstream>" namespace "std" nogil:
    cdef cppclass ostringstream:
        ostringstream() except +
        string str() except +


cdef extern from "<ostream>" namespace "std" nogil:

    cdef cppclass ostream:
        pass

cdef extern from "raft/core/mdspan.hpp" namespace "raft" nogil:
    cdef cppclass dextents[IndentType, Rank]:
        pass

cdef extern from "raft/core/serialize.hpp" namespace "raft" nogil:

    cdef void serialize_mdspan[ElementType, Extents, LayoutPolicy](
        const device_resources& handle, ostream& os,
        const host_mdspan[ElementType, Extents, LayoutPolicy]& obj)
