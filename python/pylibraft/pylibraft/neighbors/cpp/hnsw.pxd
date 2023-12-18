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

from libc.stdint cimport int8_t, int64_t, uint8_t, uint32_t, uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from pylibraft.common.cpp.mdspan cimport (
    device_vector_view,
    host_matrix_view,
    row_major,
)
from pylibraft.common.handle cimport device_resources
from pylibraft.distance.distance_type cimport DistanceType
from pylibraft.neighbors.ivf_pq.cpp.c_ivf_pq cimport (
    ann_index,
    ann_search_params,
)


cdef extern from "raft/neighbors/hnsw.hpp" \
        namespace "raft::neighbors::hnsw" nogil:

    cpdef cppclass search_params(ann_search_params):
        int ef
        int num_threads

    cdef cppclass index[T](ann_index):
        index(int dim, DistanceType metric)

        int dim()
        DistanceType metric()


cdef extern from "raft_runtime/neighbors/hnsw.hpp" \
        namespace "raft::runtime::neighbors::hnsw" nogil:
    cdef void search(
        const device_resources& handle,
        const search_params& params,
        const index[float]& index,
        host_matrix_view[float, int64_t, row_major] queries,
        host_matrix_view[uint64_t, int64_t, row_major] neighbors,
        host_matrix_view[float, int64_t, row_major] distances) except +

    cdef void search(
        const device_resources& handle,
        const search_params& params,
        const index[int8_t]& index,
        host_matrix_view[int8_t, int64_t, row_major] queries,
        host_matrix_view[uint64_t, int64_t, row_major] neighbors,
        host_matrix_view[float, int64_t, row_major] distances) except +

    cdef void search(
        const device_resources& handle,
        const search_params& params,
        const index[uint8_t]& index,
        host_matrix_view[uint8_t, int64_t, row_major] queries,
        host_matrix_view[uint64_t, int64_t, row_major] neighbors,
        host_matrix_view[float, int64_t, row_major] distances) except +

    cdef unique_ptr[index[float]] deserialize_file[float](
        const device_resources& handle,
        const string& filename,
        int dim,
        DistanceType metric) except +

    cdef unique_ptr[index[int8_t]] deserialize_file[int8_t](
        const device_resources& handle,
        const string& filename,
        int dim,
        DistanceType metric) except +

    cdef unique_ptr[index[uint8_t]] deserialize_file[uint8_t](
        const device_resources& handle,
        const string& filename,
        int dim,
        DistanceType metric) except +
