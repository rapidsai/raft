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
from libc.stdint cimport int8_t, int64_t, uint8_t, uint32_t, uint64_t
from libcpp cimport bool, nullptr
from libcpp.string cimport string

from rmm._lib.memory_resource cimport device_memory_resource

from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    device_vector_view,
    host_matrix_view,
    row_major,
)
from pylibraft.common.handle cimport device_resources
from pylibraft.common.mdspan cimport const_float, const_int8_t, const_uint8_t
from pylibraft.common.optional cimport optional
from pylibraft.distance.distance_type cimport DistanceType
from pylibraft.neighbors.ivf_pq.cpp.c_ivf_pq cimport (
    ann_index,
    ann_index_params,
    ann_search_params,
    index_params as ivfpq_ip,
    search_params as ivfpq_sp,
)


cdef extern from "raft/neighbors/cagra_types.hpp" \
        namespace "raft::neighbors::cagra" nogil:

    ctypedef enum graph_build_algo:
        IVF_PQ "raft::neighbors::cagra::graph_build_algo::IVF_PQ",
        NN_DESCENT "raft::neighbors::cagra::graph_build_algo::NN_DESCENT"

    cpdef cppclass index_params(ann_index_params):
        size_t intermediate_graph_degree
        size_t graph_degree
        graph_build_algo build_algo

    ctypedef enum search_algo:
        SINGLE_CTA "raft::neighbors::cagra::search_algo::SINGLE_CTA",
        MULTI_CTA "raft::neighbors::cagra::search_algo::MULTI_CTA",
        MULTI_KERNEL "raft::neighbors::cagra::search_algo::MULTI_KERNEL",
        AUTO "raft::neighbors::cagra::search_algo::AUTO"

    ctypedef enum hash_mode:
        HASH "raft::neighbors::cagra::hash_mode::HASH",
        SMALL "raft::neighbors::cagra::hash_mode::SMALL",
        AUTO "raft::neighbors::cagra::hash_mode::AUTO"

    cpdef cppclass search_params(ann_search_params):
        size_t max_queries
        size_t itopk_size
        size_t max_iterations
        search_algo algo
        size_t team_size
        size_t search_width
        size_t min_iterations
        size_t thread_block_size
        hash_mode hashmap_mode
        size_t hashmap_min_bitlen
        float hashmap_max_fill_rate
        uint32_t num_random_samplings
        uint64_t rand_xor_mask

    cdef cppclass index[T, IdxT](ann_index):
        index(const device_resources&)

        DistanceType metric()
        IdxT size()
        uint32_t dim()
        uint32_t graph_degree()
        device_matrix_view[T, IdxT, row_major] dataset()
        device_matrix_view[T, IdxT, row_major] graph()

        # hack: can't use the T template param here because of issues handling
        # const w/ cython. introduce a new template param to get around this
        void update_dataset[ValueT](const device_resources & handle,
                                    host_matrix_view[ValueT,
                                                     int64_t,
                                                     row_major] dataset)
        void update_dataset[ValueT](const device_resources & handle,
                                    device_matrix_view[ValueT,
                                                       int64_t,
                                                       row_major] dataset)

cdef extern from "raft_runtime/neighbors/cagra.hpp" \
        namespace "raft::runtime::neighbors::cagra" nogil:

    cdef void build_device(
        const device_resources& handle,
        const index_params& params,
        device_matrix_view[float, int64_t, row_major] dataset,
        index[float, uint32_t]& index) except +

    cdef void build_device(
        const device_resources& handle,
        const index_params& params,
        device_matrix_view[int8_t, int64_t, row_major] dataset,
        index[int8_t, uint32_t]& index) except +

    cdef void build_device(
        const device_resources& handle,
        const index_params& params,
        device_matrix_view[uint8_t, int64_t, row_major] dataset,
        index[uint8_t, uint32_t]& index) except +

    cdef void build_host(
        const device_resources& handle,
        const index_params& params,
        host_matrix_view[float, int64_t, row_major] dataset,
        index[float, uint32_t]& index) except +

    cdef void build_host(
        const device_resources& handle,
        const index_params& params,
        host_matrix_view[int8_t, int64_t, row_major] dataset,
        index[int8_t, uint32_t]& index) except +

    cdef void build_host(
        const device_resources& handle,
        const index_params& params,
        host_matrix_view[uint8_t, int64_t, row_major] dataset,
        index[uint8_t, uint32_t]& index) except +

    cdef void search(
        const device_resources& handle,
        const search_params& params,
        const index[float, uint32_t]& index,
        device_matrix_view[float, int64_t, row_major] queries,
        device_matrix_view[uint32_t, int64_t, row_major] neighbors,
        device_matrix_view[float, int64_t, row_major] distances) except +

    cdef void search(
        const device_resources& handle,
        const search_params& params,
        const index[int8_t, uint32_t]& index,
        device_matrix_view[int8_t, int64_t, row_major] queries,
        device_matrix_view[uint32_t, int64_t, row_major] neighbors,
        device_matrix_view[float, int64_t, row_major] distances) except +

    cdef void search(
        const device_resources& handle,
        const search_params& params,
        const index[uint8_t, uint32_t]& index,
        device_matrix_view[uint8_t, int64_t, row_major] queries,
        device_matrix_view[uint32_t, int64_t, row_major] neighbors,
        device_matrix_view[float, int64_t, row_major] distances) except +

    cdef void serialize(const device_resources& handle,
                        string& str,
                        const index[float, uint32_t]& index,
                        bool include_dataset) except +
    cdef void serialize_to_hnwslib(
        const device_resources& handle,
        string& str,
        const index[float, uint32_t]& index) except +

    cdef void deserialize(const device_resources& handle,
                          const string& str,
                          index[float, uint32_t]* index) except +

    cdef void serialize(const device_resources& handle,
                        string& str,
                        const index[uint8_t, uint32_t]& index,
                        bool include_dataset) except +

    cdef void serialize_to_hnwslib(
        const device_resources& handle,
        string& str,
        const index[uint8_t, uint32_t]& index) except +

    cdef void deserialize(const device_resources& handle,
                          const string& str,
                          index[uint8_t, uint32_t]* index) except +

    cdef void serialize(const device_resources& handle,
                        string& str,
                        const index[int8_t, uint32_t]& index,
                        bool include_dataset) except +

    cdef void serialize_to_hnwslib(
        const device_resources& handle,
        string& str,
        const index[int8_t, uint32_t]& index) except +

    cdef void deserialize(const device_resources& handle,
                          const string& str,
                          index[int8_t, uint32_t]* index) except +

    cdef void serialize_file(const device_resources& handle,
                             const string& filename,
                             const index[float, uint32_t]& index,
                             bool include_dataset) except +

    cdef void serialize_to_hnswlib_file(
        const device_resources& handle,
        const string& filename,
        const index[float, uint32_t]& index) except +

    cdef void deserialize_file(const device_resources& handle,
                               const string& filename,
                               index[float, uint32_t]* index) except +

    cdef void serialize_file(const device_resources& handle,
                             const string& filename,
                             const index[uint8_t, uint32_t]& index,
                             bool include_dataset) except +

    cdef void serialize_to_hnswlib_file(
        const device_resources& handle,
        const string& filename,
        const index[uint8_t, uint32_t]& index) except +

    cdef void deserialize_file(const device_resources& handle,
                               const string& filename,
                               index[uint8_t, uint32_t]* index) except +

    cdef void serialize_file(const device_resources& handle,
                             const string& filename,
                             const index[int8_t, uint32_t]& index,
                             bool include_dataset) except +

    cdef void serialize_to_hnswlib_file(
        const device_resources& handle,
        const string& filename,
        const index[int8_t, uint32_t]& index) except +

    cdef void deserialize_file(const device_resources& handle,
                               const string& filename,
                               index[int8_t, uint32_t]* index) except +

cdef class Index:
    cdef readonly bool trained
    cdef str active_index_type

cdef class IndexFloat(Index):
    pass

cdef class IndexInt8(Index):
    pass

cdef class IndexUint8(Index):
    pass
