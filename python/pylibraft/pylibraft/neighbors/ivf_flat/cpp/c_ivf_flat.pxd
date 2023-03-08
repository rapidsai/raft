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
from libc.stdint cimport (
    int8_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)
from libcpp cimport bool, nullptr
from libcpp.string cimport string

from rmm._lib.memory_resource cimport device_memory_resource

from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    device_vector_view,
    host_matrix_view,
    make_device_matrix_view,
    make_host_matrix_view,
    row_major,
)
from pylibraft.common.cpp.optional cimport optional
from pylibraft.common.handle cimport device_resources
from pylibraft.distance.distance_type cimport DistanceType
from pylibraft.neighbors.ivf_pq.cpp.c_ivf_pq cimport (
    ann_index,
    ann_index_params,
    ann_search_params,
)


cdef extern from "raft/neighbors/ivf_flat_types.hpp" \
        namespace "raft::neighbors::ivf_flat" nogil:

    cpdef cppclass index_params(ann_index_params):
        uint32_t n_lists
        uint32_t kmeans_n_iters
        double kmeans_trainset_fraction
        bool add_data_on_build
        bool adaptive_centers

    cdef cppclass index[T, IdxT](ann_index):
        index(const device_resources& handle,
              DistanceType metric,
              uint32_t n_lists,
              bool adaptive_centers,
              uint32_t dim)
        IdxT size()
        uint32_t dim()
        DistanceType metric()
        uint32_t n_lists()
        bool adaptive_centers()

    cpdef cppclass search_params(ann_search_params):
        uint32_t n_probes


cdef extern from "raft_runtime/neighbors/ivf_flat.hpp" \
        namespace "raft::runtime::neighbors::ivf_flat" nogil:

    cdef void build(const device_resources&,
                    device_matrix_view[float, uint64_t, row_major] dataset,
                    const index_params& params,
                    index[float, uint64_t]* index) except +

    cdef void build(const device_resources& handle,
                    device_matrix_view[int8_t, uint64_t, row_major] dataset,
                    const index_params& params,
                    index[int8_t, uint64_t]* index) except +

    cdef void build(const device_resources& handle,
                    device_matrix_view[uint8_t, uint64_t, row_major] dataset,
                    const index_params& params,
                    index[uint8_t, uint64_t]* index) except +

    cdef void extend(
        const device_resources& handle,
        index[float, uint64_t]* index,
        device_matrix_view[float, uint64_t, row_major] new_vectors,
        optional[device_vector_view[uint64_t, uint64_t]] new_indices) except +

    cdef void extend(
        const device_resources& handle,
        index[int8_t, uint64_t]* index,
        device_matrix_view[int8_t, uint64_t, row_major] new_vectors,
        optional[device_vector_view[uint64_t, uint64_t]] new_indices) except +

    cdef void extend(
        const device_resources& handle,
        index[uint8_t, uint64_t]* index,
        device_matrix_view[uint8_t, uint64_t, row_major] new_vectors,
        optional[device_vector_view[uint64_t, uint64_t]] new_indices) except +

    cdef void search(
        const device_resources& handle,
        const index[float, uint64_t]& index,
        device_matrix_view[float, uint64_t, row_major] queries,
        device_matrix_view[uint64_t, uint64_t, row_major] neighbors,
        device_matrix_view[float, uint64_t, row_major] distances,
        const search_params& params,
        uint32_t k) except +

    cdef void search(
        const device_resources& handle,
        const index[int8_t, uint64_t]& index,
        device_matrix_view[int8_t, uint64_t, row_major] queries,
        device_matrix_view[uint64_t, uint64_t, row_major] neighbors,
        device_matrix_view[float, uint64_t, row_major] distances,
        const search_params& params,
        uint32_t k) except +

    cdef void search(
        const device_resources& handle,
        const index[uint8_t, uint64_t]& index,
        device_matrix_view[uint8_t, uint64_t, row_major] queries,
        device_matrix_view[uint64_t, uint64_t, row_major] neighbors,
        device_matrix_view[float, uint64_t, row_major] distances,
        const search_params& params,
        uint32_t k) except +

    # cdef void save(const device_resources& handle,
    #                const string& filename,
    #                const index[uint64_t]& index) except +

    # cdef void load(const device_resources& handle,
    #                const string& filename,
    #                index[uint64_t]* index) except +
