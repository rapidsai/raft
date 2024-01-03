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

import numpy as np

import pylibraft.common.handle

from cython.operator cimport dereference as deref
from libc.stdint cimport int8_t, int64_t, uint8_t, uint32_t, uintptr_t
from libcpp cimport bool, nullptr
from libcpp.string cimport string

from rmm._lib.memory_resource cimport device_memory_resource

from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    device_vector_view,
    row_major,
)
from pylibraft.common.handle cimport device_resources
from pylibraft.common.optional cimport optional
from pylibraft.distance.distance_type cimport DistanceType


cdef extern from "library_types.h":
    ctypedef enum cudaDataType_t:
        CUDA_R_32F "CUDA_R_32F"  # float
        CUDA_R_16F "CUDA_R_16F"  # half

        # uint8 - used to refer to IVF-PQ's fp8 storage type
        CUDA_R_8U "CUDA_R_8U"

cdef extern from "raft/neighbors/ann_types.hpp" \
        namespace "raft::neighbors::ann" nogil:

    cdef cppclass ann_index "raft::neighbors::index":
        pass

    cdef cppclass ann_index_params "raft::spatial::knn::index_params":
        DistanceType metric
        float metric_arg
        bool add_data_on_build

    cdef cppclass ann_search_params "raft::spatial::knn::search_params":
        pass


cdef extern from "raft/neighbors/ivf_pq_types.hpp" \
        namespace "raft::neighbors::ivf_pq" nogil:

    ctypedef enum codebook_gen:
        PER_SUBSPACE "raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE",
        PER_CLUSTER "raft::neighbors::ivf_pq::codebook_gen::PER_CLUSTER"

    cpdef cppclass index_params(ann_index_params):
        uint32_t n_lists
        uint32_t kmeans_n_iters
        double kmeans_trainset_fraction
        uint32_t pq_bits
        uint32_t pq_dim
        codebook_gen codebook_kind
        bool force_random_rotation
        bool conservative_memory_allocation
        int random_seed

    cdef cppclass index[IdxT](ann_index):
        index(const device_resources& handle,
              DistanceType metric,
              codebook_gen codebook_kind,
              uint32_t n_lists,
              uint32_t dim,
              uint32_t pq_bits,
              uint32_t pq_dim,
              bool conservative_memory_allocation)

        IdxT size()
        uint32_t dim()
        uint32_t pq_dim()
        uint32_t pq_len()
        uint32_t pq_bits()
        DistanceType metric()
        uint32_t n_lists()
        uint32_t rot_dim()
        codebook_gen codebook_kind()
        bool conservative_memory_allocation()

    cpdef cppclass search_params(ann_search_params):
        uint32_t n_probes
        cudaDataType_t lut_dtype
        cudaDataType_t internal_distance_dtype


cdef extern from "raft_runtime/neighbors/ivf_pq.hpp" \
        namespace "raft::runtime::neighbors::ivf_pq" nogil:

    cdef void build(
        const device_resources& handle,
        const index_params& params,
        device_matrix_view[float, int64_t, row_major] dataset,
        index[int64_t]* index) except +

    cdef void build(
        const device_resources& handle,
        const index_params& params,
        device_matrix_view[int8_t, int64_t, row_major] dataset,
        index[int64_t]* index) except +

    cdef void build(
        const device_resources& handle,
        const index_params& params,
        device_matrix_view[uint8_t, int64_t, row_major] dataset,
        index[int64_t]* index) except +

    cdef void extend(
        const device_resources& handle,
        device_matrix_view[float, int64_t, row_major] new_vectors,
        optional[device_vector_view[int64_t, int64_t]] new_indices,
        index[int64_t]* index) except +

    cdef void extend(
        const device_resources& handle,
        device_matrix_view[int8_t, int64_t, row_major] new_vectors,
        optional[device_vector_view[int64_t, int64_t]] new_indices,
        index[int64_t]* index) except +

    cdef void extend(
        const device_resources& handle,
        device_matrix_view[uint8_t, int64_t, row_major] new_vectors,
        optional[device_vector_view[int64_t, int64_t]] new_indices,
        index[int64_t]* index) except +

    cdef void search(
        const device_resources& handle,
        const search_params& params,
        const index[int64_t]& index,
        device_matrix_view[float, int64_t, row_major] queries,
        device_matrix_view[int64_t, int64_t, row_major] neighbors,
        device_matrix_view[float, int64_t, row_major] distances) except +

    cdef void search(
        const device_resources& handle,
        const search_params& params,
        const index[int64_t]& index,
        device_matrix_view[int8_t, int64_t, row_major] queries,
        device_matrix_view[int64_t, int64_t, row_major] neighbors,
        device_matrix_view[float, int64_t, row_major] distances) except +

    cdef void search(
        const device_resources& handle,
        const search_params& params,
        const index[int64_t]& index,
        device_matrix_view[uint8_t, int64_t, row_major] queries,
        device_matrix_view[int64_t, int64_t, row_major] neighbors,
        device_matrix_view[float, int64_t, row_major] distances) except +

    cdef void serialize(const device_resources& handle,
                        const string& filename,
                        const index[int64_t]& index) except +

    cdef void deserialize(const device_resources& handle,
                          const string& filename,
                          index[int64_t]* index) except +
