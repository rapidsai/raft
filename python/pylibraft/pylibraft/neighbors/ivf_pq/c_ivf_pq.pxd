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

from rmm._lib.memory_resource cimport device_memory_resource

from pylibraft.common.handle cimport handle_t
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

    cdef cppclass index[IdxT](ann_index):
        index(const handle_t& handle,
              DistanceType metric,
              codebook_gen codebook_kind,
              uint32_t n_lists,
              uint32_t dim,
              uint32_t pq_bits,
              uint32_t pq_dim,
              uint32_t n_nonempty_lists)

        IdxT size()
        uint32_t dim()
        uint32_t pq_dim()
        uint32_t pq_len()
        uint32_t pq_bits()
        DistanceType metric()
        uint32_t n_lists()
        uint32_t rot_dim()
        codebook_gen codebook_kind()

    cpdef cppclass search_params(ann_search_params):
        uint32_t n_probes
        cudaDataType_t lut_dtype
        cudaDataType_t internal_distance_dtype


cdef extern from "raft/neighbors/specializations/ivf_pq_specialization.hpp" \
        namespace "raft::neighbors::ivf_pq" nogil:

    cdef void build(const handle_t& handle,
                    const index_params& params,
                    const float* dataset,
                    uint64_t n_rows,
                    uint32_t dim,
                    index[uint64_t]* index) except +

    cdef void build(const handle_t& handle,
                    const index_params& params,
                    const int8_t* dataset,
                    uint64_t n_rows,
                    uint32_t dim,
                    index[uint64_t]* index) except +

    cdef void build(const handle_t& handle,
                    const index_params& params,
                    const uint8_t* dataset,
                    uint64_t n_rows,
                    uint32_t dim,
                    index[uint64_t]* index) except +

    cdef void extend(const handle_t& handle,
                     index[uint64_t]* index,
                     const float* new_vectors,
                     const uint64_t* new_indices,
                     uint64_t n_rows) except +

    cdef void extend(const handle_t& handle,
                     index[uint64_t]* index,
                     const int8_t* new_vectors,
                     const uint64_t* new_indices,
                     uint64_t n_rows) except +

    cdef void extend(const handle_t& handle,
                     index[uint64_t]* index,
                     const uint8_t* new_vectors,
                     const uint64_t* new_indices,
                     uint64_t n_rows) except +

    cdef void search(const handle_t& handle,
                     const search_params& params,
                     const index[uint64_t]& index,
                     const float* queries,
                     uint32_t n_queries,
                     uint32_t k,
                     uint64_t* neighbors,
                     float* distances,
                     device_memory_resource* mr) except +

    cdef void search(const handle_t& handle,
                     const search_params& params,
                     const index[uint64_t]& index,
                     const int8_t* queries,
                     uint32_t n_queries,
                     uint32_t k,
                     uint64_t* neighbors,
                     float* distances,
                     device_memory_resource* mr) except +

    cdef void search(const handle_t& handle,
                     const search_params& params,
                     const index[uint64_t]& index,
                     const uint8_t* queries,
                     uint32_t n_queries,
                     uint32_t k,
                     uint64_t* neighbors,
                     float* distances,
                     device_memory_resource* mr) except +
