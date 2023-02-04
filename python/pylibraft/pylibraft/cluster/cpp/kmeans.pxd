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

import numpy as np

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp cimport bool, nullptr

from pylibraft.cluster.cpp.kmeans_types cimport KMeansParams
from pylibraft.common.cpp.mdspan cimport *
from pylibraft.common.cpp.optional cimport optional
from pylibraft.common.handle cimport device_resources


cdef extern from "raft_runtime/cluster/kmeans.hpp" \
        namespace "raft::runtime::cluster::kmeans" nogil:

    cdef void update_centroids(
        const device_resources& handle,
        const double *X,
        int n_samples,
        int n_features,
        int n_clusters,
        const double *sample_weights,
        const double *centroids,
        const int* labels,
        double *new_centroids,
        double *weight_per_cluster) except +

    cdef void update_centroids(
        const device_resources& handle,
        const float *X,
        int n_samples,
        int n_features,
        int n_clusters,
        const float *sample_weights,
        const float *centroids,
        const int* labels,
        float *new_centroids,
        float *weight_per_cluster) except +

    cdef void cluster_cost(
        const device_resources& handle,
        const float* X,
        int n_samples,
        int n_features,
        int n_clusters,
        const float * centroids,
        float * cost) except +

    cdef void cluster_cost(
        const device_resources& handle,
        const double* X,
        int n_samples,
        int n_features,
        int n_clusters,
        const double * centroids,
        double * cost) except +

    cdef void init_plus_plus(
        const device_resources & handle,
        const KMeansParams& params,
        device_matrix_view[float, int, row_major] X,
        device_matrix_view[float, int, row_major] centroids) except +

    cdef void init_plus_plus(
        const device_resources & handle,
        const KMeansParams& params,
        device_matrix_view[double, int, row_major] X,
        device_matrix_view[double, int, row_major] centroids) except +

    cdef void fit(
        const device_resources & handle,
        const KMeansParams& params,
        device_matrix_view[float, int, row_major] X,
        optional[device_vector_view[float, int]] sample_weight,
        device_matrix_view[float, int, row_major] inertia,
        host_scalar_view[float, int] inertia,
        host_scalar_view[int, int] n_iter) except +

    cdef void fit(
        const device_resources & handle,
        const KMeansParams& params,
        device_matrix_view[double, int, row_major] X,
        optional[device_vector_view[double, int]] sample_weight,
        device_matrix_view[double, int, row_major] inertia,
        host_scalar_view[double, int] inertia,
        host_scalar_view[int, int] n_iter) except +
