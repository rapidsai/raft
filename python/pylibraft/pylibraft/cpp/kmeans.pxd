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

from pylibraft.common.handle cimport handle_t


cdef extern from "raft_distance/cluster/kmeans.hpp" \
        namespace "raft::cluster::kmeans::runtime":

    cdef void update_centroids(
        const handle_t& handle,
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
        const handle_t& handle,
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
        const handle_t& handle,
        const float* X,
        int n_samples,
        int n_features,
        int n_clusters,
        const float * centroids,
        float * cost) except +

    cdef void cluster_cost(
        const handle_t& handle,
        const double* X,
        int n_samples,
        int n_features,
        int n_clusters,
        const double * centroids,
        double * cost) except +
