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


from libcpp cimport bool

from pylibraft.distance.distance_type cimport DistanceType
from pylibraft.random.cpp.rng_state cimport RngState


cdef extern from "raft/cluster/kmeans_types.hpp" \
        namespace "raft::cluster::kmeans":

    ctypedef enum InitMethod 'raft::cluster::KMeansParams::InitMethod':
        KMeansPlusPlus 'raft::cluster::kmeans::KMeansParams::InitMethod::KMeansPlusPlus' # noqa
        Random 'raft::cluster::kmeans::KMeansParams::InitMethod::Random'
        Array 'raft::cluster::kmeans::KMeansParams::InitMethod::Array'

    cdef cppclass KMeansParams:
        KMeansParams() except +
        int n_clusters
        InitMethod init
        int max_iter
        double tol
        int verbosity
        RngState rng_state
        DistanceType metric
        int n_init
        double oversampling_factor
        int batch_samples
        int batch_centroids
        bool inertia_check
