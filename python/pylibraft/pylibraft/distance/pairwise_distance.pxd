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

from libcpp cimport bool
from pylibraft.distance.distance_type cimport DistanceType
from pylibraft.common.handle cimport handle_t

cdef extern from "raft_distance/pairwise_distance.hpp" \
    namespace "raft::distance::runtime":

    cdef void pairwise_distance(const handle_t &handle,
                                float *x,
                                float *y,
                                float *dists,
                                int m,
                                int n,
                                int k,
                                DistanceType metric,
                                bool isRowMajor,
                                float metric_arg)

    cdef void pairwise_distance(const handle_t &handle,
                                double *x,
                                double *y,
                                double *dists,
                                int m,
                                int n,
                                int k,
                                DistanceType metric,
                                bool isRowMajor,
                                float metric_arg)
