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

from libc.stdint cimport uintptr_t
from cython.operator cimport dereference as deref

from pylibraft.distance.distance_type cimport DistanceType
from pylibraft.common.handle cimport handle_t
from pylibraft.distance.pairwise_distance import *


def distance(X, Y, dists, metric="euclidean"):
    """
    Compute pairwise distances between X and Y

    Parameters
    ----------

    X : CUDA array interface matrix shape (m, k)
    Y : CUDA array interface matrix shape (n, k)
    dists : Writable CUDA array interface matrix shape (m, n)
    metric : string denoting the metric type
    """

    # TODO: Validate inputs, shapes, etc...
    x_cai = X.__cuda_array_interface__
    y_cai = Y.__cuda_array_interface__
    dists_cai = dists.__cuda_array_interface__

    m = x_cai["shape"][0]
    n = y_cai["shape"][0]
    k = x_cai["shape"][1]

    x_ptr = <uintptr_t>x_cai["data"][0]
    y_ptr = <uintptr_t>y_cai["data"][0]
    d_ptr = <uintptr_t>dists_cai["data"][0]

    cdef handle_t *h = new handle_t()

    # TODO: Support single and double precision
    pairwise_distance(deref(h),
                      <float*> x_ptr,
                      <float*> y_ptr,
                      <float*> d_ptr,
                      <int>m,
                      <int>n,
                      <int>k,
                      <DistanceType>DistanceType.L2SqrtUnexpanded,
                      <bool>True, <float>0.0)
