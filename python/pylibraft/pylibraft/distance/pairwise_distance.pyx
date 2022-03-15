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
import numpy as np

from libc.stdint cimport uintptr_t
from cython.operator cimport dereference as deref

from libcpp cimport bool
from .distance_type cimport DistanceType
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

DISTANCE_TYPES = {
    "l2": DistanceType.L2SqrtUnexpanded,
    "euclidean": DistanceType.L2SqrtUnexpanded,
    "l1": DistanceType.L1,
    "cityblock": DistanceType.L1,
    "inner_product": DistanceType.InnerProduct,
    "chebyshev": DistanceType.Linf,
    "canberra": DistanceType.Canberra,
    "lp": DistanceType.LpUnexpanded,
    "correlation": DistanceType.CorrelationExpanded,
    "jaccard": DistanceType.JaccardExpanded,
    "hellinger": DistanceType.HellingerExpanded,
    "braycurtis": DistanceType.BrayCurtis,
    "jensenshannon": DistanceType.JensenShannon,
    "hamming": DistanceType.HammingUnexpanded,
    "kl_divergence": DistanceType.KLDivergence,
    "russellrao": DistanceType.RusselRaoExpanded,
    "dice": DistanceType.DiceExpanded
}

SUPPORTED_DISTANCES = list(DISTANCE_TYPES.keys())


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
    x_dt = np.dtype(x_cai["typestr"])
    y_dt = np.dtype(y_cai["typestr"])
    d_dt = np.dtype(dists_cai["typestr"])

    if metric not in SUPPORTED_DISTANCES:
        raise ValueError("metric %s is not supported" % metric)

    cdef DistanceType distance_type = DISTANCE_TYPES[metric]

    if x_dt != y_dt or x_dt != d_dt:
        raise ValueError("Inputs must have the same dtypes")

    if x_dt == np.float32:
        pairwise_distance(deref(h),
                          <float*> x_ptr,
                          <float*> y_ptr,
                          <float*> d_ptr,
                          <int>m,
                          <int>n,
                          <int>k,
                          <DistanceType>distance_type,
                          <bool>True, <float>0.0)
    elif x_dt == np.float64:
        pairwise_distance(deref(h),
                          <double*> x_ptr,
                          <double*> y_ptr,
                          <double*> d_ptr,
                          <int>m,
                          <int>n,
                          <int>k,
                          <DistanceType>distance_type,
                          <bool>True, <float>0.0)
    else:
        raise ValueError("dtype %s not supported" % x_dt)
