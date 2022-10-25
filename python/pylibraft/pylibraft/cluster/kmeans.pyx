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

from libc.stdint cimport uintptr_t
from cython.operator cimport dereference as deref

from libcpp cimport bool
from .distance_type cimport DistanceType

from pylibraft.common import Handle
from pylibraft.common.handle cimport handle_t


def is_c_cont(cai, dt):
    return "strides" not in cai or \
        cai["strides"] is None or \
        cai["strides"][1] == dt.itemsize


cdef extern from "raft_distance/kmeans.hpp" \
        namespace "raft::cluster::kmeans::runtime":

    cdef void update_centroids(
            const handle_t& handle,
            const double *X,
            int n_samples,
            int n_features,
            int n_clusters,
            const double *weight,
            const double *cur_centroids,
            const double *l2norm_x,
            double *new_centroids,
            double *new_weight,
            DistanceType metric,
            int batch_samples,
            int batch_centroids);

    cdef void update_centroids(
            const handle_t& handle,
            const float *X,
            int n_samples,
            int n_features,
            int n_clusters,
            const float *weight,
            const float *cur_centroids,
            const float *l2norm_x,
            float *new_centroids,
            float *new_weight,
            DistanceType metric,
            int batch_samples,
            int batch_centroids);

def update_centroids(X, cur_centroids, weight, l2norm_x,
                     new_centroids, new_weight,
                     metric="euclidean",
                     batch_samples, batch_centroids,
                     handle=None):
    """
    Compute pairwise distances between X and Y

    Valid values for metric:
        ["euclidean", "l2", "l1", "cityblock", "inner_product",
         "chebyshev", "canberra", "lp", "hellinger", "jensenshannon",
         "kl_divergence", "russellrao", "minkowski", "correlation",
         "cosine"]

    Parameters
    ----------

    X : CUDA array interface compliant matrix shape (m, k)
    Y : CUDA array interface compliant matrix shape (n, k)
    dists : Writable CUDA array interface matrix shape (m, n)
    metric : string denoting the metric type (default="euclidean")
    p : metric parameter (currently used only for "minkowski")
    handle : Optional RAFT handle for reusing expensive CUDA resources

    Examples
    --------

    .. code-block:: python

        import cupy as cp

        from pylibraft.common import Handle
        from pylibraft.distance import pairwise_distance

        n_samples = 5000
        n_features = 50

        in1 = cp.random.random_sample((n_samples, n_features),
                                      dtype=cp.float32)
        in2 = cp.random.random_sample((n_samples, n_features),
                                      dtype=cp.float32)
        output = cp.empty((n_samples, n_samples), dtype=cp.float32)

        # A single RAFT handle can optionally be reused across
        # pylibraft functions.
        handle = Handle()
        ...
        pairwise_distance(in1, in2, output, metric="euclidean", handle=handle)
        ...
        # pylibraft functions are often asynchronous so the
        # handle needs to be explicitly synchronized
        handle.sync()
   """

    x_cai = X.__cuda_array_interface__
    y_cai = Y.__cuda_array_interface__
    dists_cai = dists.__cuda_array_interface__

    m = x_cai["shape"][0]
    n = y_cai["shape"][0]

    x_k = x_cai["shape"][1]
    y_k = y_cai["shape"][1]

    if x_k != y_k:
        raise ValueError("Inputs must have same number of columns. "
                         "a=%s, b=%s" % (x_k, y_k))

    x_ptr = <uintptr_t>x_cai["data"][0]
    y_ptr = <uintptr_t>y_cai["data"][0]
    d_ptr = <uintptr_t>dists_cai["data"][0]

    handle = handle if handle is not None else Handle()
    cdef handle_t *h = <handle_t*><size_t>handle.getHandle()

    x_dt = np.dtype(x_cai["typestr"])
    y_dt = np.dtype(y_cai["typestr"])
    d_dt = np.dtype(dists_cai["typestr"])

    x_c_contiguous = is_c_cont(x_cai, x_dt)
    y_c_contiguous = is_c_cont(y_cai, y_dt)

    if x_c_contiguous != y_c_contiguous:
        raise ValueError("Inputs must have matching strides")

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
                          <int>x_k,
                          <DistanceType>distance_type,
                          <bool>x_c_contiguous,
                          <float>p)
    elif x_dt == np.float64:
        pairwise_distance(deref(h),
                          <double*> x_ptr,
                          <double*> y_ptr,
                          <double*> d_ptr,
                          <int>m,
                          <int>n,
                          <int>x_k,
                          <DistanceType>distance_type,
                          <bool>x_c_contiguous,
                          <float>p)
    else:
        raise ValueError("dtype %s not supported" % x_dt)
