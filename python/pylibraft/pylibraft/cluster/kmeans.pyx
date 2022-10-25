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
from pylibraft.distance.distance_type cimport DistanceType

from pylibraft.common import Handle
from pylibraft.common.handle cimport handle_t

from pylibraft.distance import DISTANCE_TYPES


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
            const double *centroids,
            const double *weight,
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
            const float *centroids,
            const float *weight,
            const float *l2norm_x,
            float *new_centroids,
            float *new_weight,
            DistanceType metric,
            int batch_samples,
            int batch_centroids);

def compute_new_centroids(X,
                     centroids,
                     weight,
                     l2norm_x,
                     new_centroids,
                     new_weight,
                     batch_samples,
                     batch_centroids,
                     metric="euclidean",
                     handle=None):
    """
    Compute new centroids given an input matrix and existing centroids

    Valid values for metric:
        ["euclidean", "sqeuclidean"]

    Parameters
    ----------

    X : Input CUDA array interface compliant matrix shape (m, k)
    centroids : Input CUDA array interface compliant matrix shape
                    (n_clusters, k)
    weight : Input CUDA array interface compliant matrix shape (n_clusters, 1)
    l2norm_x : Input CUDA array interface compliant matrix shape (m, 1)
    new_centroids : Writable CUDA array interface compliant matrix shape
                    (n_clusters, k)
    new_weight : Writable CUDA array interface compliant matrix shape
                 (n_clusters, 1)
    batch_samples : integer specifying the batch size for X to compute
                    distances in batches
    batch_centroids : integer specifying the batch size for centroids
                      to compute distances in batches
    handle : Optional RAFT handle for reusing expensive CUDA resources

    Examples
    --------

    .. code-block:: python

        import cupy as cp

        from pylibraft.common import Handle
        from pylibaft.cluster.kmeans import update_centroids
        from pylibraft.distance import fused_l2_nn_argmin

        # A single RAFT handle can optionally be reused across
        # pylibraft functions.
        handle = Handle()

        n_samples = 5000
        n_features = 50
        n_clusters = 3

        X = cp.random.random_sample((n_samples, n_features),
                                      dtype=cp.float32)
        centroids = cp.random.random_sample((n_clusters, n_features),
                                                dtype=cp.float32)

        argmin = cp.empty((n_samples, ), dtype=cp.int32)

        l2norm_x = cupy.linalg.norm(X, axis=0, ord=2)

        fused_l2_nn_argmin(centroids, X, argmin, handle=handle)

        weight, _ = cp.histogram(argmin, bins=cp.arange(0, n_clusters+1)).astype(cp.float32)

        new_weight = cp.empty((n_clusters, ), dtype=cp.float32)
        new_centroids = cp.empty((n_clusters, n_features), dtype=cp.float32)

        compute_new_centroids(X, centroids, weight, l2norm_x, new_centroids, new_weight, n_samples, n_clusters)

        # pylibraft functions are often asynchronous so the
        # handle needs to be explicitly synchronized
        handle.sync()
   """

    x_cai = X.__cuda_array_interface__
    centroids_cai = centroids.__cuda_array_interface__
    weight_cai = weight.__cuda_array_interface__
    l2norm_x_cai = l2norm_x.__cuda_array_interface__
    new_centroids_cai = new_centroids.__cuda_array_interface__
    new_weight_cai = new_weight.__cuda_array_interface__

    m = x_cai["shape"][0]
    n_clusters = centroids_cai["shape"][0]

    x_k = x_cai["shape"][1]
    centroids_k = centroids_cai["shape"][1]
    new_centroids_k = centroids_cai["shape"][1]

    if x_k != centroids_k:
        raise ValueError("Inputs must have same number of columns. "
                         "a=%s, b=%s" % (x_k, centroids_k))

    x_ptr = <uintptr_t>x_cai["data"][0]
    centroids_ptr = <uintptr_t>centroids_cai["data"][0]
    weight_ptr = <uintptr_t>weight_cai["data"][0]
    l2norm_x_ptr = <uintptr_t>l2norm_x_cai["data"][0]
    new_centroids_ptr = <uintptr_t>new_centroids_cai["data"][0]
    new_weight_ptr = <uintptr_t>new_weight_cai["data"][0]

    handle = handle if handle is not None else Handle()
    cdef handle_t *h = <handle_t*><size_t>handle.getHandle()

    x_dt = np.dtype(x_cai["typestr"])
    centroids_dt = np.dtype(centroids_cai["typestr"])
    weight_dt = np.dtype(weight_cai["typestr"])
    new_centroids_dt = np.dtype(new_centroids_cai["typestr"])
    new_weight_dt = np.dtype(new_weight_cai["typestr"])
    l2norm_x_dt = np.dtype(l2norm_x_cai["typestr"])

    x_c_contiguous = is_c_cont(x_cai, x_dt)
    centroids_c_contiguous = is_c_cont(centroids_cai, centroids_dt)
    new_centroids_c_contiguous = is_c_cont(new_centroids_cai, new_centroids_dt)

    if not x_c_contiguous or not centroids_c_contiguous \
        or not new_centroids_c_contiguous:
            raise ValueError("Inputs must all be c contiguous")

    cdef DistanceType distance_type = DISTANCE_TYPES[metric]

    if x_dt != centroids_dt or x_dt != weight_dt \
        or x_dt != new_centroids_dt or x_dt != new_weight_dt \
        or x_dt != l2norm_x_dt:
        raise ValueError("Inputs must all have the same dtypes "
                          "(float32 or float64)")

    if x_dt == np.float32:
        update_centroids(deref(h),
                          <float*> x_ptr,
                          <int> m,
                          <int> x_k,
                          <int> n_clusters,
                          <float*> centroids_ptr,
                          <float*> weight_ptr,
                          <float*> l2norm_x_ptr,
                          <float*> new_centroids_ptr,
                          <float*> new_weight_ptr,
                          <DistanceType>distance_type,
                          <int>batch_samples,
                          <int>batch_centroids)
    elif x_dt == np.float64:
        update_centroids(deref(h),
                          <double*> x_ptr,
                          <int> m,
                          <int> x_k,
                          <int> n_clusters,
                          <double*> centroids_ptr,
                          <double*> weight_ptr,
                          <double*> l2norm_x_ptr,
                          <double*> new_centroids_ptr,
                          <double*> new_weight_ptr,
                          <DistanceType>distance_type,
                          <int>batch_samples,
                          <int>batch_centroids)
    else:
        raise ValueError("dtype %s not supported" % x_dt)
