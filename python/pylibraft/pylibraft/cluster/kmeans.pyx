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
from libcpp cimport nullptr

from pylibraft.common import Handle
from pylibraft.common.handle import auto_sync_handle
from pylibraft.common.handle cimport handle_t
from pylibraft.common.input_validation import *
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
        const double *sample_weights,
        const double *centroids,
        const int* labels,
        double *new_centroids,
        double *weight_per_cluster)

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
        float *weight_per_cluster)


@auto_sync_handle
def compute_new_centroids(X,
                          centroids,
                          labels,
                          new_centroids,
                          sample_weights=None,
                          weight_per_cluster=None,
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
    labels : Input CUDA array interface compliant matrix shape
               (m, 1)
    new_centroids : Writable CUDA array interface compliant matrix shape
                    (n_clusters, k)
    sample_weights : Optional input CUDA array interface compliant matrix shape
                     (n_clusters, 1) default: None
    weight_per_cluster : Optional writable CUDA array interface compliant
                         matrix shape (n_clusters, 1) default: None
    batch_samples : Optional integer specifying the batch size for X to compute
                    distances in batches. default: m
    batch_centroids : Optional integer specifying the batch size for centroids
                      to compute distances in batches. default: n_clusters
    {handle_docstring}

    Examples
    --------

    .. code-block:: python

        import cupy as cp

        from pylibraft.common import Handle
        from pylibraft.cluster.kmeans import compute_new_centroids
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

        labels = cp.random.randint(0, high=n_clusters, size=n_samples,
                                   dtype=cp.int32)

        new_centroids = cp.empty((n_clusters, n_features), dtype=cp.float32)

        compute_new_centroids(
            X, centroids, labels, new_centroids, handle=handle
        )

        # pylibraft functions are often asynchronous so the
        # handle needs to be explicitly synchronized
        handle.sync()
   """

    x_cai = X.__cuda_array_interface__
    centroids_cai = centroids.__cuda_array_interface__
    new_centroids_cai = new_centroids.__cuda_array_interface__
    labels_cai = labels.__cuda_array_interface__

    m = x_cai["shape"][0]
    x_k = x_cai["shape"][1]
    n_clusters = centroids_cai["shape"][0]

    centroids_k = centroids_cai["shape"][1]
    new_centroids_k = centroids_cai["shape"][1]

    x_dt = np.dtype(x_cai["typestr"])
    centroids_dt = np.dtype(centroids_cai["typestr"])
    new_centroids_dt = np.dtype(new_centroids_cai["typestr"])
    labels_dt = np.dtype(labels_cai["typestr"])

    if not do_cols_match(X, centroids):
        raise ValueError("X and centroids must have same number of columns.")

    if not do_rows_match(X, labels):
        raise ValueError("X and labels must have same number of rows")

    x_ptr = <uintptr_t>x_cai["data"][0]
    centroids_ptr = <uintptr_t>centroids_cai["data"][0]
    new_centroids_ptr = <uintptr_t>new_centroids_cai["data"][0]
    labels_ptr = <uintptr_t>labels_cai["data"][0]

    if sample_weights is not None:
        sample_weights_cai = sample_weights.__cuda_array_interface__
        sample_weights_ptr = <uintptr_t>sample_weights_cai["data"][0]
        sample_weights_dt = np.dtype(sample_weights_cai["typestr"])
    else:
        sample_weights_ptr = <uintptr_t>nullptr

    if weight_per_cluster is not None:
        weight_per_cluster_cai = weight_per_cluster.__cuda_array_interface__
        weight_per_cluster_ptr = <uintptr_t>weight_per_cluster_cai["data"][0]
        weight_per_cluster_dt = np.dtype(weight_per_cluster_cai["typestr"])
    else:
        weight_per_cluster_ptr = <uintptr_t>nullptr

    handle = handle if handle is not None else Handle()
    cdef handle_t *h = <handle_t*><size_t>handle.getHandle()

    x_c_contiguous = is_c_cont(x_cai, x_dt)
    centroids_c_contiguous = is_c_cont(centroids_cai, centroids_dt)
    new_centroids_c_contiguous = is_c_cont(new_centroids_cai, new_centroids_dt)

    if not x_c_contiguous or not centroids_c_contiguous \
            or not new_centroids_c_contiguous:
        raise ValueError("Inputs must all be c contiguous")

    if not do_dtypes_match(X, centroids, new_centroids):
        raise ValueError("Inputs must all have the same dtypes "
                         "(float32 or float64)")

    if x_dt == np.float32:
        update_centroids(deref(h),
                         <float*> x_ptr,
                         <int> m,
                         <int> x_k,
                         <int> n_clusters,
                         <float*> sample_weights_ptr,
                         <float*> centroids_ptr,
                         <int*> labels_ptr,
                         <float*> new_centroids_ptr,
                         <float*> weight_per_cluster_ptr)
    elif x_dt == np.float64:
        update_centroids(deref(h),
                         <double*> x_ptr,
                         <int> m,
                         <int> x_k,
                         <int> n_clusters,
                         <double*> sample_weights_ptr,
                         <double*> centroids_ptr,
                         <int*> labels_ptr,
                         <double*> new_centroids_ptr,
                         <double*> weight_per_cluster_ptr)
    else:
        raise ValueError("dtype %s not supported" % x_dt)
