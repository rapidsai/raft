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
from libcpp cimport nullptr

from collections import namedtuple
from enum import IntEnum

from pylibraft.common import Handle, cai_wrapper, device_ndarray
from pylibraft.common.handle import auto_sync_handle

from pylibraft.common.handle cimport device_resources
from pylibraft.random.cpp.rng_state cimport RngState

from pylibraft.common.input_validation import *
from pylibraft.distance import DISTANCE_TYPES

from pylibraft.cluster.cpp cimport kmeans as cpp_kmeans, kmeans_types
from pylibraft.cluster.cpp.kmeans cimport (
    cluster_cost as cpp_cluster_cost,
    init_plus_plus as cpp_init_plus_plus,
    update_centroids,
)
from pylibraft.common.cpp.mdspan cimport *
from pylibraft.common.cpp.optional cimport optional
from pylibraft.common.handle cimport device_resources

from pylibraft.common import auto_convert_output


@auto_sync_handle
@auto_convert_output
def compute_new_centroids(X,
                          centroids,
                          labels,
                          new_centroids,
                          sample_weights=None,
                          weight_per_cluster=None,
                          handle=None):
    """
    Compute new centroids given an input matrix and existing centroids

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

    >>> import cupy as cp
    >>> from pylibraft.common import Handle
    >>> from pylibraft.cluster.kmeans import compute_new_centroids
    >>> # A single RAFT handle can optionally be reused across
    >>> # pylibraft functions.
    >>> handle = Handle()
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 3
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                               dtype=cp.float32)
    >>> centroids = cp.random.random_sample((n_clusters, n_features),
    ...                                         dtype=cp.float32)
    ...
    >>> labels = cp.random.randint(0, high=n_clusters, size=n_samples,
    ...                            dtype=cp.int32)
    >>> new_centroids = cp.empty((n_clusters, n_features),
    ...                          dtype=cp.float32)
    >>> compute_new_centroids(
    ...     X, centroids, labels, new_centroids, handle=handle
    ... )
    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
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
    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    x_c_contiguous = is_c_contiguous(x_cai)
    centroids_c_contiguous = is_c_contiguous(centroids_cai)
    new_centroids_c_contiguous = is_c_contiguous(new_centroids_cai)

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


@auto_sync_handle
@auto_convert_output
def init_plus_plus(X, n_clusters=None, seed=None, handle=None, centroids=None):
    """
    Compute initial centroids using the "kmeans++" algorithm.

    Parameters
    ----------

    X : Input CUDA array interface compliant matrix shape (m, k)
    n_clusters : Number of clusters to select
    seed : Controls the random sampling of centroids
    centroids : Optional writable CUDA array interface compliant matrix shape
                (n_clusters, k). Use instead of passing `n_clusters`.
    {handle_docstring}

    Examples
    --------

    >>> import cupy as cp
    >>> from pylibraft.cluster.kmeans import init_plus_plus
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 3
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                               dtype=cp.float32)

    >>> centroids = init_plus_plus(X, n_clusters)
    """
    if (n_clusters is not None and
            centroids is not None and n_clusters != centroids.shape[0]):
        msg = ("Parameters 'n_clusters' and 'centroids' "
               "are exclusive. Only pass one at a time.")
        raise RuntimeError(msg)

    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    X_cai = cai_wrapper(X)
    X_cai.validate_shape_dtype(expected_dims=2)
    dtype = X_cai.dtype

    if centroids is not None:
        n_clusters = centroids.shape[0]
    else:
        centroids_shape = (n_clusters, X_cai.shape[1])
        centroids = device_ndarray.empty(centroids_shape, dtype=dtype)

    centroids_cai = cai_wrapper(centroids)

    # Can't set attributes of KMeansParameters after creating it, so taking
    # a detour via a dict to collect the possible constructor arguments
    params_ = dict(n_clusters=n_clusters)
    if seed is not None:
        params_["seed"] = seed
    params = KMeansParams(**params_)

    if dtype == np.float64:
        cpp_init_plus_plus(
            deref(h), params.c_obj,
            make_device_matrix_view[double, int, row_major](
                <double *><uintptr_t>X_cai.data,
                <int>X_cai.shape[0], <int>X_cai.shape[1]),
            make_device_matrix_view[double, int, row_major](
                <double *><uintptr_t>centroids_cai.data,
                <int>centroids_cai.shape[0], <int>centroids_cai.shape[1]),
        )
    elif dtype == np.float32:
        cpp_init_plus_plus(
            deref(h), params.c_obj,
            make_device_matrix_view[float, int, row_major](
                <float *><uintptr_t>X_cai.data,
                <int>X_cai.shape[0], <int>X_cai.shape[1]),
            make_device_matrix_view[float, int, row_major](
                <float *><uintptr_t>centroids_cai.data,
                <int>centroids_cai.shape[0], <int>centroids_cai.shape[1]),
        )
    else:
        raise ValueError(f"Unhandled dtype ({dtype}) for X.")

    return centroids


@auto_sync_handle
@auto_convert_output
def cluster_cost(X, centroids, handle=None):
    """
    Compute cluster cost given an input matrix and existing centroids

    Parameters
    ----------
    X : Input CUDA array interface compliant matrix shape (m, k)
    centroids : Input CUDA array interface compliant matrix shape
                    (n_clusters, k)
    {handle_docstring}

    Examples
    --------

    >>> import cupy as cp
    >>> from pylibraft.cluster.kmeans import cluster_cost
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 3
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                             dtype=cp.float32)
    >>> centroids = cp.random.random_sample((n_clusters, n_features),
    ...                                      dtype=cp.float32)
    >>> inertia = cluster_cost(X, centroids)
    """
    x_cai = X.__cuda_array_interface__
    centroids_cai = centroids.__cuda_array_interface__

    m = x_cai["shape"][0]
    x_k = x_cai["shape"][1]
    n_clusters = centroids_cai["shape"][0]

    centroids_k = centroids_cai["shape"][1]

    x_dt = np.dtype(x_cai["typestr"])
    centroids_dt = np.dtype(centroids_cai["typestr"])

    if not do_cols_match(X, centroids):
        raise ValueError("X and centroids must have same number of columns.")

    x_ptr = <uintptr_t>x_cai["data"][0]
    centroids_ptr = <uintptr_t>centroids_cai["data"][0]

    handle = handle if handle is not None else Handle()
    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    x_c_contiguous = is_c_contiguous(x_cai)
    centroids_c_contiguous = is_c_contiguous(centroids_cai)

    if not x_c_contiguous or not centroids_c_contiguous:
        raise ValueError("Inputs must all be c contiguous")

    if not do_dtypes_match(X, centroids):
        raise ValueError("Inputs must all have the same dtypes "
                         "(float32 or float64)")

    cdef float f_cost = 0
    cdef double d_cost = 0

    if x_dt == np.float32:
        cpp_cluster_cost(deref(h),
                         <float*> x_ptr,
                         <int> m,
                         <int> x_k,
                         <int> n_clusters,
                         <float*> centroids_ptr,
                         <float*> &f_cost)
        return f_cost
    elif x_dt == np.float64:
        cpp_cluster_cost(deref(h),
                         <double*> x_ptr,
                         <int> m,
                         <int> x_k,
                         <int> n_clusters,
                         <double*> centroids_ptr,
                         <double*> &d_cost)
        return d_cost
    else:
        raise ValueError("dtype %s not supported" % x_dt)


class InitMethod(IntEnum):
    """ Method for initializing kmeans """
    KMeansPlusPlus = <int> kmeans_types.InitMethod.KMeansPlusPlus
    Random = <int> kmeans_types.InitMethod.Random
    Array = <int> kmeans_types.InitMethod.Array


cdef class KMeansParams:
    """ Specifies hyper-parameters for the kmeans algorithm.

    Parameters
    ----------
    n_clusters : int, optional
        The number of clusters to form as well as the number of centroids
        to generate
    max_iter : int, optional
        Maximum number of iterations of the k-means algorithm for a single run
    tol : float, optional
        Relative tolerance with regards to inertia to declare convergence
    verbosity : int, optional
    seed: int, optional
        Seed to the random number generator.
    metric : str, optional
        Metric names to use for distance computation, see
        :func:`pylibraft.distance.pairwise_distance` for valid values.
    init : InitMethod, optional
    n_init : int, optional
        Number of instance k-means algorithm will be run with different seeds.
    oversampling_factor : float, optional
        Oversampling factor for use in the k-means algorithm
    """
    cdef kmeans_types.KMeansParams c_obj

    def __init__(self,
                 n_clusters: Optional[int] = None,
                 max_iter: Optional[int] = None,
                 tol: Optional[float] = None,
                 verbosity: Optional[int] = None,
                 seed: Optional[int] = None,
                 metric: Optional[str] = None,
                 init: Optional[InitMethod] = None,
                 n_init: Optional[int] = None,
                 oversampling_factor: Optional[float] = None,
                 batch_samples: Optional[int] = None,
                 batch_centroids: Optional[int] = None,
                 inertia_check: Optional[bool] = None):
        if n_clusters is not None:
            self.c_obj.n_clusters = n_clusters
        if max_iter is not None:
            self.c_obj.max_iter = max_iter
        if tol is not None:
            self.c_obj.tol = tol
        if verbosity is not None:
            self.c_obj.verbosity = verbosity
        if seed is not None:
            self.c_obj.rng_state.seed = seed
        if metric is not None:
            distance = DISTANCE_TYPES.get(metric)
            if distance is None:
                valid_metrics = list(DISTANCE_TYPES.keys())
                raise ValueError(f"Unknown metric '{metric}'. Valid values "
                                 f"are: {valid_metrics}")
            self.c_obj.metric = distance
        if init is not None:
            self.c_obj.init = init
        if n_init is not None:
            self.c_obj.n_init = n_init
        if oversampling_factor is not None:
            self.c_obj.oversampling_factor = oversampling_factor
        if batch_samples is not None:
            self.c_obj.batch_samples = batch_samples
        if batch_centroids is not None:
            self.c_obj.batch_centroids = batch_centroids
        if inertia_check is not None:
            self.c_obj.inertia_check = inertia_check

    @property
    def n_clusters(self):
        return self.c_obj.n_clusters

    @property
    def max_iter(self):
        return self.c_obj.max_iter

    @property
    def tol(self):
        return self.c_obj.tol

    @property
    def verbosity(self):
        return self.c_obj.verbosity

    @property
    def seed(self):
        return self.c_obj.rng_state.seed

    @property
    def init(self):
        return InitMethod(self.c_obj.init)

    @property
    def oversampling_factor(self):
        return self.c_obj.oversampling_factor

    @property
    def batch_samples(self):
        return self.c_obj.batch_samples

    @property
    def batch_centroids(self):
        return self.c_obj.batch_centroids

    @property
    def inertia_check(self):
        return self.c_obj.inertia_check

FitOutput = namedtuple("FitOutput", "centroids inertia n_iter")


@auto_sync_handle
@auto_convert_output
def fit(
    KMeansParams params, X, centroids=None, sample_weights=None, handle=None
):
    """
    Find clusters with the k-means algorithm

    Parameters
    ----------

    params : KMeansParams
        Parameters to use to fit KMeans model
    X : Input CUDA array interface compliant matrix shape (m, k)
    centroids : Optional writable CUDA array interface compliant matrix
                shape (n_clusters, k)
    sample_weights : Optional input CUDA array interface compliant matrix shape
                     (n_clusters, 1) default: None
    {handle_docstring}

    Returns
    -------
    centroids : raft.device_ndarray
        The computed centroids for each cluster
    inertia : float
       Sum of squared distances of samples to their closest cluster center
    n_iter : int
        The number of iterations used to fit the model

    Examples
    --------

    >>> import cupy as cp
    >>> from pylibraft.cluster.kmeans import fit, KMeansParams
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 3
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                             dtype=cp.float32)

    >>> params = KMeansParams(n_clusters=n_clusters)
    >>> centroids, inertia, n_iter = fit(params, X)
    """
    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    cdef float f_inertia = 0.0
    cdef double d_inertia = 0.0
    cdef int n_iter = 0

    cdef optional[device_vector_view[const double, int]] d_sample_weights
    cdef optional[device_vector_view[const float, int]] f_sample_weights

    X_cai = cai_wrapper(X)
    dtype = X_cai.dtype

    if centroids is None:
        centroids_shape = (params.n_clusters, X_cai.shape[1])
        centroids = device_ndarray.empty(centroids_shape, dtype=dtype)
    centroids_cai = cai_wrapper(centroids)

    # validate inputs have are all c-contiguous, and have a consistent dtype
    # and expected shape
    X_cai.validate_shape_dtype(2)
    centroids_cai.validate_shape_dtype(2, dtype)
    if sample_weights is not None:
        sample_weights_cai = cai_wrapper(sample_weights)
        sample_weights_cai.validate_shape_dtype(1, dtype)

    if dtype == np.float64:
        if sample_weights is not None:
            d_sample_weights = make_device_vector_view(
                <const double *><uintptr_t>sample_weights_cai.data,
                <int>sample_weights_cai.shape[0])

        cpp_kmeans.fit(
            deref(h),
            params.c_obj,
            make_device_matrix_view[double, int, row_major](
                <double *><uintptr_t>X_cai.data,
                <int>X_cai.shape[0], <int>X_cai.shape[1]),
            d_sample_weights,
            make_device_matrix_view[double, int, row_major](
                <double *><uintptr_t>centroids_cai.data,
                <int>centroids_cai.shape[0], <int>centroids_cai.shape[1]),
            make_host_scalar_view[double, int](&d_inertia),
            make_host_scalar_view[int, int](&n_iter))
        return FitOutput(centroids, d_inertia, n_iter)

    elif dtype == np.float32:
        if sample_weights is not None:
            f_sample_weights = make_device_vector_view(
                <const float *><uintptr_t>sample_weights_cai.data,
                <int>sample_weights_cai.shape[0])

        cpp_kmeans.fit(
            deref(h),
            params.c_obj,
            make_device_matrix_view[float, int, row_major](
                <float *><uintptr_t>X_cai.data,
                <int>X_cai.shape[0], <int>X_cai.shape[1]),
            f_sample_weights,
            make_device_matrix_view[float, int, row_major](
                <float *><uintptr_t>centroids_cai.data,
                <int>centroids_cai.shape[0], <int>centroids_cai.shape[1]),
            make_host_scalar_view[float, int](&f_inertia),
            make_host_scalar_view[int, int](&n_iter))
        return FitOutput(centroids, f_inertia, n_iter)

    else:
        raise ValueError(f"unhandled dtype {dtype}")
