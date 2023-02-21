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
from libcpp cimport bool

from .distance_type cimport DistanceType

from pylibraft.common import Handle
from pylibraft.common.handle import auto_sync_handle

from pylibraft.common.handle cimport device_resources

from pylibraft.common import auto_convert_output, cai_wrapper, device_ndarray


cdef extern from "raft_runtime/distance/pairwise_distance.hpp" \
        namespace "raft::runtime::distance" nogil:

    cdef void pairwise_distance(const device_resources &handle,
                                float *x,
                                float *y,
                                float *dists,
                                int m,
                                int n,
                                int k,
                                DistanceType metric,
                                bool isRowMajor,
                                float metric_arg) except +

    cdef void pairwise_distance(const device_resources &handle,
                                double *x,
                                double *y,
                                double *dists,
                                int m,
                                int n,
                                int k,
                                DistanceType metric,
                                bool isRowMajor,
                                float metric_arg) except +

DISTANCE_TYPES = {
    "l2": DistanceType.L2SqrtUnexpanded,
    "sqeuclidean": DistanceType.L2Unexpanded,
    "euclidean": DistanceType.L2SqrtUnexpanded,
    "l1": DistanceType.L1,
    "cityblock": DistanceType.L1,
    "inner_product": DistanceType.InnerProduct,
    "chebyshev": DistanceType.Linf,
    "canberra": DistanceType.Canberra,
    "cosine": DistanceType.CosineExpanded,
    "lp": DistanceType.LpUnexpanded,
    "correlation": DistanceType.CorrelationExpanded,
    "jaccard": DistanceType.JaccardExpanded,
    "hellinger": DistanceType.HellingerExpanded,
    "braycurtis": DistanceType.BrayCurtis,
    "jensenshannon": DistanceType.JensenShannon,
    "hamming": DistanceType.HammingUnexpanded,
    "kl_divergence": DistanceType.KLDivergence,
    "minkowski": DistanceType.LpUnexpanded,
    "russellrao": DistanceType.RusselRaoExpanded,
    "dice": DistanceType.DiceExpanded,
}

SUPPORTED_DISTANCES = ["euclidean", "l1", "cityblock", "l2", "inner_product",
                       "chebyshev", "minkowski", "canberra", "kl_divergence",
                       "correlation", "russellrao", "hellinger", "lp",
                       "hamming", "jensenshannon", "cosine", "sqeuclidean"]


@auto_sync_handle
@auto_convert_output
def distance(X, Y, out=None, metric="euclidean", p=2.0, handle=None):
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
    out : Optional writable CUDA array interface matrix shape (m, n)
    metric : string denoting the metric type (default="euclidean")
    p : metric parameter (currently used only for "minkowski")
    {handle_docstring}

    Returns
    -------

    raft.device_ndarray containing pairwise distances

    Examples
    --------
    To compute pairwise distances on cupy arrays:

    >>> import cupy as cp
    >>> from pylibraft.common import Handle
    >>> from pylibraft.distance import pairwise_distance
    >>> n_samples = 5000
    >>> n_features = 50
    >>> in1 = cp.random.random_sample((n_samples, n_features),
    ...                               dtype=cp.float32)
    >>> in2 = cp.random.random_sample((n_samples, n_features),
    ...                               dtype=cp.float32)

    A single RAFT handle can optionally be reused across
    pylibraft functions.

    >>> handle = Handle()
    >>> output = pairwise_distance(in1, in2, metric="euclidean", handle=handle)

    pylibraft functions are often asynchronous so the
    handle needs to be explicitly synchronized

    >>> handle.sync()

    It's also possible to write to a pre-allocated output array:

    >>> import cupy as cp
    >>> from pylibraft.common import Handle
    >>> from pylibraft.distance import pairwise_distance
    >>> n_samples = 5000
    >>> n_features = 50
    >>> in1 = cp.random.random_sample((n_samples, n_features),
    ...                              dtype=cp.float32)
    >>> in2 = cp.random.random_sample((n_samples, n_features),
    ...                              dtype=cp.float32)
    >>> output = cp.empty((n_samples, n_samples), dtype=cp.float32)

    A single RAFT handle can optionally be reused across
    pylibraft functions.

    >>>
    >>> handle = Handle()
    >>> pairwise_distance(in1, in2, out=output,
    ...                  metric="euclidean", handle=handle)
    array(...)

    pylibraft functions are often asynchronous so the
    handle needs to be explicitly synchronized

    >>> handle.sync()
    """

    x_cai = cai_wrapper(X)
    y_cai = cai_wrapper(Y)

    m = x_cai.shape[0]
    n = y_cai.shape[0]

    x_dt = x_cai.dtype
    y_dt = y_cai.dtype

    if out is None:
        dists = device_ndarray.empty((m, n), dtype=y_dt)
    else:
        dists = out

    x_k = x_cai.shape[1]
    y_k = y_cai.shape[1]

    dists_cai = cai_wrapper(dists)

    if x_k != y_k:
        raise ValueError("Inputs must have same number of columns. "
                         "a=%s, b=%s" % (x_k, y_k))

    x_ptr = <uintptr_t>x_cai.data
    y_ptr = <uintptr_t>y_cai.data
    d_ptr = <uintptr_t>dists_cai.data

    handle = handle if handle is not None else Handle()
    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    d_dt = dists_cai.dtype

    x_c_contiguous = x_cai.c_contiguous
    y_c_contiguous = y_cai.c_contiguous

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

    return dists
