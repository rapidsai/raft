#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

from pylibraft.common import (
    Handle,
    auto_convert_output,
    cai_wrapper,
    device_ndarray,
)
from pylibraft.common.handle import auto_sync_handle

from pylibraft.common.handle cimport device_resources


cdef extern from "raft_runtime/distance/fused_distance_nn.hpp" \
        namespace "raft::runtime::distance" nogil:

    void fused_distance_nn_min_arg(
        const device_resources &handle,
        int* min,
        const float* x,
        const float* y,
        int m,
        int n,
        int k,
        bool sqrt,
        DistanceType metric,
        bool isRowMajor,
        float metric_arg) except +


DISTANCE_TYPES = {
    "l2": DistanceType.L2SqrtExpanded,
    "sqeuclidean": DistanceType.L2Expanded,
    "euclidean": DistanceType.L2SqrtExpanded,
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

SUPPORTED_DISTANCES = ["euclidean", "l2", "cosine", "sqeuclidean"]


@auto_sync_handle
@auto_convert_output
def fused_distance_nn_argmin(X, Y, out=None, sqrt=True, metric="euclidean",
                             handle=None):
    """
    Compute the 1-nearest neighbors between X and Y using the distance metrics

    Valid values for metric:
        ["euclidean", "l2", "cosine", "sqeuclidean"]

    Parameters
    ----------

    X : CUDA array interface compliant matrix shape (m, k)
    Y : CUDA array interface compliant matrix shape (n, k)
    out : Writable CUDA array interface matrix shape (m, 1)
    metric : string denoting the metric type (default="euclidean")

    {handle_docstring}

    Examples
    --------
    To compute the 1-nearest neighbors argmin:

    >>> import cupy as cp
    >>> from pylibraft.common import Handle
    >>> from pylibraft.distance import fused_distance_nn_argmin
    >>> n_samples = 5000
    >>> n_clusters = 5
    >>> n_features = 50
    >>> in1 = cp.random.random_sample((n_samples, n_features),
    ...                               dtype=cp.float32)
    >>> in2 = cp.random.random_sample((n_clusters, n_features),
    ...                               dtype=cp.float32)
    >>> # A single RAFT handle can optionally be reused across
    >>> # pylibraft functions.
    >>> handle = Handle()

    >>> output = fused_distance_nn_argmin(in1, in2, handle=handle)

    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()

    The output can also be computed in-place on a preallocated
    array:

    >>> import cupy as cp
    >>> from pylibraft.common import Handle
    >>> from pylibraft.distance import fused_distance_nn_argmin
    >>> n_samples = 5000
    >>> n_clusters = 5
    >>> n_features = 50
    >>> in1 = cp.random.random_sample((n_samples, n_features),
    ...                               dtype=cp.float32)
    >>> in2 = cp.random.random_sample((n_clusters, n_features),
    ...                               dtype=cp.float32)
    >>> output = cp.empty((n_samples, 1), dtype=cp.int32)
    >>> # A single RAFT handle can optionally be reused across
    >>> # pylibraft functions.
    >>> handle = Handle()

    >>> fused_distance_nn_argmin(in1, in2, out=output, handle=handle)
    array(...)

    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
   """

    x_cai = cai_wrapper(X)
    y_cai = cai_wrapper(Y)

    x_dt = x_cai.dtype
    y_dt = y_cai.dtype

    m = x_cai.shape[0]
    n = y_cai.shape[0]

    if out is None:
        output = device_ndarray.empty((m,), dtype="int32")
    else:
        output = out

    output_cai = cai_wrapper(output)

    x_k = x_cai.shape[1]
    y_k = y_cai.shape[1]

    if x_k != y_k:
        raise ValueError("Inputs must have same number of columns. "
                         "a=%s, b=%s" % (x_k, y_k))

    if metric not in SUPPORTED_DISTANCES:
        raise ValueError("metric %s is not supported" % metric)

    cdef DistanceType distance_type = DISTANCE_TYPES[metric]

    x_ptr = <uintptr_t>x_cai.data
    y_ptr = <uintptr_t>y_cai.data

    d_ptr = <uintptr_t>output_cai.data

    handle = handle if handle is not None else Handle()
    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    d_dt = output_cai.dtype

    x_c_contiguous = x_cai.c_contiguous
    y_c_contiguous = y_cai.c_contiguous

    if x_c_contiguous != y_c_contiguous:
        raise ValueError("Inputs must have matching strides")

    if not x_c_contiguous:
        raise ValueError("Inputs must be C contiguous")

    if x_dt != y_dt:
        raise ValueError("Inputs must have the same dtypes")
    if d_dt != np.int32:
        raise ValueError("Output array must be int32")
    # unused arg for now.
    metric_arg = 0.0
    if x_dt == np.float32:
        fused_distance_nn_min_arg(deref(h),
                                  <int*> d_ptr,
                                  <float*> x_ptr,
                                  <float*> y_ptr,
                                  <int>m,
                                  <int>n,
                                  <int>x_k,
                                  <bool>sqrt,
                                  <DistanceType>distance_type,
                                  <bool>x_c_contiguous,
                                  <float>metric_arg)
    else:
        raise ValueError("dtype %s not supported" % x_dt)

    return output
