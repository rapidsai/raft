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
from pylibraft.common.handle import auto_sync_handle
from pylibraft.common.handle cimport handle_t


def is_c_cont(cai, dt):
    return "strides" not in cai or \
        cai["strides"] is None or \
        cai["strides"][1] == dt.itemsize


cdef extern from "raft_distance/fused_l2_min_arg.hpp" \
        namespace "raft::distance::runtime":

    void fused_l2_nn_min_arg(
        const handle_t &handle,
        int* min,
        const float* x,
        const float* y,
        int m,
        int n,
        int k,
        bool sqrt)

    void fused_l2_nn_min_arg(
        const handle_t &handle,
        int* min,
        const double* x,
        const double* y,
        int m,
        int n,
        int k,
        bool sqrt)


@auto_sync_handle
def fused_l2_nn_argmin(X, Y, output, sqrt=True, handle=None):
    """
    Compute the 1-nearest neighbors between X and Y using the L2 distance

    Parameters
    ----------

    X : CUDA array interface compliant matrix shape (m, k)
    Y : CUDA array interface compliant matrix shape (n, k)
    output : Writable CUDA array interface matrix shape (m, 1)
    handle : Optional RAFT handle for reusing expensive CUDA resources

    Examples
    --------

    .. code-block:: python

        import cupy as cp

        from pylibraft.common import Handle
        from pylibraft.distance import fused_l2_nn

        n_samples = 5000
        n_clusters = 5
        n_features = 50

        in1 = cp.random.random_sample((n_samples, n_features),
                                      dtype=cp.float32)
        in2 = cp.random.random_sample((n_clusters, n_features),
                                      dtype=cp.float32)
        output = cp.empty((n_samples, 1), dtype=cp.int32)

        # A single RAFT handle can optionally be reused across
        # pylibraft functions.
        handle = Handle()
        ...
        fused_l2_nn_argmin(in1, in2, output, handle=handle)
        ...
        # pylibraft functions are often asynchronous so the
        # handle needs to be explicitly synchronized
        handle.sync()
   """

    x_cai = X.__cuda_array_interface__
    y_cai = Y.__cuda_array_interface__
    output_cai = output.__cuda_array_interface__

    m = x_cai["shape"][0]
    n = y_cai["shape"][0]

    x_k = x_cai["shape"][1]
    y_k = y_cai["shape"][1]

    if x_k != y_k:
        raise ValueError("Inputs must have same number of columns. "
                         "a=%s, b=%s" % (x_k, y_k))

    x_ptr = <uintptr_t>x_cai["data"][0]
    y_ptr = <uintptr_t>y_cai["data"][0]

    d_ptr = <uintptr_t>output_cai["data"][0]

    handle = handle if handle is not None else Handle()
    cdef handle_t *h = <handle_t*><size_t>handle.getHandle()

    x_dt = np.dtype(x_cai["typestr"])
    y_dt = np.dtype(y_cai["typestr"])
    d_dt = np.dtype(output_cai["typestr"])

    x_c_contiguous = is_c_cont(x_cai, x_dt)
    y_c_contiguous = is_c_cont(y_cai, y_dt)

    if x_c_contiguous != y_c_contiguous:
        raise ValueError("Inputs must have matching strides")

    if x_dt != y_dt:
        raise ValueError("Inputs must have the same dtypes")
    if d_dt != np.int32:
        raise ValueError("Output array must be int32")

    if x_dt == np.float32:
        fused_l2_nn_min_arg(deref(h),
                            <int*> d_ptr,
                            <float*> x_ptr,
                            <float*> y_ptr,
                            <int>m,
                            <int>n,
                            <int>x_k,
                            <bool>sqrt)
    elif x_dt == np.float64:
        fused_l2_nn_min_arg(deref(h),
                            <int*> d_ptr,
                            <double*> x_ptr,
                            <double*> y_ptr,
                            <int>m,
                            <int>n,
                            <int>x_k,
                            <bool>sqrt)
    else:
        raise ValueError("dtype %s not supported" % x_dt)
