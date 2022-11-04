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

from libc.stdint cimport uintptr_t, int64_t
from cython.operator cimport dereference as deref
from pylibraft.common import Handle
from pylibraft.common.handle import auto_sync_handle
from pylibraft.common.handle cimport handle_t
from .rng_state cimport RngState

from libcpp cimport bool


cdef extern from "raft_distance/random/rmat_rectangular_generator.hpp" \
        namespace "raft::random::runtime":

    cdef void rmat_rectangular_gen(const handle_t &handle,
                                   int* out,
                                   int* out_src,
                                   int* out_dst,
                                   const float* theta,
                                   int r_scale,
                                   int c_scale,
                                   int n_edges,
                                   RngState& r)

    cdef void rmat_rectangular_gen(const handle_t &handle,
                                   int64_t* out,
                                   int64_t* out_src,
                                   int64_t* out_dst,
                                   const float* theta,
                                   int64_t r_scale,
                                   int64_t c_scale,
                                   int64_t n_edges,
                                   RngState& r)

    cdef void rmat_rectangular_gen(const handle_t &handle,
                                   int* out,
                                   int* out_src,
                                   int* out_dst,
                                   const double* theta,
                                   int r_scale,
                                   int c_scale,
                                   int n_edges,
                                   RngState& r)

    cdef void rmat_rectangular_gen(const handle_t &handle,
                                   int64_t* out,
                                   int64_t* out_src,
                                   int64_t* out_dst,
                                   const double* theta,
                                   int64_t r_scale,
                                   int64_t c_scale,
                                   int64_t n_edges,
                                   RngState& r)


@auto_sync_handle
def rmat(out, theta, r_scale, c_scale, seed=12345, handle=None):
    """
    Generate RMAT adjacency list based on the input distribution.

    Parameters
    ----------

    out: CUDA array interface compliant matrix shape (n_edges, 2). This will
         contain the src/dst node ids stored consecutively like a pair.
    theta: CUDA array interface compliant matrix shape
           (max(r_scale, c_scale) * 4) This stores the probability distribution
           at each RMAT level
    r_scale: log2 of number of source nodes
    c_scale: log2 of number of destination nodes
    seed: random seed used for reproducibility
    {handle_docstring}

    Examples
    --------

    .. code-block:: python

        import cupy as cp

        from pylibraft.common import Handle
        from pylibraft.random import rmat

        n_edges = 5000
        r_scale = 16
        c_scale = 14
        theta_len = max(r_scale, c_scale) * 4

        out = cp.empty((n_edges, 2), dtype=cp.int32)
        theta = cp.random.random_sample(theta_len, dtype=cp.float32)

        # A single RAFT handle can optionally be reused across
        # pylibraft functions.
        handle = Handle()
        ...
        rmat(out, theta, r_scale, c_scale, handle=handle)
        ...
        # pylibraft functions are often asynchronous so the
        # handle needs to be explicitly synchronized
        handle.sync()
   """

    if theta is None:
        raise Exception("'theta' cannot be None!")
    if out is None:
        raise Exception("'out' cannot be None!")

    out_cai = out.__cuda_array_interface__
    theta_cai = theta.__cuda_array_interface__

    n_edges = out_cai["shape"][0]
    out_ptr = <uintptr_t>out_cai["data"][0]
    theta_ptr = <uintptr_t>theta_cai["data"][0]
    out_dt = np.dtype(out_cai["typestr"])
    theta_dt = np.dtype(theta_cai["typestr"])

    cdef RngState *rng = new RngState(seed)

    handle = handle if handle is not None else Handle()
    cdef handle_t *h = <handle_t*><size_t>handle.getHandle()

    if out_dt == np.int32 and theta_dt == np.float32:
        rmat_rectangular_gen(deref(h),
                             <int*> out_ptr,
                             <int*> NULL,
                             <int*> NULL,
                             <float*> theta_ptr,
                             <int>r_scale,
                             <int>c_scale,
                             <int>n_edges,
                             deref(rng))
    elif out_dt == np.int64 and theta_dt == np.float32:
        rmat_rectangular_gen(deref(h),
                             <int64_t*> out_ptr,
                             <int64_t*> NULL,
                             <int64_t*> NULL,
                             <float*> theta_ptr,
                             <int64_t>r_scale,
                             <int64_t>c_scale,
                             <int64_t>n_edges,
                             deref(rng))
    elif out_dt == np.int32 and theta_dt == np.float64:
        rmat_rectangular_gen(deref(h),
                             <int*> out_ptr,
                             <int*> NULL,
                             <int*> NULL,
                             <double*> theta_ptr,
                             <int>r_scale,
                             <int>c_scale,
                             <int>n_edges,
                             deref(rng))
    elif out_dt == np.int64 and theta_dt == np.float64:
        rmat_rectangular_gen(deref(h),
                             <int64_t*> out_ptr,
                             <int64_t*> NULL,
                             <int64_t*> NULL,
                             <double*> theta_ptr,
                             <int64_t>r_scale,
                             <int64_t>c_scale,
                             <int64_t>n_edges,
                             deref(rng))
    else:
        raise ValueError("dtype out=%s and theta=%s not supported" %
                         (out_dt, theta_dt))
