#
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
from libcpp cimport bool, nullptr
from libcpp.vector cimport vector

from pylibraft.common import (
    DeviceResources,
    auto_convert_output,
    cai_wrapper,
    device_ndarray,
)

from libc.stdint cimport int64_t, uintptr_t

from pylibraft.common.cpp.optional cimport optional
from pylibraft.common.handle cimport device_resources
from pylibraft.common.mdspan cimport get_dmv_bool, get_dmv_float, get_dmv_int64

from pylibraft.common.handle import auto_sync_handle
from pylibraft.common.interruptible import cuda_interruptible
from pylibraft.neighbors.common import _check_input_array, _get_metric

cimport pylibraft.neighbors.cpp.eps_neighborhood as c_eps_neighborhood
from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    device_vector_view,
    host_matrix_view,
    make_device_matrix_view,
    make_device_vector_view,
    make_host_matrix_view,
    row_major,
)
from pylibraft.neighbors.cpp.eps_neighborhood cimport (
    BallCoverIndex as c_BallCoverIndex,
    build_rbc_index as c_build_rbc_index,
    eps_neighbors_l2 as c_eps_neighbors_l2,
    eps_neighbors_l2_rbc as c_eps_neighbors_l2_rbc,
    eps_neighbors_l2_rbc_pass1 as c_eps_neighbors_l2_rbc_pass1,
    eps_neighbors_l2_rbc_pass2 as c_eps_neighbors_l2_rbc_pass2,
)


cdef class RbcIndex:
    cdef readonly bool trained
    cdef str data_type

    def __cinit__(self):
        self.trained = False
        self.data_type = None


cdef class RbcIndexFloat(RbcIndex):
    cdef c_BallCoverIndex[int64_t, float, int64_t, int64_t]* index

    def __cinit__(self, dataset, handle):
        cdef device_resources* handle_ = \
            <device_resources*><size_t>handle.getHandle()
        self.index = new c_BallCoverIndex[int64_t, float, int64_t, int64_t](
            deref(handle_),
            get_dmv_float(dataset, check_shape=True),
            _get_metric("euclidean"))


@auto_sync_handle
@auto_convert_output
def build_rbc_index(dataset, handle=None):
    """
    Builds a random ball cover index from dataset using the L2-norm.

    Parameters
    ----------
    dataset : array interface compliant matrix, row-major layout,
        shape (n_samples, dim). Supported dtype [float]
    {handle_docstring}

    Returns
    -------
    index : Index

    Examples
    --------
    see 'eps_neighbors_sparse'

    """
    if handle is None:
        handle = DeviceResources()

    dataset_cai = cai_wrapper(dataset)

    # we require c-contiguous (rowmajor) inputs here
    _check_input_array(dataset_cai, [np.dtype("float32")])

    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    cdef RbcIndexFloat rbc_index_float

    if dataset_cai.dtype == np.float32:
        rbc_index_float = RbcIndexFloat(dataset=dataset_cai, handle=handle)
        rbc_index_float.data_type = "float32"
        with cuda_interruptible():
            c_build_rbc_index(
                deref(handle_),
                deref(rbc_index_float.index))
        rbc_index_float.trained = True
        return rbc_index_float
    else:
        raise TypeError("dtype %s not supported" % dataset_cai.dtype)


@auto_sync_handle
@auto_convert_output
def eps_neighbors(dataset, queries, eps, method="brute", handle=None):
    """
    Perform an epsilon neighborhood search using the L2-norm.

    Parameters
    ----------
    dataset : array interface compliant matrix, row-major layout,
        shape (n_samples, dim). Supported dtype [float]
    queries : array interface compliant matrix, row-major layout,
        shape (n_queries, dim) Supported dtype [float]
    eps : threshold
    method : string, default = "brute"
        Valid values ["brute", "ball_tree"]
    {handle_docstring}

    Returns
    -------
    adj: array interface compliant object containing bool adjacency mask
         shape (n_queries, n_samples)

    vd: array interface compliant object containing row sums of adj
        shape (n_queries + 1). vd[n_queries] contains the total sum

    Examples
    --------
    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors.eps_neighborhood import eps_neighbors
    >>> handle = DeviceResources()
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> eps = 0.1
    >>> adj, vd = eps_neighbors(dataset, queries, eps, handle=handle)
    >>> adj = cp.asarray(adj)
    >>> vd = cp.asarray(vd)
    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
    """

    if handle is None:
        handle = DeviceResources()

    dataset_cai = cai_wrapper(dataset)
    queries_cai = cai_wrapper(queries)

    # we require c-contiguous (rowmajor) inputs here
    _check_input_array(dataset_cai, [np.dtype("float32")])
    _check_input_array(queries_cai, [np.dtype("float32")],
                       exp_cols=dataset_cai.shape[1])

    n_queries = queries_cai.shape[0]
    n_samples = dataset_cai.shape[0]

    adj = device_ndarray.empty((n_queries, n_samples), dtype='bool')
    vd = device_ndarray.empty((n_queries + 1, ), dtype='int64')
    adj_cai = cai_wrapper(adj)
    vd_cai = cai_wrapper(vd)

    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    vd_vector_view = make_device_vector_view(
        <int64_t *><uintptr_t>vd_cai.data, <int64_t>vd_cai.shape[0])

    if dataset_cai.dtype == np.float32:
        with cuda_interruptible():
            if method == "ball_cover":
                c_eps_neighbors_l2_rbc(
                    deref(handle_),
                    get_dmv_float(dataset_cai, check_shape=True),
                    get_dmv_float(queries_cai, check_shape=True),
                    get_dmv_bool(adj_cai, check_shape=True),
                    vd_vector_view,
                    eps)
            elif method == "brute":
                c_eps_neighbors_l2(
                    deref(handle_),
                    get_dmv_float(dataset_cai, check_shape=True),
                    get_dmv_float(queries_cai, check_shape=True),
                    get_dmv_bool(adj_cai, check_shape=True),
                    vd_vector_view,
                    eps)
            else:
                raise ValueError("Unsupported method %s" % method)
    else:
        raise TypeError("dtype %s not supported" % dataset_cai.dtype)

    return (adj, vd)


@auto_sync_handle
@auto_convert_output
def eps_neighbors_sparse(RbcIndex rbc_index, queries, eps, handle=None):
    """
    Perform an epsilon neighborhood search with random ball cover (rbc)
    using the L2-norm.

    Parameters
    ----------
    rbc_index : RbcIndex created via 'build_rbc_index'.
        Supported dtype [float]
    queries : array interface compliant matrix, row-major layout,
        shape (n_queries, dim) Supported dtype [float]
    eps : threshold
    {handle_docstring}

    Returns
    -------
    adj_ia: array interface compliant object containing row indices for
            adj_ja

    adj_ja: array interface compliant object containing adjacency mask
            column indices

    vd: array interface compliant object containing row sums of adj
        shape (n_queries + 1). vd[n_queries] contains the total sum

    Examples
    --------
    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors.eps_neighborhood import eps_neighbors_sparse
    >>> from pylibraft.neighbors.eps_neighborhood import build_rbc_index
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> eps = 0.1
    >>> handle = DeviceResources()
    >>> rbc_index = build_rbc_index(dataset)
    >>> adj_ia, adj_ja, vd = eps_neighbors_sparse(rbc_index, queries, eps)
    >>> adj_ia = cp.asarray(adj_ia)
    >>> adj_ja = cp.asarray(adj_ja)
    >>> vd = cp.asarray(vd)
    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
    """
    if not rbc_index.trained:
        raise ValueError("Index need to be built before calling extend.")

    if handle is None:
        handle = DeviceResources()

    queries_cai = cai_wrapper(queries)

    _check_input_array(queries_cai, [np.dtype(rbc_index.data_type)])

    n_queries = queries_cai.shape[0]

    adj_ia = device_ndarray.empty((n_queries + 1, ), dtype='int64')
    vd = device_ndarray.empty((n_queries + 1, ), dtype='int64')
    adj_ia_cai = cai_wrapper(adj_ia)
    vd_cai = cai_wrapper(vd)

    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    vd_vector_view = make_device_vector_view(
        <int64_t *><uintptr_t>vd_cai.data, <int64_t>vd_cai.shape[0])
    adj_ia_vector_view = make_device_vector_view(
        <int64_t *><uintptr_t>adj_ia_cai.data, <int64_t>adj_ia_cai.shape[0])

    cdef RbcIndexFloat rbc_index_float

    if queries_cai.dtype == np.float32:
        rbc_index_float = rbc_index
        with cuda_interruptible():
            c_eps_neighbors_l2_rbc_pass1(
                deref(handle_),
                deref(rbc_index_float.index),
                get_dmv_float(queries_cai, check_shape=True),
                adj_ia_vector_view,
                vd_vector_view,
                eps)
    else:
        raise TypeError("dtype %s not supported" % queries_cai.dtype)

    handle.sync()
    n_nnz = adj_ia.copy_to_host()[n_queries]
    adj_ja = device_ndarray.empty((n_nnz, ), dtype='int64')
    adj_ja_cai = cai_wrapper(adj_ja)
    adj_ja_vector_view = make_device_vector_view(
        <int64_t *><uintptr_t>adj_ja_cai.data, <int64_t>adj_ja_cai.shape[0])

    if queries_cai.dtype == np.float32:
        with cuda_interruptible():
            c_eps_neighbors_l2_rbc_pass2(
                deref(handle_),
                deref(rbc_index_float.index),
                get_dmv_float(queries_cai, check_shape=True),
                adj_ia_vector_view,
                adj_ja_vector_view,
                vd_vector_view,
                eps)
    else:
        raise TypeError("dtype %s not supported" % queries_cai.dtype)

    return (adj_ia, adj_ja, vd)
