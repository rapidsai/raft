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
from libc.stdint cimport int8_t, int64_t, uint8_t, uintptr_t
from libcpp cimport bool, nullptr

from pylibraft.distance.distance_type cimport DistanceType

from pylibraft.common import (
    DeviceResources,
    auto_convert_output,
    cai_wrapper,
    device_ndarray,
)

from pylibraft.common.handle cimport device_resources

from pylibraft.common.handle import auto_sync_handle
from pylibraft.common.input_validation import is_c_contiguous
from pylibraft.common.interruptible import cuda_interruptible

from pylibraft.distance.distance_type cimport DistanceType

import pylibraft.neighbors.ivf_pq as ivf_pq
from pylibraft.neighbors.common import _get_metric

cimport pylibraft.neighbors.ivf_pq.cpp.c_ivf_pq as c_ivf_pq
from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    host_matrix_view,
    make_host_matrix_view,
    row_major,
)
from pylibraft.common.mdspan cimport (
    get_dmv_float,
    get_dmv_int8,
    get_dmv_int64,
    get_dmv_uint8,
)
from pylibraft.neighbors.common cimport _get_metric_string
from pylibraft.neighbors.ivf_pq.cpp.c_ivf_pq cimport (
    index_params,
    search_params,
)


# We omit the const qualifiers in the interface for refine, because cython
# has an issue parsing it (https://github.com/cython/cython/issues/4180).
cdef extern from "raft_runtime/neighbors/refine.hpp" \
        namespace "raft::runtime::neighbors" nogil:

    cdef void c_refine "raft::runtime::neighbors::refine" (
        const device_resources& handle,
        device_matrix_view[float, int64_t, row_major] dataset,
        device_matrix_view[float, int64_t, row_major] queries,
        device_matrix_view[int64_t, int64_t, row_major] candidates,
        device_matrix_view[int64_t, int64_t, row_major] indices,
        device_matrix_view[float, int64_t, row_major] distances,
        DistanceType metric) except +

    cdef void c_refine "raft::runtime::neighbors::refine" (
        const device_resources& handle,
        device_matrix_view[uint8_t, int64_t, row_major] dataset,
        device_matrix_view[uint8_t, int64_t, row_major] queries,
        device_matrix_view[int64_t, int64_t, row_major] candidates,
        device_matrix_view[int64_t, int64_t, row_major] indices,
        device_matrix_view[float, int64_t, row_major] distances,
        DistanceType metric) except +

    cdef void c_refine "raft::runtime::neighbors::refine" (
        const device_resources& handle,
        device_matrix_view[int8_t, int64_t, row_major] dataset,
        device_matrix_view[int8_t, int64_t, row_major] queries,
        device_matrix_view[int64_t, int64_t, row_major] candidates,
        device_matrix_view[int64_t, int64_t, row_major] indices,
        device_matrix_view[float, int64_t, row_major] distances,
        DistanceType metric) except +

    cdef void c_refine "raft::runtime::neighbors::refine" (
        const device_resources& handle,
        host_matrix_view[float, int64_t, row_major] dataset,
        host_matrix_view[float, int64_t, row_major] queries,
        host_matrix_view[int64_t, int64_t, row_major] candidates,
        host_matrix_view[int64_t, int64_t, row_major] indices,
        host_matrix_view[float, int64_t, row_major] distances,
        DistanceType metric) except +

    cdef void c_refine "raft::runtime::neighbors::refine" (
        const device_resources& handle,
        host_matrix_view[uint8_t, int64_t, row_major] dataset,
        host_matrix_view[uint8_t, int64_t, row_major] queries,
        host_matrix_view[int64_t, int64_t, row_major] candidates,
        host_matrix_view[int64_t, int64_t, row_major] indices,
        host_matrix_view[float, int64_t, row_major] distances,
        DistanceType metric) except +

    cdef void c_refine "raft::runtime::neighbors::refine" (
        const device_resources& handle,
        host_matrix_view[int8_t, int64_t, row_major] dataset,
        host_matrix_view[int8_t, int64_t, row_major] queries,
        host_matrix_view[int64_t, int64_t, row_major] candidates,
        host_matrix_view[int64_t, int64_t, row_major] indices,
        host_matrix_view[float, int64_t, row_major] distances,
        DistanceType metric) except +


def _get_array_params(array_interface, check_dtype=None):
    dtype = np.dtype(array_interface["typestr"])
    if check_dtype is None and dtype != check_dtype:
        raise TypeError("dtype %s not supported" % dtype)
    shape = array_interface["shape"]
    if len(shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(shape))
    data = array_interface["data"][0]
    return (shape, dtype, data)


cdef host_matrix_view[float, int64_t, row_major] \
        get_host_matrix_view_float(array) except *:
    shape, dtype, data = _get_array_params(
        array.__array_interface__, check_dtype=np.float32)
    return make_host_matrix_view[float, int64_t, row_major](
        <float*><uintptr_t>data, shape[0], shape[1])


cdef host_matrix_view[int64_t, int64_t, row_major] \
        get_host_matrix_view_int64_t(array) except *:
    shape, dtype, data = _get_array_params(
        array.__array_interface__, check_dtype=np.int64)
    return make_host_matrix_view[int64_t, int64_t, row_major](
        <int64_t*><uintptr_t>data, shape[0], shape[1])


cdef host_matrix_view[uint8_t, int64_t, row_major] \
        get_host_matrix_view_uint8(array) except *:
    shape, dtype, data = _get_array_params(
        array.__array_interface__, check_dtype=np.uint8)
    return make_host_matrix_view[uint8_t, int64_t, row_major](
        <uint8_t*><uintptr_t>data, shape[0], shape[1])


cdef host_matrix_view[int8_t, int64_t, row_major] \
        get_host_matrix_view_int8(array) except *:
    shape, dtype, data = _get_array_params(
        array.__array_interface__, check_dtype=np.int8)
    return make_host_matrix_view[int8_t, int64_t, row_major](
        <int8_t*><uintptr_t>data, shape[0], shape[1])


@auto_sync_handle
@auto_convert_output
def refine(dataset, queries, candidates, k=None, indices=None, distances=None,
           metric="sqeuclidean", handle=None):
    """
    Refine nearest neighbor search.

    Refinement is an operation that follows an approximate NN search. The
    approximate search has already selected n_candidates neighbor candidates
    for each query. We narrow it down to k neighbors. For each query, we
    calculate the exact distance between the query and its n_candidates
    neighbor candidate, and select the k nearest ones.

    Input arrays can be either CUDA array interface compliant matrices or
    array interface compliant matrices in host memory. All array must be in
    the same memory space.

    Parameters
    ----------
    index_params : IndexParams object
    dataset : array interface compliant matrix, shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    queries : array interface compliant matrix, shape (n_queries, dim)
        Supported dtype [float, int8, uint8]
    candidates : array interface compliant matrix, shape (n_queries, k0)
        dtype int64
    k : int
        Number of neighbors to search (k <= k0). Optional if indices or
        distances arrays are given (in which case their second dimension
        is k).
    indices :  Optional array interface compliant matrix shape
                (n_queries, k), dtype int64. If supplied, neighbor
                indices will be written here in-place. (default None)
        Supported dtype int64
    distances :  Optional array interface compliant matrix shape
                (n_queries, k), dtype float. If supplied, neighbor
                indices will be written here in-place. (default None)

    {handle_docstring}

    Returns
    -------
    index: ivf_pq.Index

    Examples
    --------

    >>> import cupy as cp

    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import ivf_pq, refine

    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000

    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> handle = DeviceResources()
    >>> index_params = ivf_pq.IndexParams(n_lists=1024, metric="sqeuclidean",
    ...                                   pq_dim=10)
    >>> index = ivf_pq.build(index_params, dataset, handle=handle)

    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 40
    >>> _, candidates = ivf_pq.search(ivf_pq.SearchParams(), index,
    ...                               queries, k, handle=handle)

    >>> k = 10
    >>> distances, neighbors = refine(dataset, queries, candidates, k,
    ...                               handle=handle)
    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)

    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
    """

    if handle is None:
        handle = DeviceResources()

    if hasattr(dataset, "__cuda_array_interface__"):
        return _refine_device(dataset, queries, candidates, k, indices,
                              distances, metric, handle)
    else:
        return _refine_host(dataset, queries, candidates, k, indices,
                            distances, metric, handle)


def _refine_device(dataset, queries, candidates, k, indices, distances,
                   metric, handle):
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    if k is None:
        if indices is not None:
            k = cai_wrapper(indices).shape[1]
        elif distances is not None:
            k = cai_wrapper(distances).shape[1]
        else:
            raise ValueError("Argument k must be specified if both indices "
                             "and distances arg is None")

    queries_cai = cai_wrapper(queries)
    dataset_cai = cai_wrapper(dataset)
    candidates_cai = cai_wrapper(candidates)
    n_queries = cai_wrapper(queries).shape[0]

    if indices is None:
        indices = device_ndarray.empty((n_queries, k), dtype='int64')

    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype='float32')

    indices_cai = cai_wrapper(indices)
    distances_cai = cai_wrapper(distances)

    cdef DistanceType c_metric = _get_metric(metric)

    if dataset_cai.dtype == np.float32:
        with cuda_interruptible():
            c_refine(deref(handle_),
                     get_dmv_float(dataset_cai, check_shape=True),
                     get_dmv_float(queries_cai, check_shape=True),
                     get_dmv_int64(candidates_cai, check_shape=True),
                     get_dmv_int64(indices_cai, check_shape=True),
                     get_dmv_float(distances_cai, check_shape=True),
                     c_metric)
    elif dataset_cai.dtype == np.int8:
        with cuda_interruptible():
            c_refine(deref(handle_),
                     get_dmv_int8(dataset_cai, check_shape=True),
                     get_dmv_int8(queries_cai, check_shape=True),
                     get_dmv_int64(candidates_cai, check_shape=True),
                     get_dmv_int64(indices_cai, check_shape=True),
                     get_dmv_float(distances_cai, check_shape=True),
                     c_metric)
    elif dataset_cai.dtype == np.uint8:
        with cuda_interruptible():
            c_refine(deref(handle_),
                     get_dmv_uint8(dataset_cai, check_shape=True),
                     get_dmv_uint8(queries_cai, check_shape=True),
                     get_dmv_int64(candidates_cai, check_shape=True),
                     get_dmv_int64(indices_cai, check_shape=True),
                     get_dmv_float(distances_cai, check_shape=True),
                     c_metric)
    else:
        raise TypeError("dtype %s not supported" % dataset_cai.dtype)

    return (distances, indices)


def _refine_host(dataset, queries, candidates, k, indices, distances,
                 metric, handle):
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    if k is None:
        if indices is not None:
            k = indices.__array_interface__["shape"][1]
        elif distances is not None:
            k = distances.__array_interface__["shape"][1]
        else:
            raise ValueError("Argument k must be specified if both indices "
                             "and distances arg is None")

    n_queries = queries.__array_interface__["shape"][0]

    if indices is None:
        indices = np.empty((n_queries, k), dtype='int64')

    if distances is None:
        distances = np.empty((n_queries, k), dtype='float32')

    cdef DistanceType c_metric = _get_metric(metric)

    dtype = np.dtype(dataset.__array_interface__["typestr"])

    if dtype == np.float32:
        with cuda_interruptible():
            c_refine(deref(handle_),
                     get_host_matrix_view_float(dataset),
                     get_host_matrix_view_float(queries),
                     get_host_matrix_view_int64_t(candidates),
                     get_host_matrix_view_int64_t(indices),
                     get_host_matrix_view_float(distances),
                     c_metric)
    elif dtype == np.int8:
        with cuda_interruptible():
            c_refine(deref(handle_),
                     get_host_matrix_view_int8(dataset),
                     get_host_matrix_view_int8(queries),
                     get_host_matrix_view_int64_t(candidates),
                     get_host_matrix_view_int64_t(indices),
                     get_host_matrix_view_float(distances),
                     c_metric)
    elif dtype == np.uint8:
        with cuda_interruptible():
            c_refine(deref(handle_),
                     get_host_matrix_view_uint8(dataset),
                     get_host_matrix_view_uint8(queries),
                     get_host_matrix_view_int64_t(candidates),
                     get_host_matrix_view_int64_t(indices),
                     get_host_matrix_view_float(distances),
                     c_metric)
    else:
        raise TypeError("dtype %s not supported" % dtype)

    return (distances, indices)
