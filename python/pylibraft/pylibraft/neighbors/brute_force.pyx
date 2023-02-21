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
from libcpp cimport bool, nullptr
from libcpp.vector cimport vector

from pylibraft.distance.distance_type cimport DistanceType

from pylibraft.common import (
    DeviceResources,
    auto_convert_output,
    cai_wrapper,
    device_ndarray,
)

from libc.stdint cimport int64_t, uint32_t, uintptr_t

from pylibraft.common.cpp.optional cimport optional
from pylibraft.common.handle cimport device_resources

from pylibraft.common.handle import auto_sync_handle
from pylibraft.common.input_validation import is_c_contiguous
from pylibraft.common.interruptible import cuda_interruptible

from pylibraft.distance.distance_type cimport DistanceType

# TODO: Centralize this

from pylibraft.neighbors.ivf_pq.ivf_pq import _get_metric

from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    host_matrix_view,
    make_device_matrix_view,
    make_host_matrix_view,
    row_major,
)
from pylibraft.neighbors.cpp.brute_force cimport knn as c_knn


cdef device_matrix_view[float, uint32_t, row_major] \
        make_device_matrix_view_float(array) except *:
    cai = cai_wrapper(array)
    if cai.dtype != np.float32:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    return make_device_matrix_view[float, uint32_t, row_major](
        <float*><uintptr_t>cai.data, cai.shape[0], cai.shape[1])

cdef device_matrix_view[int64_t, uint32_t, row_major] \
        make_device_matrix_view_int64(array) except *:
    cai = cai_wrapper(array)
    if cai.dtype != np.int64:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    return make_device_matrix_view[int64_t, uint32_t, row_major](
        <int64_t*><uintptr_t>cai.data, cai.shape[0], cai.shape[1])


def _get_array_params(array_interface, check_dtype=None):
    dtype = np.dtype(array_interface["typestr"])
    if check_dtype is None and dtype != check_dtype:
        raise TypeError("dtype %s not supported" % dtype)
    shape = array_interface["shape"]
    if len(shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(shape))
    data = array_interface["data"][0]
    return (shape, dtype, data)


@auto_sync_handle
@auto_convert_output
def knn(dataset, queries, k=None, indices=None, distances=None,
        metric="sqeuclidean", metric_arg=2.0,
        global_id_offset=0, handle=None):
    """
    Perform a brute-force nearest neighbors search.

    Parameters
    ----------
    dataset : array interface compliant matrix, row-major layout,
        shape (n_samples, dim). Supported dtype [float]
    queries : array interface compliant matrix, row-major layout,
        shape (n_queries, dim) Supported dtype [float]
    k : int
        Number of neighbors to search (k <= 2048). Optional if indices or
        distances arrays are given (in which case their second dimension
        is k).
    indices :  Optional array interface compliant matrix shape
                (n_queries, k), dtype int64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
        Supported dtype uint64
    distances :  Optional array interface compliant matrix shape
                (n_queries, k), dtype float. If supplied, neighbor
                indices will be written here in-place. (default None)

    {handle_docstring}

    Returns
    -------
    indices: array interface compliant object containing resulting indices
             shape (n_queries, k)

    distances: array interface compliant object containing resulting distances
               shape (n_queries, k)

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
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 40
    >>> distances, neighbors = knn(dataset, queries, k)
    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)

    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    """

    if handle is None:
        handle = DeviceResources()

    dataset_cai = cai_wrapper(dataset)
    queries_cai = cai_wrapper(queries)

    cdef device_matrix_view[const float, uint32_t, row_major] dataset_view = \
        make_device_matrix_view[float, uint32_t, row_major](
            <const float*><uintptr_t>dataset_cai.data, dataset_cai.shape[0],
            dataset_cai.shape[1])(dataset)

    cdef device_matrix_view[const float, uint32_t, row_major] queries_view = \
        make_device_matrix_view[float, uint32_t, row_major](
            <const float*><uintptr_t>queries_cai.data, queries_cai.shape[0],
            queries_cai.shape[1])(dataset)

    if k is None:
        if indices is not None:
            k = cai_wrapper(indices).shape[1]
        elif distances is not None:
            k = cai_wrapper(distances).shape[1]
        else:
            raise ValueError("Argument k must be specified if both indices "
                             "and distances arg is None")

    n_queries = cai_wrapper(queries).shape[0]

    if indices is None:
        indices = device_ndarray.empty((n_queries, k), dtype='int64_t')

    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype='float32')

    cdef device_matrix_view[float, uint32_t, row_major] distances_view = \
        make_device_matrix_view_float(queries)

    cdef device_matrix_view[int64_t, uint32_t, row_major] indices_view = \
        make_device_matrix_view_int64(queries)

    cdef DistanceType c_metric = _get_metric(metric)

    dataset_cai = cai_wrapper(dataset)

    cdef optional[float] c_metric_arg = metric_arg
    cdef optional[int64_t] c_global_offset = global_id_offset

    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    cdef vector[device_matrix_view[const float, uint32_t, row_major]] \
        dataset_vec
    dataset_vec.push_back(dataset_view)

    if dataset_cai.dtype == np.float32:
        with cuda_interruptible():
            c_knn(deref(handle_),
                  dataset_vec,
                  queries_view,
                  indices_view,
                  distances_view,
                  k,
                  c_metric,
                  c_metric_arg,
                  c_global_offset)
    else:
        raise TypeError("dtype %s not supported" % dataset_cai.dtype)

    return (distances, indices)
