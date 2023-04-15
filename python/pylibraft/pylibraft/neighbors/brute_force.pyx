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

from libc.stdint cimport int64_t, uintptr_t

from pylibraft.common.cpp.optional cimport optional
from pylibraft.common.handle cimport device_resources
from pylibraft.common.mdspan cimport get_dmv_float, get_dmv_int64

from pylibraft.common.handle import auto_sync_handle
from pylibraft.common.interruptible import cuda_interruptible

from pylibraft.distance.distance_type cimport DistanceType

# TODO: Centralize this

from pylibraft.distance.pairwise_distance import DISTANCE_TYPES

from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    host_matrix_view,
    make_device_matrix_view,
    make_host_matrix_view,
    row_major,
)
from pylibraft.neighbors.cpp.brute_force cimport knn as c_knn


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
    >>> from pylibraft.neighbors.brute_force import knn

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
    """

    if handle is None:
        handle = DeviceResources()

    dataset_cai = cai_wrapper(dataset)
    queries_cai = cai_wrapper(queries)

    if k is None:
        if indices is not None:
            k = cai_wrapper(indices).shape[1]
        elif distances is not None:
            k = cai_wrapper(distances).shape[1]
        else:
            raise ValueError("Argument k must be specified if both indices "
                             "and distances arg is None")

    n_queries = queries_cai.shape[0]

    if indices is None:
        indices = device_ndarray.empty((n_queries, k), dtype='int64')

    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype='float32')

    cdef DistanceType c_metric = DISTANCE_TYPES[metric]

    distances_cai = cai_wrapper(distances)
    indices_cai = cai_wrapper(indices)

    cdef optional[float] c_metric_arg = <float>metric_arg
    cdef optional[int64_t] c_global_offset = <int64_t>global_id_offset

    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    if dataset_cai.dtype == np.float32:
        with cuda_interruptible():
            c_knn(deref(handle_),
                  get_dmv_float(dataset_cai, check_shape=True),
                  get_dmv_float(queries_cai, check_shape=True),
                  get_dmv_int64(indices_cai, check_shape=True),
                  get_dmv_float(distances_cai, check_shape=True),
                  c_metric,
                  c_metric_arg,
                  c_global_offset)
    else:
        raise TypeError("dtype %s not supported" % dataset_cai.dtype)

    return (distances, indices)
