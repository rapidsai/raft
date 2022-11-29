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

from cython.operator cimport dereference as deref
from libc.stdint cimport (
    int8_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)
from libcpp cimport bool, nullptr

from pylibraft.distance.distance_type cimport DistanceType

from pylibraft.common import Handle, cai_wrapper, device_ndarray
from pylibraft.common.handle cimport handle_t
from pylibraft.common.handle import auto_sync_handle
from pylibraft.common.input_validation import is_c_contiguous
from pylibraft.common.interruptible import cuda_interruptible
from pylibraft.distance.distance_type cimport DistanceType

import pylibraft.neighbors.ivf_pq  as ivf_pq

from pylibraft.neighbors.ivf_pq.ivf_pq  import _get_metric

cimport pylibraft.common.mdspan as mdspan

cimport pylibraft.neighbors.ivf_pq.c_ivf_pq as c_ivf_pq
from pylibraft.neighbors.ivf_pq.c_ivf_pq cimport index_params, search_params


# We will omit the const qualifiers in the interface for refine, because cython has
# an issue parsing it (https://github.com/cython/cython/issues/4180). 
cdef extern from "raft/neighbors/specializations/refine.hpp" \
        namespace "raft::neighbors" nogil:

    cdef void c_refine "raft::neighbors::refine" (const handle_t& handle,
        mdspan.device_matrix_view[float, uint64_t, mdspan.row_major] dataset,
        mdspan.device_matrix_view[float, uint64_t, mdspan.row_major] queries,
        mdspan.device_matrix_view[uint64_t, uint64_t, mdspan.row_major] candidates,
        mdspan.device_matrix_view[uint64_t, uint64_t, mdspan.row_major] indices,
        mdspan.device_matrix_view[float, uint64_t, mdspan.row_major] distances,
        DistanceType metric) except +
    
    cdef void c_refine "raft::neighbors::refine" (const handle_t& handle,
        mdspan.device_matrix_view[uint8_t, uint64_t, mdspan.row_major] dataset,
        mdspan.device_matrix_view[uint8_t, uint64_t, mdspan.row_major] queries,
        mdspan.device_matrix_view[uint64_t, uint64_t, mdspan.row_major] candidates,
        mdspan.device_matrix_view[uint64_t, uint64_t, mdspan.row_major] indices,
        mdspan.device_matrix_view[float, uint64_t, mdspan.row_major] distances,
        DistanceType metric) except +

    cdef void c_refine "raft::neighbors::refine" (const handle_t& handle,
        mdspan.device_matrix_view[int8_t, uint64_t, mdspan.row_major] dataset,
        mdspan.device_matrix_view[int8_t, uint64_t, mdspan.row_major] queries,
        mdspan.device_matrix_view[uint64_t, uint64_t, mdspan.row_major] candidates,
        mdspan.device_matrix_view[uint64_t, uint64_t, mdspan.row_major] indices,
        mdspan.device_matrix_view[float, uint64_t, mdspan.row_major] distances,
        DistanceType metric) except +


cdef mdspan.device_matrix_view[float, uint64_t, mdspan.row_major] get_device_matrix_view_float(array) except *:
    cai = cai_wrapper(array)
    if cai.dtype != np.float32:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got rank %d D" % len(cai.shape))
    return  mdspan.make_device_matrix_view[float, uint64_t, mdspan.row_major](<float*><uintptr_t>cai.data, cai.shape[0], cai.shape[1])
    

cdef mdspan.device_matrix_view[uint64_t, uint64_t, mdspan.row_major] get_device_matrix_view_uint64(array) except *:
    cai = cai_wrapper(array)
    if cai.dtype != np.uint64:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got rank %d D" % len(cai.shape))
    return  mdspan.make_device_matrix_view[uint64_t, uint64_t, mdspan.row_major](<uint64_t*><uintptr_t>cai.data, cai.shape[0], cai.shape[1])


cdef mdspan.device_matrix_view[uint8_t, uint64_t, mdspan.row_major] get_device_matrix_view_uint8(array) except *:
    cai = cai_wrapper(array)
    if cai.dtype != np.uint8:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got rank %d D" % len(cai.shape))
    return  mdspan.make_device_matrix_view[uint8_t, uint64_t, mdspan.row_major](<uint8_t*><uintptr_t>cai.data, cai.shape[0], cai.shape[1])
    

cdef mdspan.device_matrix_view[int8_t, uint64_t, mdspan.row_major] get_device_matrix_view_int8(array) except *:
    cai = cai_wrapper(array)
    if cai.dtype != np.int8:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got rank %d D" % len(cai.shape))
    return  mdspan.make_device_matrix_view[int8_t, uint64_t, mdspan.row_major](<int8_t*><uintptr_t>cai.data, cai.shape[0], cai.shape[1])


@auto_sync_handle
def refine(dataset, queries, candidates, k=None, indices=None, distances=None, metric = "l2_expanded", handle=None):
    """
    Refine nearest neighbor search.

    Refinement is an operation that follows an approximate NN search. The approximate search has
    already selected n_candidates neighbor candidates for each query. We narrow it down to k
    neighbors. For each query, we calculate the exact distance between the query and its
    n_candidates neighbor candidate, and select the k nearest ones.
    
    Input arrays can be either CUDA array interface compliant matrices or array interface 
    compliant matrices in host memory. All array must be in the same memory space. 

    Parameters
    ----------
    index_params : IndexParams object
    dataset : array interface compliant matrix, shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    queries : array interface compliant matrix, shape (n_queries, dim)
        Supported dtype [float, int8, uint8]
    candidates : array interface compliant matrix, shape (n_queries, k0)
        dtype [float, int8, uint64_t]    
    k : int 
        Number of neighbors to search (k <= k0). Optional if indices or 
        distances arrays are given (in which case their second dimension
        is k).
    indices :  Optional CUDA array interface compliant matrix shape
                (n_queries, k), dtype uint64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
        Supported dtype [float, int8, uint8]
    distances :  Optional CUDA array interface compliant matrix shape
                (n_queries, k), dtype float. If supplied, neighbor
                indices will be written here in-place. (default None)
       
    {handle_docstring}

    Returns
    -------
    index: ivf_pq.Index

    Examples
    --------

    .. code-block:: python

        import cupy as cp

        from pylibraft.common import Handle
        from pylibraft.neighbors import ivf_pq, refine

        n_samples = 50000
        n_features = 50
        n_queries = 1000

        dataset = cp.random.random_sample((n_samples, n_features),
            dtype=cp.float32)
        handle = Handle()
        index_params = ivf_pq.IndexParams(
            n_lists=1024,
            metric="l2_expanded",
            pq_dim=10)
        index = ivf_pq.build(index_params, dataset, handle=handle)

        # Search using the built index
        queries = cp.random.random_sample((n_queries, n_features),
                                          dtype=cp.float32)
        k = 40
        _, candidates = ivf_pq.search(ivf_pq.SearchParams(), index,
                                             queries, k, handle=handle)

        k = 10
        distances, neighbors = refine(dataset, queries, candidates, k, handle=handle)
        distances = cp.asarray(distances)
        neighbors = cp.asarray(neighbors)


        # pylibraft functions are often asynchronous so the
        # handle needs to be explicitly synchronized
        handle.sync()

    """
        
    if handle is None:
        handle = Handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    cdef mdspan.device_matrix_view[float, uint64_t, mdspan.row_major] dataset_view_fp32
    cdef mdspan.device_matrix_view[uint8_t, uint64_t, mdspan.row_major] dataset_view_uint8
    cdef mdspan.device_matrix_view[int8_t, uint64_t, mdspan.row_major] dataset_view_int8
    cdef mdspan.device_matrix_view[float, uint64_t, mdspan.row_major] queries_view_fp32
    cdef mdspan.device_matrix_view[uint8_t, uint64_t, mdspan.row_major] queries_view_uint8
    cdef mdspan.device_matrix_view[int8_t, uint64_t, mdspan.row_major] queries_view_int8
     
    cdef mdspan.device_matrix_view[uint64_t, uint64_t, mdspan.row_major] candidates_view = \
         get_device_matrix_view_uint64(candidates)
    
    if k is None:
        if indices is not None:
            k = cai_wrapper(indices).shape[1]
        elif distances is not None:
            k = cai_wrapper(distances).shape[1]
        else:
            raise ValueError("Argument k must be specified if both indices and distances arg is None")
        
    n_queries = cai_wrapper(queries).shape[0]
    
    if indices is None:
        indices = device_ndarray.empty((n_queries, k), dtype='uint64')

    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype='float32')
    
    cdef mdspan.device_matrix_view[uint64_t, uint64_t, mdspan.row_major] indices_view = \
        get_device_matrix_view_uint64(indices)

    cdef mdspan.device_matrix_view[float, uint64_t, mdspan.row_major] distances_view = \
         get_device_matrix_view_float(distances)

    cdef DistanceType c_metric = _get_metric(metric)
    
    dataset_cai = cai_wrapper(dataset)

    if dataset_cai.dtype == np.float32:
        with cuda_interruptible():
            dataset_view_fp32 = get_device_matrix_view_float(dataset)
            queries_view_fp32 = get_device_matrix_view_float(queries)
            c_refine(deref(handle_),
                    dataset_view_fp32,
                    queries_view_fp32,
                    candidates_view,
                    indices_view,
                    distances_view,
                    c_metric)
    elif dataset_cai.dtype == np.int8:
        with cuda_interruptible():
            dataset_view_int8 = get_device_matrix_view_int8(dataset)
            queries_view_int8 = get_device_matrix_view_int8(queries)
            c_refine(deref(handle_),
                    dataset_view_int8,
                    queries_view_int8,
                    candidates_view,
                    indices_view,
                    distances_view,
                    c_metric)   
    elif dataset_cai.dtype == np.uint8:
            dataset_view_uint8 = get_device_matrix_view_uint8(dataset)
            queries_view_uint8 = get_device_matrix_view_uint8(queries)
            c_refine(deref(handle_),
                    dataset_view_uint8,
                    queries_view_uint8,
                    candidates_view,
                    indices_view,
                    distances_view,
                    c_metric)    
    else:
       raise TypeError("dtype %s not supported" % dataset_cai.dtype)

    return (distances, indices)


