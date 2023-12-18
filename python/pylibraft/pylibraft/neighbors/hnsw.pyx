#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

from cython.operator cimport dereference as deref
from libc.stdint cimport int8_t, uint8_t, uint32_t
from libcpp cimport bool
from libcpp.string cimport string

cimport pylibraft.neighbors.cagra.cpp.c_cagra as c_cagra
from pylibraft.distance.distance_type cimport DistanceType
from pylibraft.neighbors.cagra.cagra cimport (
    Index,
    IndexFloat,
    IndexInt8,
    IndexUint8,
)

from pylibraft.common.handle import auto_sync_handle

from pylibraft.common.handle cimport device_resources

from pylibraft.common import DeviceResources, ai_wrapper, auto_convert_output

cimport pylibraft.neighbors.cpp.hnsw as c_hnsw

from pylibraft.neighbors.common import _check_input_array, _get_metric

from pylibraft.common.mdspan cimport (
    get_hmv_float,
    get_hmv_int8,
    get_hmv_uint8,
    get_hmv_uint64,
)
from pylibraft.neighbors.common cimport _get_metric_string

import numpy as np


cdef class HnswIndex:
    cdef readonly bool trained
    cdef str active_index_type

    def __cinit__(self):
        self.trained = False
        self.active_index_type = None

cdef class HnswIndexFloat(HnswIndex):
    cdef c_hnsw.index[float] * index

    def __cinit__(self):
        pass

    def __repr__(self):
        m_str = "metric=" + _get_metric_string(self.index.metric())
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in ["dim"]]
        attr_str = [m_str] + attr_str
        return "Index(type=hnsw, " + (", ".join(attr_str)) + ")"

    @property
    def dim(self):
        return self.index[0].dim()

    @property
    def metric(self):
        return self.index[0].metric()

    def __dealloc__(self):
        if self.index is not NULL:
            del self.index

cdef class HnswIndexInt8(HnswIndex):
    cdef c_hnsw.index[int8_t] * index

    def __cinit__(self):
        pass

    def __repr__(self):
        m_str = "metric=" + _get_metric_string(self.index.metric())
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in ["dim"]]
        attr_str = [m_str] + attr_str
        return "Index(type=hnsw, " + (", ".join(attr_str)) + ")"

    @property
    def dim(self):
        return self.index[0].dim()

    @property
    def metric(self):
        return self.index[0].metric()

    def __dealloc__(self):
        if self.index is not NULL:
            del self.index

cdef class HnswIndexUint8(HnswIndex):
    cdef c_hnsw.index[uint8_t] * index

    def __cinit__(self):
        pass

    def __repr__(self):
        m_str = "metric=" + _get_metric_string(self.index.metric())
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in ["dim"]]
        attr_str = [m_str] + attr_str
        return "Index(type=hnsw, " + (", ".join(attr_str)) + ")"

    @property
    def dim(self):
        return self.index[0].dim()

    @property
    def metric(self):
        return self.index[0].metric()

    def __dealloc__(self):
        if self.index is not NULL:
            del self.index


@auto_sync_handle
def save(filename, Index index, handle=None):
    """
    Saves the CAGRA index as an hnswlib base layer only index to a file.

    Saving / loading the index is experimental. The serialization format is
    subject to change.

    Parameters
    ----------
    filename : string
        Name of the file.
    index : Index
        Trained CAGRA index.
    {handle_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import cagra
    >>> from pylibraft.neighbors import hnsw
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> handle = DeviceResources()
    >>> index = cagra.build(cagra.IndexParams(), dataset, handle=handle)
    >>> # Serialize the CAGRA index to hnswlib base layer only index format
    >>> hnsw.save("my_index.bin", index, handle=handle)
    """
    if not index.trained:
        raise ValueError("Index need to be built before saving it.")

    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    cdef string c_filename = filename.encode('utf-8')

    cdef IndexFloat idx_float
    cdef IndexInt8 idx_int8
    cdef IndexUint8 idx_uint8

    cdef c_cagra.index[float, uint32_t] * c_index_float
    cdef c_cagra.index[int8_t, uint32_t] * c_index_int8
    cdef c_cagra.index[uint8_t, uint32_t] * c_index_uint8

    if index.active_index_type == "float32":
        idx_float = index
        c_index_float = \
            <c_cagra.index[float, uint32_t] *><size_t> idx_float.index
        c_cagra.serialize_to_hnswlib_file(
            deref(handle_), c_filename, deref(c_index_float))
    elif index.active_index_type == "byte":
        idx_int8 = index
        c_index_int8 = \
            <c_cagra.index[int8_t, uint32_t] *><size_t> idx_int8.index
        c_cagra.serialize_to_hnswlib_file(
            deref(handle_), c_filename, deref(c_index_int8))
    elif index.active_index_type == "ubyte":
        idx_uint8 = index
        c_index_uint8 = \
            <c_cagra.index[uint8_t, uint32_t] *><size_t> idx_uint8.index
        c_cagra.serialize_to_hnswlib_file(
            deref(handle_), c_filename, deref(c_index_uint8))
    else:
        raise ValueError(
            "Index dtype %s not supported" % index.active_index_type)


def load(filename, dim, dtype, metric="sqeuclidean", handle=None):
    """
    Loads base layer only hnswlib index from file, which was originally
    saved as a built CAGRA index.

    Saving / loading the index is experimental. The serialization format is
    subject to change, therefore loading an index saved with a previous
    version of raft is not guaranteed to work.

    Parameters
    ----------
    filename : string
        Name of the file.
    dim : int
        Dimensions of the training dataest
    dtype : np.dtype of the saved index
        Valid values for dtype: ["float", "byte", "ubyte"]
    metric : string denoting the metric type, default="sqeuclidean"
        Valid values for metric: ["sqeuclidean", "inner_product"], where
            - sqeuclidean is the euclidean distance without the square root
              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,
            - inner product distance is defined as
              distance(a, b) = \\sum_i a_i * b_i.
    {handle_docstring}

    Returns
    -------
    index : HnswIndex

    Examples
    --------
    >>> from pylibraft.neighbors import hnsw
    >>> dim = 50 # Assuming training dataset has 50 dimensions
    >>> index = hnsw.load("my_index.bin", dim, "sqeuclidean")
    """
    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    cdef string c_filename = filename.encode('utf-8')
    cdef HnswIndexFloat idx_float
    cdef HnswIndexInt8 idx_int8
    cdef HnswIndexUint8 idx_uint8

    cdef DistanceType c_metric = _get_metric(metric)

    if dtype == np.float32:
        idx_float = HnswIndexFloat()
        c_hnsw.deserialize_file(
            deref(handle_), c_filename, idx_float.index, <int> dim, c_metric)
        idx_float.trained = True
        idx_float.active_index_type = 'float32'
        return idx_float
    elif dtype == np.byte:
        idx_int8 = HnswIndexInt8(dim, metric)
        c_hnsw.deserialize_file(
            deref(handle_), c_filename, idx_int8.index, <int> dim, c_metric)
        idx_int8.trained = True
        idx_int8.active_index_type = 'byte'
        return idx_int8
    elif dtype == np.ubyte:
        idx_uint8 = HnswIndexUint8(dim, metric)
        c_hnsw.deserialize_file(
            deref(handle_), c_filename, idx_uint8.index, <int> dim, c_metric)
        idx_uint8.trained = True
        idx_uint8.active_index_type = 'ubyte'
        return idx_uint8
    else:
        raise ValueError("Dataset dtype %s not supported" % dtype)


cdef class SearchParams:
    """
    Hnswlib search parameters

    Parameters
    ----------
    ef: int, default=200
        Size of list from which final neighbors k will be selected.
        ef should be greater than or equal to k.
    num_threads: int, default=1
        Number of host threads to use to search the hnswlib index
        and increase concurrency
    """
    cdef c_hnsw.search_params params

    def __init__(self, ef=200, num_threads=1):
        self.params.ef = ef
        self.params.num_threads = num_threads

    def __repr__(self):
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in [
                        "ef", "num_threads"]]
        return "SearchParams(type=hnsw, " + (
            ", ".join(attr_str)) + ")"

    @property
    def ef(self):
        return self.params.ef

    @property
    def num_threads(self):
        return self.params.num_threads


@auto_sync_handle
@auto_convert_output
def search(SearchParams search_params,
           HnswIndex index,
           queries,
           k,
           neighbors=None,
           distances=None,
           handle=None):
    """
    Find the k nearest neighbors for each query.

    Parameters
    ----------
    search_params : SearchParams
    index : HnswIndex
        Trained CAGRA index saved as base layer only hnswlib index.
    queries : array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    k : int
        The number of neighbors.
    neighbors : Optional array interface compliant matrix shape
                (n_queries, k), dtype int64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
    distances : Optional array interface compliant matrix shape
                (n_queries, k) If supplied, the distances to the
                neighbors will be written here in-place. (default None)
    {handle_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> import numpy as np
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import cagra
    >>> from pylibraft.neighbors import hnsw
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> handle = DeviceResources()
    >>> index = cagra.build(cagra.IndexParams(), dataset, handle=handle)
    >>>
    >>> Save CAGRA built index as base layer only hnswlib index
    >>> hnsw.save("my_index.bin", index)
    >>>
    >>> Load saved base layer only hnswlib index
    >>> hnsw_index = hnsw.load("my_index.bin", n_features, dataset.dtype)
    >>>
    >>> # Search hnswlib using the loaded index
    >>> queries = np.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> search_params = hnsw.SearchParams(
    ...     ef=20,
    ...     num_threads=5
    ... )
    >>> distances, neighbors = hnsw.search(search_params, hnsw_index,
    ...                                    queries, k, handle=handle)
    """

    if not index.trained:
        raise ValueError("Index need to be built before calling search.")

    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    queries_ai = ai_wrapper(queries)
    queries_dt = queries_ai.dtype
    cdef uint32_t n_queries = queries_ai.shape[0]

    _check_input_array(queries_ai, [np.dtype('float32'), np.dtype('byte'),
                                    np.dtype('ubyte')],
                       exp_cols=index.dim)

    if neighbors is None:
        neighbors = np.empty((n_queries, k), dtype='uint64')

    neighbors_ai = ai_wrapper(neighbors)
    _check_input_array(neighbors_ai, [np.dtype('uint64')],
                       exp_rows=n_queries, exp_cols=k)

    if distances is None:
        distances = np.empty((n_queries, k), dtype='float32')

    distances_ai = ai_wrapper(distances)
    _check_input_array(distances_ai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    cdef c_hnsw.search_params params = search_params.params
    cdef HnswIndexFloat idx_float
    cdef HnswIndexInt8 idx_int8
    cdef HnswIndexUint8 idx_uint8

    if queries_dt == np.float32:
        idx_float = index
        c_hnsw.search(deref(handle_),
                      params,
                      deref(idx_float.index),
                      get_hmv_float(queries_ai, check_shape=True),
                      get_hmv_uint64(neighbors_ai, check_shape=True),
                      get_hmv_float(distances_ai, check_shape=True))
    elif queries_dt == np.byte:
        idx_int8 = index
        c_hnsw.search(deref(handle_),
                      params,
                      deref(idx_int8.index),
                      get_hmv_int8(queries_ai, check_shape=True),
                      get_hmv_uint64(neighbors_ai, check_shape=True),
                      get_hmv_float(distances_ai, check_shape=True))
    elif queries_dt == np.ubyte:
        idx_uint8 = index
        c_hnsw.search(deref(handle_),
                      params,
                      deref(idx_uint8.index),
                      get_hmv_uint8(queries_ai, check_shape=True),
                      get_hmv_uint64(neighbors_ai, check_shape=True),
                      get_hmv_float(distances_ai, check_shape=True))
    else:
        raise ValueError("query dtype %s not supported" % queries_dt)

    return (distances, neighbors)
