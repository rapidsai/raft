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

import warnings

import numpy as np

from cython.operator cimport dereference as deref
from libc.stdint cimport (
    int8_t,
    int32_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)
from libcpp cimport bool, nullptr
from libcpp.string cimport string

from pylibraft.distance.distance_type cimport DistanceType

from pylibraft.common import (
    DeviceResources,
    ai_wrapper,
    auto_convert_output,
    cai_wrapper,
    device_ndarray,
)
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from pylibraft.common.handle cimport device_resources

from pylibraft.common.handle import auto_sync_handle
from pylibraft.common.input_validation import is_c_contiguous

from rmm._lib.memory_resource cimport (
    DeviceMemoryResource,
    device_memory_resource,
)

cimport pylibraft.neighbors.cagra.cpp.c_cagra as c_cagra
from pylibraft.common.optional cimport make_optional, optional

from pylibraft.neighbors.common import _check_input_array, _get_metric

from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    device_vector_view,
    make_device_vector_view,
    row_major,
)
from pylibraft.common.mdspan cimport (
    get_const_dmv_float,
    get_const_dmv_int8,
    get_const_dmv_uint8,
    get_const_hmv_float,
    get_const_hmv_int8,
    get_const_hmv_uint8,
    get_dmv_float,
    get_dmv_int8,
    get_dmv_int64,
    get_dmv_uint8,
    get_dmv_uint32,
    get_hmv_float,
    get_hmv_int8,
    get_hmv_int64,
    get_hmv_uint8,
    get_hmv_uint32,
    make_optional_view_int64,
)
from pylibraft.neighbors.common cimport _get_metric_string


cdef class IndexParams:
    """"
    Parameters to build index for CAGRA nearest neighbor search

    Parameters
    ----------
    metric : string denoting the metric type, default="sqeuclidean"
        Valid values for metric: ["sqeuclidean"], where
            - sqeuclidean is the euclidean distance without the square root
              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2
    intermediate_graph_degree : int, default = 128

    graph_degree : int, default = 64

    build_algo: string denoting the graph building algorithm to use,
                default = "ivf_pq"
        Valid values for algo: ["ivf_pq", "nn_descent"], where
        - ivf_pq will use the IVF-PQ algorithm for building the knn graph
        - nn_descent (experimental) will use the NN-Descent algorithm for
          building the knn graph. It is expected to be generally
          faster than ivf_pq.
    """
    cdef c_cagra.index_params params

    def __init__(self, *,
                 metric="sqeuclidean",
                 intermediate_graph_degree=128,
                 graph_degree=64,
                 build_algo="ivf_pq"):
        self.params.metric = _get_metric(metric)
        self.params.metric_arg = 0
        self.params.intermediate_graph_degree = intermediate_graph_degree
        self.params.graph_degree = graph_degree
        if build_algo == "ivf_pq":
            self.params.build_algo = c_cagra.graph_build_algo.IVF_PQ
        elif build_algo == "nn_descent":
            self.params.build_algo = c_cagra.graph_build_algo.NN_DESCENT

    @property
    def metric(self):
        return self.params.metric

    @property
    def intermediate_graph_degree(self):
        return self.params.intermediate_graph_degree

    @property
    def graph_degree(self):
        return self.params.graph_degree


cdef class Index:
    cdef readonly bool trained
    cdef str active_index_type

    def __cinit__(self):
        self.trained = False
        self.active_index_type = None


cdef class IndexFloat(Index):
    cdef c_cagra.index[float, uint32_t] * index

    def __cinit__(self, handle=None):
        if handle is None:
            handle = DeviceResources()
        cdef device_resources* handle_ = \
            <device_resources*><size_t>handle.getHandle()

        self.index = new c_cagra.index[float, uint32_t](
            deref(handle_))

    def __repr__(self):
        m_str = "metric=" + _get_metric_string(self.index.metric())
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in ["metric", "dim", "graph_degree"]]
        attr_str = [m_str] + attr_str
        return "Index(type=CAGRA, " + (", ".join(attr_str)) + ")"

    @auto_sync_handle
    def update_dataset(self, dataset, handle=None):
        """ Replace the dataset with a new dataset.

        Parameters
        ----------
        dataset : array interface compliant matrix shape (n_samples, dim)
        {handle_docstring}
        """
        cdef device_resources* handle_ = \
            <device_resources*><size_t>handle.getHandle()

        dataset_ai = wrap_array(dataset)
        dataset_dt = dataset_ai.dtype
        _check_input_array(dataset_ai, [np.dtype("float32")])

        if dataset_ai.from_cai:
            self.index[0].update_dataset(deref(handle_),
                                         get_const_dmv_float(dataset_ai,
                                                             check_shape=True))
        else:
            self.index[0].update_dataset(deref(handle_),
                                         get_const_hmv_float(dataset_ai,
                                                             check_shape=True))

    @property
    def metric(self):
        return self.index[0].metric()

    @property
    def size(self):
        return self.index[0].size()

    @property
    def dim(self):
        return self.index[0].dim()

    @property
    def graph_degree(self):
        return self.index[0].graph_degree()

    def __dealloc__(self):
        if self.index is not NULL:
            del self.index


cdef class IndexInt8(Index):
    cdef c_cagra.index[int8_t, uint32_t] * index

    def __cinit__(self, handle=None):
        if handle is None:
            handle = DeviceResources()
        cdef device_resources* handle_ = \
            <device_resources*><size_t>handle.getHandle()

        self.index = new c_cagra.index[int8_t, uint32_t](
            deref(handle_))

    @auto_sync_handle
    def update_dataset(self, dataset, handle=None):
        """ Replace the dataset with a new dataset.

        Parameters
        ----------
        dataset : array interface compliant matrix shape (n_samples, dim)
        {handle_docstring}
        """
        cdef device_resources* handle_ = \
            <device_resources*><size_t>handle.getHandle()

        dataset_ai = wrap_array(dataset)
        dataset_dt = dataset_ai.dtype
        _check_input_array(dataset_ai, [np.dtype("byte")])

        if dataset_ai.from_cai:
            self.index[0].update_dataset(deref(handle_),
                                         get_const_dmv_int8(dataset_ai,
                                                            check_shape=True))
        else:
            self.index[0].update_dataset(deref(handle_),
                                         get_const_hmv_int8(dataset_ai,
                                                            check_shape=True))

    def __repr__(self):
        m_str = "metric=" + _get_metric_string(self.index.metric())
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in ["metric", "dim", "graph_degree"]]
        attr_str = [m_str] + attr_str
        return "Index(type=CAGRA, " + (", ".join(attr_str)) + ")"

    @property
    def metric(self):
        return self.index[0].metric()

    @property
    def size(self):
        return self.index[0].size()

    @property
    def dim(self):
        return self.index[0].dim()

    @property
    def graph_degree(self):
        return self.index[0].graph_degree()

    def __dealloc__(self):
        if self.index is not NULL:
            del self.index


cdef class IndexUint8(Index):
    cdef c_cagra.index[uint8_t, uint32_t] * index

    def __cinit__(self, handle=None):
        if handle is None:
            handle = DeviceResources()
        cdef device_resources* handle_ = \
            <device_resources*><size_t>handle.getHandle()

        self.index = new c_cagra.index[uint8_t, uint32_t](
            deref(handle_))

    @auto_sync_handle
    def update_dataset(self, dataset, handle=None):
        """ Replace the dataset with a new dataset.

        Parameters
        ----------
        dataset : array interface compliant matrix shape (n_samples, dim)
        {handle_docstring}
        """
        cdef device_resources* handle_ = \
            <device_resources*><size_t>handle.getHandle()

        dataset_ai = wrap_array(dataset)
        dataset_dt = dataset_ai.dtype
        _check_input_array(dataset_ai, [np.dtype("ubyte")])

        if dataset_ai.from_cai:
            self.index[0].update_dataset(deref(handle_),
                                         get_const_dmv_uint8(dataset_ai,
                                                             check_shape=True))
        else:
            self.index[0].update_dataset(deref(handle_),
                                         get_const_hmv_uint8(dataset_ai,
                                                             check_shape=True))

    def __repr__(self):
        m_str = "metric=" + _get_metric_string(self.index.metric())
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in ["metric", "dim", "graph_degree"]]
        attr_str = [m_str] + attr_str
        return "Index(type=CAGRA, " + (", ".join(attr_str)) + ")"

    @property
    def metric(self):
        return self.index[0].metric()

    @property
    def size(self):
        return self.index[0].size()

    @property
    def dim(self):
        return self.index[0].dim()

    @property
    def graph_degree(self):
        return self.index[0].graph_degree()

    def __dealloc__(self):
        if self.index is not NULL:
            del self.index


@auto_sync_handle
@auto_convert_output
def build(IndexParams index_params, dataset, handle=None):
    """
    Build the CAGRA index from the dataset for efficient search.

    The build performs two different steps- first an intermediate knn-graph is
    constructed, then it's optimized it to create the final graph. The
    index_params object controls the node degree of these graphs.

    It is required that both the dataset and the optimized graph fit the
    GPU memory.

    The following distance metrics are supported:
        - L2

    Parameters
    ----------
    index_params : IndexParams object
    dataset : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    {handle_docstring}

    Returns
    -------
    index: cagra.Index

    Examples
    --------

    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> handle = DeviceResources()
    >>> build_params = cagra.IndexParams(metric="sqeuclidean")
    >>> index = cagra.build(build_params, dataset, handle=handle)
    >>> distances, neighbors = cagra.search(cagra.SearchParams(),
    ...                                      index, dataset,
    ...                                      k, handle=handle)
    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)
    """
    dataset_ai = wrap_array(dataset)
    dataset_dt = dataset_ai.dtype
    _check_input_array(dataset_ai, [np.dtype('float32'), np.dtype('byte'),
                                    np.dtype('ubyte')])

    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    cdef IndexFloat idx_float
    cdef IndexInt8 idx_int8
    cdef IndexUint8 idx_uint8

    if dataset_ai.from_cai:
        if dataset_dt == np.float32:
            idx_float = IndexFloat(handle)
            idx_float.active_index_type = "float32"
            with cuda_interruptible():
                c_cagra.build_device(
                    deref(handle_),
                    index_params.params,
                    get_dmv_float(dataset_ai, check_shape=True),
                    deref(idx_float.index))
            idx_float.trained = True
            return idx_float
        elif dataset_dt == np.byte:
            idx_int8 = IndexInt8(handle)
            idx_int8.active_index_type = "byte"
            with cuda_interruptible():
                c_cagra.build_device(
                    deref(handle_),
                    index_params.params,
                    get_dmv_int8(dataset_ai, check_shape=True),
                    deref(idx_int8.index))
            idx_int8.trained = True
            return idx_int8
        elif dataset_dt == np.ubyte:
            idx_uint8 = IndexUint8(handle)
            idx_uint8.active_index_type = "ubyte"
            with cuda_interruptible():
                c_cagra.build_device(
                    deref(handle_),
                    index_params.params,
                    get_dmv_uint8(dataset_ai, check_shape=True),
                    deref(idx_uint8.index))
            idx_uint8.trained = True
            return idx_uint8
        else:
            raise TypeError("dtype %s not supported" % dataset_dt)
    else:
        if dataset_dt == np.float32:
            idx_float = IndexFloat(handle)
            idx_float.active_index_type = "float32"
            with cuda_interruptible():
                c_cagra.build_host(
                    deref(handle_),
                    index_params.params,
                    get_hmv_float(dataset_ai, check_shape=True),
                    deref(idx_float.index))
            idx_float.trained = True
            return idx_float
        elif dataset_dt == np.byte:
            idx_int8 = IndexInt8(handle)
            idx_int8.active_index_type = "byte"
            with cuda_interruptible():
                c_cagra.build_host(
                    deref(handle_),
                    index_params.params,
                    get_hmv_int8(dataset_ai, check_shape=True),
                    deref(idx_int8.index))
            idx_int8.trained = True
            return idx_int8
        elif dataset_dt == np.ubyte:
            idx_uint8 = IndexUint8(handle)
            idx_uint8.active_index_type = "ubyte"
            with cuda_interruptible():
                c_cagra.build_host(
                    deref(handle_),
                    index_params.params,
                    get_hmv_uint8(dataset_ai, check_shape=True),
                    deref(idx_uint8.index))
            idx_uint8.trained = True
            return idx_uint8
        else:
            raise TypeError("dtype %s not supported" % dataset_dt)


cdef class SearchParams:
    """
    CAGRA search parameters

    Parameters
    ----------
    max_queries: int, default = 0
        Maximum number of queries to search at the same time (batch size).
        Auto select when 0.
    itopk_size: int, default = 64
        Number of intermediate search results retained during the search.
        This is the main knob to adjust trade off between accuracy and
        search speed. Higher values improve the search accuracy.
    max_iterations: int, default = 0
        Upper limit of search iterations. Auto select when 0.
    algo: string denoting the search algorithm to use, default = "auto"
        Valid values for algo: ["auto", "single_cta", "multi_cta"], where
        - auto will automatically select the best value based on query size
        - single_cta is better when query contains larger number of
        vectors (e.g >10)
        - multi_cta is better when query contains only a few vectors
    team_size: int, default = 0
        Number of threads used to calculate a single distance. 4, 8, 16,
        or 32.
    search_width: int, default = 1
        Number of graph nodes to select as the starting point for the
        search in each iteration.
    min_iterations: int, default = 0
        Lower limit of search iterations.
    thread_block_size: int, default = 0
        Thread block size. 0, 64, 128, 256, 512, 1024.
        Auto selection when 0.
    hashmap_mode: string denoting the type of hash map to use. It's
        usually better to allow the algorithm to select this value.,
        default = "auto"
        Valid values for hashmap_mode: ["auto", "small", "hash"], where
        - auto will automatically select the best value based on algo
        - small will use the small shared memory hash table with resetting.
        - hash will use a single hash table in global memory.
    hashmap_min_bitlen: int, default = 0
        Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
    hashmap_max_fill_rate: float, default = 0.5
        Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
    num_random_samplings: int, default = 1
        Number of iterations of initial random seed node selection. 1 or
        more.
    rand_xor_mask: int, default = 0x128394
        Bit mask used for initial random seed node selection.
    """
    cdef c_cagra.search_params params

    def __init__(self, *,
                 max_queries=0,
                 itopk_size=64,
                 max_iterations=0,
                 algo="auto",
                 team_size=0,
                 search_width=1,
                 min_iterations=0,
                 thread_block_size=0,
                 hashmap_mode="auto",
                 hashmap_min_bitlen=0,
                 hashmap_max_fill_rate=0.5,
                 num_random_samplings=1,
                 rand_xor_mask=0x128394):
        self.params.max_queries = max_queries
        self.params.itopk_size = itopk_size
        self.params.max_iterations = max_iterations
        if algo == "single_cta":
            self.params.algo = c_cagra.search_algo.SINGLE_CTA
        elif algo == "multi_cta":
            self.params.algo = c_cagra.search_algo.MULTI_CTA
        elif algo == "multi_kernel":
            self.params.algo = c_cagra.search_algo.MULTI_KERNEL
        elif algo == "auto":
            self.params.algo = c_cagra.search_algo.AUTO
        else:
            raise ValueError("`algo` value not supported.")

        self.params.team_size = team_size
        self.params.search_width = search_width
        self.params.min_iterations = min_iterations
        self.params.thread_block_size = thread_block_size
        if hashmap_mode == "hash":
            self.params.hashmap_mode = c_cagra.hash_mode.HASH
        elif hashmap_mode == "small":
            self.params.hashmap_mode = c_cagra.hash_mode.SMALL
        elif hashmap_mode == "auto":
            self.params.hashmap_mode = c_cagra.hash_mode.AUTO
        else:
            raise ValueError("`hashmap_mode` value not supported.")

        self.params.hashmap_min_bitlen = hashmap_min_bitlen
        self.params.hashmap_max_fill_rate = hashmap_max_fill_rate
        self.params.num_random_samplings = num_random_samplings
        self.params.rand_xor_mask = rand_xor_mask

    def __repr__(self):
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in [
                        "max_queries", "itopk_size", "max_iterations", "algo",
                        "team_size", "search_width", "min_iterations",
                        "thread_block_size", "hashmap_mode",
                        "hashmap_min_bitlen", "hashmap_max_fill_rate",
                        "num_random_samplings", "rand_xor_mask"]]
        return "SearchParams(type=CAGRA, " + (", ".join(attr_str)) + ")"

    @property
    def max_queries(self):
        return self.params.max_queries

    @property
    def itopk_size(self):
        return self.params.itopk_size

    @property
    def max_iterations(self):
        return self.params.max_iterations

    @property
    def algo(self):
        return self.params.algo

    @property
    def team_size(self):
        return self.params.team_size

    @property
    def search_width(self):
        return self.params.search_width

    @property
    def min_iterations(self):
        return self.params.min_iterations

    @property
    def thread_block_size(self):
        return self.params.thread_block_size

    @property
    def hashmap_mode(self):
        return self.params.hashmap_mode

    @property
    def hashmap_min_bitlen(self):
        return self.params.hashmap_min_bitlen

    @property
    def hashmap_max_fill_rate(self):
        return self.params.hashmap_max_fill_rate

    @property
    def num_random_samplings(self):
        return self.params.num_random_samplings

    @property
    def rand_xor_mask(self):
        return self.params.rand_xor_mask


@auto_sync_handle
@auto_convert_output
def search(SearchParams search_params,
           Index index,
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
    index : Index
        Trained CAGRA index.
    queries : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    k : int
        The number of neighbors.
    neighbors : Optional CUDA array interface compliant matrix shape
                (n_queries, k), dtype int64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
    distances : Optional CUDA array interface compliant matrix shape
                (n_queries, k) If supplied, the distances to the
                neighbors will be written here in-place. (default None)
    {handle_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> handle = DeviceResources()
    >>> index = cagra.build(cagra.IndexParams(), dataset, handle=handle)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> search_params = cagra.SearchParams(
    ...     max_queries=100,
    ...     itopk_size=64
    ... )
    >>> # Using a pooling allocator reduces overhead of temporary array
    >>> # creation during search. This is useful if multiple searches
    >>> # are performad with same query size.
    >>> distances, neighbors = cagra.search(search_params, index, queries,
    ...                                     k, handle=handle)
    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
    >>> neighbors = cp.asarray(neighbors)
    >>> distances = cp.asarray(distances)
    """

    if not index.trained:
        raise ValueError("Index need to be built before calling search.")

    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    queries_cai = cai_wrapper(queries)
    queries_dt = queries_cai.dtype
    cdef uint32_t n_queries = queries_cai.shape[0]

    _check_input_array(queries_cai, [np.dtype('float32'), np.dtype('byte'),
                                     np.dtype('ubyte')],
                       exp_cols=index.dim)

    if neighbors is None:
        neighbors = device_ndarray.empty((n_queries, k), dtype='uint32')

    neighbors_cai = cai_wrapper(neighbors)
    _check_input_array(neighbors_cai, [np.dtype('uint32')],
                       exp_rows=n_queries, exp_cols=k)

    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype='float32')

    distances_cai = cai_wrapper(distances)
    _check_input_array(distances_cai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    cdef c_cagra.search_params params = search_params.params
    cdef IndexFloat idx_float
    cdef IndexInt8 idx_int8
    cdef IndexUint8 idx_uint8

    if queries_dt == np.float32:
        idx_float = index
        with cuda_interruptible():
            c_cagra.search(deref(handle_),
                           params,
                           deref(idx_float.index),
                           get_dmv_float(queries_cai, check_shape=True),
                           get_dmv_uint32(neighbors_cai, check_shape=True),
                           get_dmv_float(distances_cai, check_shape=True))
    elif queries_dt == np.byte:
        idx_int8 = index
        with cuda_interruptible():
            c_cagra.search(deref(handle_),
                           params,
                           deref(idx_int8.index),
                           get_dmv_int8(queries_cai, check_shape=True),
                           get_dmv_uint32(neighbors_cai, check_shape=True),
                           get_dmv_float(distances_cai, check_shape=True))
    elif queries_dt == np.ubyte:
        idx_uint8 = index
        with cuda_interruptible():
            c_cagra.search(deref(handle_),
                           params,
                           deref(idx_uint8.index),
                           get_dmv_uint8(queries_cai, check_shape=True),
                           get_dmv_uint32(neighbors_cai, check_shape=True),
                           get_dmv_float(distances_cai, check_shape=True))
    else:
        raise ValueError("query dtype %s not supported" % queries_dt)

    return (distances, neighbors)


@auto_sync_handle
def save(filename, Index index, bool include_dataset=True, handle=None):
    """
    Saves the index to a file.

    Saving / loading the index is experimental. The serialization format is
    subject to change.

    Parameters
    ----------
    filename : string
        Name of the file.
    index : Index
        Trained CAGRA index.
    include_dataset : bool
        Whether or not to write out the dataset along with the index. Including
        the dataset in the serialized index will use extra disk space, and
        might not be desired if you already have a copy of the dataset on
        disk. If this option is set to false, you will have to call
        `index.update_dataset(dataset)` after loading the index.
    {handle_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> handle = DeviceResources()
    >>> index = cagra.build(cagra.IndexParams(), dataset, handle=handle)
    >>> # Serialize and deserialize the cagra index built
    >>> cagra.save("my_index.bin", index, handle=handle)
    >>> index_loaded = cagra.load("my_index.bin", handle=handle)
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

    if index.active_index_type == "float32":
        idx_float = index
        c_cagra.serialize_file(
            deref(handle_), c_filename, deref(idx_float.index),
            include_dataset)
    elif index.active_index_type == "byte":
        idx_int8 = index
        c_cagra.serialize_file(
            deref(handle_), c_filename, deref(idx_int8.index), include_dataset)
    elif index.active_index_type == "ubyte":
        idx_uint8 = index
        c_cagra.serialize_file(
            deref(handle_), c_filename, deref(idx_uint8.index),
            include_dataset)
    else:
        raise ValueError(
            "Index dtype %s not supported" % index.active_index_type)


@auto_sync_handle
def load(filename, handle=None):
    """
    Loads index from file.

    Saving / loading the index is experimental. The serialization format is
    subject to change, therefore loading an index saved with a previous
    version of raft is not guaranteed to work.

    Parameters
    ----------
    filename : string
        Name of the file.
    {handle_docstring}

    Returns
    -------
    index : Index

    """
    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    cdef string c_filename = filename.encode('utf-8')
    cdef IndexFloat idx_float
    cdef IndexInt8 idx_int8
    cdef IndexUint8 idx_uint8

    with open(filename, "rb") as f:
        type_str = f.read(3).decode("utf8")
    dataset_dt = np.dtype(type_str)

    if dataset_dt == np.float32:
        idx_float = IndexFloat(handle)
        c_cagra.deserialize_file(
            deref(handle_), c_filename, idx_float.index)
        idx_float.trained = True
        idx_float.active_index_type = 'float32'
        return idx_float
    elif dataset_dt == np.byte:
        idx_int8 = IndexInt8(handle)
        c_cagra.deserialize_file(
            deref(handle_), c_filename, idx_int8.index)
        idx_int8.trained = True
        idx_int8.active_index_type = 'byte'
        return idx_int8
    elif dataset_dt == np.ubyte:
        idx_uint8 = IndexUint8(handle)
        c_cagra.deserialize_file(
            deref(handle_), c_filename, idx_uint8.index)
        idx_uint8.trained = True
        idx_uint8.active_index_type = 'ubyte'
        return idx_uint8
    else:
        raise ValueError("Dataset dtype %s not supported" % dataset_dt)
