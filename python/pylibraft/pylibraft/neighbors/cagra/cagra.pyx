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
cimport pylibraft.neighbors.ivf_flat.cpp.c_ivf_flat as c_ivf_flat
cimport pylibraft.neighbors.ivf_pq.cpp.c_ivf_pq as c_ivf_pq
from pylibraft.common.optional cimport make_optional, optional

from pylibraft.neighbors.common import _check_input_array, _get_metric

from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    device_vector_view,
    make_device_vector_view,
    row_major,
)
from pylibraft.common.mdspan cimport (
    get_dmv_float_uint32,
    get_dmv_int8_uint32,
    get_dmv_int64_uint32,
    get_dmv_uint8_uint32,
    get_dmv_uint32_uint32,
    get_hmv_float_uint32,
    get_hmv_int8_uint32,
    get_hmv_int64_uint32,
    get_hmv_uint8_uint32,
    get_hmv_uint32_uint32,
    make_optional_view_int64,
)
from pylibraft.neighbors.common cimport _get_metric_string
from pylibraft.neighbors.ivf_pq cimport ivf_pq
from pylibraft.neighbors.ivf_pq.cpp.c_ivf_pq cimport (
    index_params as IVFPQ_IP,
    search_params as IVFPQ_SP,
)


cdef class IndexParams:
    cdef c_cagra.index_params params

    def __init__(self, *,
                 metric="sqeuclidean",
                 intermediate_graph_degree=128,
                 graph_degree=64,
                 add_data_on_build=True):
        """"
        Parameters to build index for IVF-PQ nearest neighbor search

        Parameters
        ----------
        metric : string denoting the metric type, default="sqeuclidean"
            Valid values for metric: ["sqeuclidean", "inner_product",
            "euclidean"], where
            - sqeuclidean is the euclidean distance without the square root
              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,
            - euclidean is the euclidean distance
            - inner product distance is defined as
              distance(a, b) = \\sum_i a_i * b_i.
        intermediate_graph_degree : int, default = 128

        graph_degree : int, default = 64

        add_data_on_build : bool, default = True
            After training the coarse and fine quantizers, we will populate
            the index with the dataset if add_data_on_build == True, otherwise
            the index is left empty, and the extend method can be used
            to add new vectors to the index.
        """
        self.params.metric = _get_metric(metric)
        self.params.metric_arg = 0
        self.params.intermediate_graph_degree = intermediate_graph_degree
        self.params.graph_degree = graph_degree
        self.params.add_data_on_build = add_data_on_build

    @property
    def metric(self):
        return self.params.metric

    @property
    def intermediate_graph_degree(self):
        return self.params.intermediate_graph_degree

    @property
    def graph_degree(self):
        return self.params.graph_degree

    @property
    def add_data_on_build(self):
        return self.params.add_data_on_build


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
        attr_str = m_str + attr_str
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


cdef class IndexInt8(Index):
    cdef c_cagra.index[int8_t, uint32_t] * index

    def __cinit__(self, handle=None):
        if handle is None:
            handle = DeviceResources()
        cdef device_resources* handle_ = \
            <device_resources*><size_t>handle.getHandle()

        self.index = new c_cagra.index[int8_t, uint32_t](
            deref(handle_))

    def __repr__(self):
        m_str = "metric=" + _get_metric_string(self.index.metric())
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in ["metric", "dim", "graph_degree"]]
        attr_str = m_str + attr_str
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

    def __repr__(self):
        m_str = "metric=" + _get_metric_string(self.index.metric())
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in ["metric", "dim", "graph_degree"]]
        attr_str = m_str + attr_str
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
def build_knn_graph(dataset,
                    IndexParams index_params,
                    knn_graph=None,
                    graph_degree=128,
                    ivf_pq.IndexParams build_params=None,
                    ivf_pq.SearchParams search_params=None,
                    refine_rate=0,
                    handle=None):
    """
    Build a kNN graph, the first building block for a CAGRA index.

    This function uses the IVF-PQ method to build a kNN graph. The output is
    a dense matrix that stores the neighbor indices for each point
    in the dataset. Each point has the same number of neighbors.

    Use pylibraft.cagra.build for an alternative, single function
    method for building a CAGRA index.

    The following distance metrics are supported:
    - L2Expanded

    Parameters
    ----------
    handle:
        RAFT resources
    dataset: array interface compliant matrix shape (n_samples, dim)
        a host or device matrix, Supported dtype [float, int8, uint8]/
    knn_graph: array interface compliant matrix shape (n_samples, dim)
        a host matrix view to store the output knn graph [n_rows, graph_degree]
    refine_rate: float
        refinement rate for ivf-pq search
    build_params:
         ivf_pq index building parameters for knn graph
    search_params:
        ivf_pq search parameters
    {handle_docstring}

    Returns
    -------
    knn_graph: array interface compliant matrix shape (n_samples, dim)

    Examples
    --------

    >>> import cupy as cp

    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import cagra

    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
    """
    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    dataset_ai = wrap_array(dataset)
    dataset_dt = dataset_ai.dtype
    _check_input_array(dataset_ai, [np.dtype('float32'), np.dtype('byte'),
                                    np.dtype('ubyte')])

    if knn_graph is None:
        knn_graph = device_ndarray.empty((dataset_ai.shape()[0],
                                          graph_degree),
                                         dtype='uint32')

    knn_graph_ai = wrap_array(knn_graph)
    knn_graph_dt = knn_graph_ai.dtype
    if knn_graph_ai.from_cai:
        raise ValueError(
            "Parameter `knn_graph` has to be a host (NumPy) matrix."
        )
    _check_input_array(knn_graph_ai, [np.dtype('uint32')])

    cdef optional[float] refine_rate_opt = <float>refine_rate
    cdef optional[IVFPQ_IP] build_params_opt = \
        <IVFPQ_IP> build_params.params
    cdef optional[IVFPQ_SP] search_params_opt = \
        <IVFPQ_SP> search_params.params

    if dataset_ai.from_cai:
        if dataset_ai.dtype == np.float32:
            with cuda_interruptible():
                c_cagra.build_knn_graph_device(
                    deref(handle_),
                    get_dmv_float_uint32(dataset_ai,
                                         check_shape=True),
                    get_hmv_uint32_uint32(knn_graph_ai,
                                          check_shape=True),
                    refine_rate_opt,
                    build_params_opt,
                    search_params_opt
                )

        elif dataset_ai.dtype == np.byte:
            with cuda_interruptible():
                c_cagra.build_knn_graph_device(
                    deref(handle_),
                    get_dmv_int8_uint32(dataset_ai,
                                        check_shape=True),
                    get_hmv_uint32_uint32(knn_graph_ai,
                                          check_shape=True),
                    refine_rate_opt,
                    build_params_opt,
                    search_params_opt
                )

        elif dataset_ai.dtype == np.ubyte:
            with cuda_interruptible():
                c_cagra.build_knn_graph_device(
                    deref(handle_),
                    get_dmv_uint8_uint32(dataset_ai,
                                         check_shape=True),
                    get_hmv_uint32_uint32(knn_graph_ai,
                                          check_shape=True),
                    refine_rate_opt,
                    build_params_opt,
                    search_params_opt
                )

    else:
        if dataset_ai.dtype == np.float32:
            with cuda_interruptible():
                c_cagra.build_knn_graph_host(
                    deref(handle_),
                    get_hmv_float_uint32(dataset_ai,
                                         check_shape=True),
                    get_hmv_uint32_uint32(knn_graph_ai,
                                          check_shape=True),
                    refine_rate_opt,
                    build_params_opt,
                    search_params_opt
                )

        elif dataset_ai.dtype == np.byte:
            with cuda_interruptible():
                c_cagra.build_knn_graph_host(
                    deref(handle_),
                    get_hmv_int8_uint32(dataset_ai,
                                        check_shape=True),
                    get_hmv_uint32_uint32(knn_graph_ai,
                                          check_shape=True),
                    refine_rate_opt,
                    build_params_opt,
                    search_params_opt
                )

        elif dataset_ai.dtype == np.ubyte:
            with cuda_interruptible():
                c_cagra.build_knn_graph_host(
                    deref(handle_),
                    get_hmv_uint8_uint32(dataset_ai,
                                         check_shape=True),
                    get_hmv_uint32_uint32(knn_graph_ai,
                                          check_shape=True),
                    refine_rate_opt,
                    build_params_opt,
                    search_params_opt
                )

        return knn_graph


@auto_sync_handle
def sort_knn_graph(dataset, knn_graph, handle=None):
    """
    Sort a KNN graph index.

    If a KNN graph is not built using
    pylibraft.cagra.build_knn_graph, then it is necessary to call
    this function before calling pylibraft.cagra.optimize. If the
    graph is built by pylibraft.cagra.build_knn_graph, it is
    already sorted and you do not need to call this function.

    Parameters
    ----------
    handle:
        RAFT resources
    dataset: array interface compliant matrix shape (n_samples, dim)
        a host or device matrix, Supported dtype [float, int8, uint8]/
    knn_graph: array interface compliant matrix shape (n_samples, dim)
        a host matrix view to store the output knn graph [n_rows, graph_degree]
    {handle_docstring}

    Examples
    --------

    >>> import cupy as cp

    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import cagra

    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
    """
    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    dataset_ai = wrap_array(dataset)
    dataset_dt = dataset_ai.dtype
    _check_input_array(dataset_ai, [np.dtype('float32'), np.dtype('byte'),
                                    np.dtype('ubyte')])

    knn_graph_ai = wrap_array(knn_graph)
    knn_graph_dt = knn_graph_ai.dtype
    if knn_graph_ai.from_cai:
        raise ValueError(
            "Parameter `knn_graph` has to be a host matrix."
        )
    _check_input_array(knn_graph_ai, [np.dtype('uint32')])

    if dataset_ai.from_cai:
        if dataset_ai.dtype == np.float32:
            with cuda_interruptible():
                c_cagra.sort_knn_graph_device(
                    deref(handle_),
                    get_dmv_float_uint32(dataset_ai,
                                         check_shape=True),
                    get_hmv_uint32_uint32(knn_graph_ai,
                                          check_shape=True)
                )
        elif dataset_ai.dtype == np.byte:
            with cuda_interruptible():
                c_cagra.sort_knn_graph_device(
                    deref(handle_),
                    get_dmv_int8_uint32(dataset_ai,
                                        check_shape=True),
                    get_hmv_uint32_uint32(knn_graph_ai,
                                          check_shape=True)
                )
        elif dataset_ai.dtype == np.ubyte:
            with cuda_interruptible():
                c_cagra.sort_knn_graph_device(
                    deref(handle_),
                    get_dmv_uint8_uint32(dataset_ai,
                                         check_shape=True),
                    get_hmv_uint32_uint32(knn_graph_ai,
                                          check_shape=True)
                )
    else:
        if dataset_ai.dtype == np.float32:
            with cuda_interruptible():
                c_cagra.sort_knn_graph_host(
                    deref(handle_),
                    get_hmv_float_uint32(dataset_ai,
                                         check_shape=True),
                    get_hmv_uint32_uint32(knn_graph_ai,
                                          check_shape=True)
                )
        elif dataset_ai.dtype == np.byte:
            with cuda_interruptible():
                c_cagra.sort_knn_graph_host(
                    deref(handle_),
                    get_hmv_int8_uint32(dataset_ai,
                                        check_shape=True),
                    get_hmv_uint32_uint32(knn_graph_ai,
                                          check_shape=True)
                )
        elif dataset_ai.dtype == np.ubyte:
            with cuda_interruptible():
                c_cagra.sort_knn_graph_host(
                    deref(handle_),
                    get_hmv_uint8_uint32(dataset_ai,
                                         check_shape=True),
                    get_hmv_uint32_uint32(knn_graph_ai,
                                          check_shape=True)
                )


@auto_sync_handle
def optimize(knn_graph, new_graph=None, handle=None):
    """
    Prune a KNN graph.

    Decreases the number of neighbors for each node.

    Parameters
    ----------
    handle:
        RAFT resources
    knn_graph: array interface compliant matrix shape (n_samples, dim)
        a host matrix containing the knn_graph of shape [n_rows, graph_degree]
    knn_graph: array interface compliant matrix shape (n_samples, dim)
        a host matrix view to store the output knn graph [n_rows, graph_degree]
    {handle_docstring}

    Examples
    --------

    >>> import cupy as cp

    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import cagra

    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
    """
    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    knn_graph_ai = wrap_array(knn_graph)
    knn_graph_dt = knn_graph_ai.dtype
    _check_input_array(knn_graph_ai, [np.dtype('uint32')])

    if new_graph is None:
        new_graph = device_ndarray.empty(knn_graph_ai.shape(), dtype='uint32')
    new_graph_ai = wrap_array(new_graph)
    new_graph_dt = new_graph_ai.dtype
    if new_graph_ai.from_cai:
        raise ValueError(
            "Parameter `new_graph` has to be a host (NumPy) matrix."
        )
    _check_input_array(new_graph_ai, [np.dtype('uint32')])

    if knn_graph_ai.from_cai:
        c_cagra.optimize_device(
            deref(handle_),
            get_dmv_uint32_uint32(knn_graph_ai, check_shape=True),
            get_hmv_uint32_uint32(new_graph_ai, check_shape=True))
    else:
        c_cagra.optimize_host(
            deref(handle_),
            get_hmv_uint32_uint32(knn_graph_ai, check_shape=True),
            get_hmv_uint32_uint32(new_graph_ai, check_shape=True))

    return new_graph


@auto_sync_handle
@auto_convert_output
def build(IndexParams index_params, dataset, handle=None):
    """
    Build the CAGRA index from the dataset for efficient search.

    The build consist of two steps: build an intermediate knn-graph, and
    optimize it to create the final graph. The index_params object controls
    the node degree of these graphs.

    It is required that dataset and the optimized graph fit the GPU memory.

    To customize the parameters for knn-graph building and pruning, and to
    reus the intermediate results, you could build the index in two steps
    using pylibraft.cagra.build_knn_graph and
    pylibraft.cagra.optimize.

     The following distance metrics are supported:
     - L2

    Parameters
    ----------
    index_params : IndexParams object
    dataset : array interface compliant matrix shape (n_samples, dim)
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

    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
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
                    get_dmv_float_uint32(dataset_ai,
                                         check_shape=True),
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
                    get_dmv_int8_uint32(dataset_ai,
                                        check_shape=True),
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
                    get_dmv_uint8_uint32(dataset_ai,
                                         check_shape=True),
                    deref(idx_uint8.index))
            idx_uint8.trained = True
            return idx_uint8
        else:
            if dataset_dt == np.float32:
                idx_float = IndexFloat(handle)
                idx_float.active_index_type = "float32"
                with cuda_interruptible():
                    c_cagra.build_host(
                        deref(handle_),
                        index_params.params,
                        get_hmv_float_uint32(dataset_ai,
                                             check_shape=True),
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
                        get_hmv_int8_uint32(dataset_ai,
                                            check_shape=True),
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
                        get_hmv_uint8_uint32(dataset_ai,
                                             check_shape=True),
                        deref(idx_uint8.index))
                idx_uint8.trained = True
                return idx_uint8
    else:
        raise TypeError("dtype %s not supported" % dataset_dt)


cdef class SearchParams:
    cdef c_cagra.search_params params

    def __init__(self, *,
                 max_queries=0,
                 itopk_size=64,
                 max_iterations=0,
                 algo="auto",
                 team_size=0,
                 num_parents=1,
                 min_iterations=0,
                 thread_block_size=0,
                 hashmap_mode="auto",
                 hashmap_min_bitlen=0,
                 hashmap_max_fill_rate=0.5,
                 num_random_samplings=1,
                 rand_xor_mask=0x128394):
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
            pper limit of search iterations. Auto select when 0.*/
        algo: search_algo, default = c_cagra.search_algo.AUTO
            Which search implementation to use.
        team_size: int, default = 0
            Number of threads used to calculate a single distance. 4, 8, 16,
            or 32.
        num_parents: int, default = 1
            Number of graph nodes to select as the starting point for the
            search in each iteration.
        min_iterations: int, default = 0
            Lower limit of search iterations.
        thread_block_size: int, default = 0
            Thread block size. 0, 64, 128, 256, 512, 1024.
            Auto selection when 0.
        hashmap_mode: hash_mode, default = c_cagra.hash_mode.AUTO
            Hashmap type. Auto selection when AUTO.
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
        self.params.num_parents = num_parents
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
        # todo(dantegd): add all relevant attrs
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in ["max_queries"]]
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
    def num_parents(self):
        return self.params.num_parents

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
                           get_dmv_float_uint32(queries_cai,
                                                check_shape=True),
                           get_dmv_uint32_uint32(neighbors_cai,
                                                 check_shape=True),
                           get_dmv_float_uint32(distances_cai,
                                                check_shape=True))
    elif queries_dt == np.byte:
        idx_int8 = index
        with cuda_interruptible():
            c_cagra.search(deref(handle_),
                           params,
                           deref(idx_int8.index),
                           get_dmv_int8_uint32(queries_cai,
                                               check_shape=True),
                           get_dmv_uint32_uint32(neighbors_cai,
                                                 check_shape=True),
                           get_dmv_float_uint32(distances_cai,
                                                check_shape=True))
    elif queries_dt == np.ubyte:
        idx_uint8 = index
        with cuda_interruptible():
            c_cagra.search(deref(handle_),
                           params,
                           deref(idx_uint8.index),
                           get_dmv_uint8_uint32(queries_cai,
                                                check_shape=True),
                           get_dmv_uint32_uint32(neighbors_cai,
                                                 check_shape=True),
                           get_dmv_float_uint32(distances_cai,
                                                check_shape=True))
    else:
        raise ValueError("query dtype %s not supported" % queries_dt)

    return (distances, neighbors)


@auto_sync_handle
def save(filename, Index index, handle=None):
    """
    Saves the index to file.

    Saving / loading the index is. The serialization format is
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

    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)

    >>> # Build index
    >>> handle = DeviceResources()
    >>> index = cagra.build(cagra.IndexParams(), dataset, handle=handle)
    >>> cagra.save("my_index.bin", index, handle=handle)
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
            deref(handle_), c_filename, deref(idx_float.index))
    elif index.active_index_type == "byte":
        idx_int8 = index
        c_cagra.serialize_file(
            deref(handle_), c_filename, deref(idx_int8.index))
    elif index.active_index_type == "ubyte":
        idx_uint8 = index
        c_cagra.serialize_file(
            deref(handle_), c_filename, deref(idx_uint8.index))
    else:
        raise ValueError(
            "Index dtype %s not supported" % index.active_index_type)


@auto_sync_handle
def load(filename, handle=None):
    """
    Loads index from file.

    Saving / loading the index is. The serialization format is
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

    Examples
    --------
    >>> import cupy as cp

    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import cagra

    """
    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    cdef string c_filename = filename.encode('utf-8')
    cdef IndexFloat idx_float
    cdef IndexInt8 idx_int8
    cdef IndexUint8 idx_uint8

    # we extract the dtype from the arrai interfaces in the file
    with open(filename, 'rb') as f:
        type_str = f.read(700).decode("utf-8", errors='ignore')

    dataset_dt = np.dtype(type_str[673:676])

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
