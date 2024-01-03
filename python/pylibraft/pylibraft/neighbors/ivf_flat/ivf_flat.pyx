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
from libc.stdint cimport int8_t, int64_t, uint8_t, uint32_t, uintptr_t
from libcpp cimport bool, nullptr
from libcpp.string cimport string

from pylibraft.distance.distance_type cimport DistanceType

from pylibraft.common import (
    DeviceResources,
    ai_wrapper,
    auto_convert_output,
    device_ndarray,
)
from pylibraft.common.cai_wrapper import cai_wrapper

from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    device_vector_view,
    make_device_vector_view,
    row_major,
)

from pylibraft.common.interruptible import cuda_interruptible

from pylibraft.common.handle cimport device_resources

from pylibraft.common.handle import auto_sync_handle
from pylibraft.common.input_validation import is_c_contiguous

from rmm._lib.memory_resource cimport (
    DeviceMemoryResource,
    device_memory_resource,
)

cimport pylibraft.neighbors.ivf_flat.cpp.c_ivf_flat as c_ivf_flat
from pylibraft.common.cpp.optional cimport optional

from pylibraft.neighbors.common import _check_input_array, _get_metric

from pylibraft.common.mdspan cimport (
    get_dmv_float,
    get_dmv_int8,
    get_dmv_int64,
    get_dmv_uint8,
)
from pylibraft.neighbors.common cimport _get_metric_string
from pylibraft.neighbors.ivf_flat.cpp.c_ivf_flat cimport (
    index_params,
    search_params,
)


cdef class IndexParams:
    """
    Parameters to build index for IVF-FLAT nearest neighbor search

    Parameters
    ----------
    n_list : int, default = 1024
        The number of clusters used in the coarse quantizer.
    metric : string denoting the metric type, default="sqeuclidean"
        Valid values for metric: ["sqeuclidean", "inner_product",
        "euclidean"], where
            - sqeuclidean is the euclidean distance without the square root
              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,
            - euclidean is the euclidean distance
            - inner product distance is defined as
              distance(a, b) = \\sum_i a_i * b_i.
    kmeans_n_iters : int, default = 20
        The number of iterations searching for kmeans centers during index
        building.
    kmeans_trainset_fraction : int, default = 0.5
        If kmeans_trainset_fraction is less than 1, then the dataset is
        subsampled, and only n_samples * kmeans_trainset_fraction rows
        are used for training.
    add_data_on_build : bool, default = True
        After training the coarse and fine quantizers, we will populate
        the index with the dataset if add_data_on_build == True, otherwise
        the index is left empty, and the extend method can be used
        to add new vectors to the index.
    adaptive_centers : bool, default = False
        By default (adaptive_centers = False), the cluster centers are
        trained in `ivf_flat::build`, and and never modified in
        `ivf_flat::extend`. The alternative behavior (adaptive_centers
        = true) is to update the cluster centers for new data when it is
        added. In this case, `index.centers()` are always exactly the
        centroids of the data in the corresponding clusters. The drawback
        of this behavior is that the centroids depend on the order of
        adding new data (through the classification of the added data);
        that is, `index.centers()` "drift" together with the changing
        distribution of the newly added data.
    random_seed : int, default = 0
        Seed used for random sampling if kmeans_trainset_fraction < 1.
        Value -1 disables random sampling, and results in sampling with a
        fixed stride.
   
    """
    cdef c_ivf_flat.index_params params

    def __init__(self, *,
                 n_lists=1024,
                 metric="sqeuclidean",
                 kmeans_n_iters=20,
                 kmeans_trainset_fraction=0.5,
                 add_data_on_build=True,
                 bool adaptive_centers=False,
                 random_seed=0):
        self.params.n_lists = n_lists
        self.params.metric = _get_metric(metric)
        self.params.metric_arg = 0
        self.params.kmeans_n_iters = kmeans_n_iters
        self.params.kmeans_trainset_fraction = kmeans_trainset_fraction
        self.params.add_data_on_build = add_data_on_build
        self.params.adaptive_centers = adaptive_centers
        self.params.random_seed = random_seed

    @property
    def n_lists(self):
        return self.params.n_lists

    @property
    def metric(self):
        return self.params.metric

    @property
    def kmeans_n_iters(self):
        return self.params.kmeans_n_iters

    @property
    def kmeans_trainset_fraction(self):
        return self.params.kmeans_trainset_fraction

    @property
    def add_data_on_build(self):
        return self.params.add_data_on_build

    @property
    def adaptive_centers(self):
        return self.params.adaptive_centers

    @property
    def random_seed(self):
        return self.params.random_seed


cdef class Index:
    cdef readonly bool trained
    cdef str active_index_type

    def __cinit__(self):
        self.trained = False
        self.active_index_type = None


cdef class IndexFloat(Index):
    cdef c_ivf_flat.index[float, int64_t] * index

    def __cinit__(self, handle=None):
        if handle is None:
            handle = DeviceResources()
        cdef device_resources* handle_ = \
            <device_resources*><size_t>handle.getHandle()

        # this is to keep track of which index type is being used
        # We create a placeholder object. The actual parameter values do
        # not matter, it will be replaced with a built index object later.
        self.index = new c_ivf_flat.index[float, int64_t](
            deref(handle_), _get_metric("sqeuclidean"),
            <uint32_t>1,
            <bool>False,
            <bool>False,
            <uint32_t>4)

    def __repr__(self):
        m_str = "metric=" + _get_metric_string(self.index.metric())
        attr_str = [
            attr + "=" + str(getattr(self, attr))
            for attr in ["size", "dim", "n_lists", "adaptive_centers"]
        ]
        attr_str = [m_str] + attr_str
        return "Index(type=IVF-FLAT, " + (", ".join(attr_str)) + ")"

    @property
    def dim(self):
        return self.index[0].dim()

    @property
    def size(self):
        return self.index[0].size()

    @property
    def metric(self):
        return self.index[0].metric()

    @property
    def n_lists(self):
        return self.index[0].n_lists()

    @property
    def adaptive_centers(self):
        return self.index[0].adaptive_centers()


cdef class IndexInt8(Index):
    cdef c_ivf_flat.index[int8_t, int64_t] * index

    def __cinit__(self, handle=None):
        if handle is None:
            handle = DeviceResources()
        cdef device_resources* handle_ = \
            <device_resources*><size_t>handle.getHandle()

        # this is to keep track of which index type is being used
        # We create a placeholder object. The actual parameter values do
        # not matter, it will be replaced with a built index object later.
        self.index = new c_ivf_flat.index[int8_t, int64_t](
            deref(handle_), _get_metric("sqeuclidean"),
            <uint32_t>1,
            <bool>False,
            <bool>False,
            <uint32_t>4)

    def __repr__(self):
        m_str = "metric=" + _get_metric_string(self.index.metric())
        attr_str = [
            attr + "=" + str(getattr(self, attr))
            for attr in ["size", "dim", "n_lists", "adaptive_centers"]
        ]
        attr_str = [m_str] + attr_str
        return "Index(type=IVF-FLAT, " + (", ".join(attr_str)) + ")"

    @property
    def dim(self):
        return self.index[0].dim()

    @property
    def size(self):
        return self.index[0].size()

    @property
    def metric(self):
        return self.index[0].metric()

    @property
    def n_lists(self):
        return self.index[0].n_lists()

    @property
    def adaptive_centers(self):
        return self.index[0].adaptive_centers()


cdef class IndexUint8(Index):
    cdef c_ivf_flat.index[uint8_t, int64_t] * index

    def __cinit__(self, handle=None):
        if handle is None:
            handle = DeviceResources()
        cdef device_resources* handle_ = \
            <device_resources*><size_t>handle.getHandle()

        # this is to keep track of which index type is being used
        # We create a placeholder object. The actual parameter values do
        # not matter, it will be replaced with a built index object later.
        self.index = new c_ivf_flat.index[uint8_t, int64_t](
            deref(handle_), _get_metric("sqeuclidean"),
            <uint32_t>1,
            <bool>False,
            <bool>False,
            <uint32_t>4)

    def __repr__(self):
        m_str = "metric=" + _get_metric_string(self.index.metric())
        attr_str = [
            attr + "=" + str(getattr(self, attr))
            for attr in ["size", "dim", "n_lists", "adaptive_centers"]
        ]
        attr_str = [m_str] + attr_str
        return "Index(type=IVF-FLAT, " + (", ".join(attr_str)) + ")"

    @property
    def dim(self):
        return self.index[0].dim()

    @property
    def size(self):
        return self.index[0].size()

    @property
    def metric(self):
        return self.index[0].metric()

    @property
    def n_lists(self):
        return self.index[0].n_lists()

    @property
    def adaptive_centers(self):
        return self.index[0].adaptive_centers()


@auto_sync_handle
@auto_convert_output
def build(IndexParams index_params, dataset, handle=None):
    """
    Builds an IVF-FLAT index that can be used for nearest neighbor search.

    Parameters
    ----------
    index_params : IndexParams object
    dataset : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    {handle_docstring}

    Returns
    -------
    index: ivf_flat.Index

    Examples
    --------

    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import ivf_flat
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> handle = DeviceResources()
    >>> index_params = ivf_flat.IndexParams(
    ...     n_lists=1024,
    ...     metric="sqeuclidean")
    >>> index = ivf_flat.build(index_params, dataset, handle=handle)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> distances, neighbors = ivf_flat.search(ivf_flat.SearchParams(),
    ...                                        index, queries, k,
    ...                                        handle=handle)
    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)
    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
    """
    dataset_cai = cai_wrapper(dataset)
    dataset_dt = dataset_cai.dtype
    _check_input_array(dataset_cai, [np.dtype('float32'), np.dtype('byte'),
                                     np.dtype('ubyte')])

    cdef int64_t n_rows = dataset_cai.shape[0]
    cdef uint32_t dim = dataset_cai.shape[1]

    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    cdef IndexFloat idx_float
    cdef IndexInt8 idx_int8
    cdef IndexUint8 idx_uint8

    if dataset_dt == np.float32:
        idx_float = IndexFloat(handle)
        idx_float.active_index_type = "float32"
        with cuda_interruptible():
            c_ivf_flat.build(deref(handle_),
                             index_params.params,
                             get_dmv_float(dataset_cai, check_shape=True),
                             deref(idx_float.index))
        idx_float.trained = True
        return idx_float
    elif dataset_dt == np.byte:
        idx_int8 = IndexInt8(handle)
        idx_int8.active_index_type = "byte"
        with cuda_interruptible():
            c_ivf_flat.build(deref(handle_),
                             index_params.params,
                             get_dmv_int8(dataset_cai, check_shape=True),
                             deref(idx_int8.index))
        idx_int8.trained = True
        return idx_int8
    elif dataset_dt == np.ubyte:
        idx_uint8 = IndexUint8(handle)
        idx_uint8.active_index_type = "ubyte"
        with cuda_interruptible():
            c_ivf_flat.build(deref(handle_),
                             index_params.params,
                             get_dmv_uint8(dataset_cai, check_shape=True),
                             deref(idx_uint8.index))
        idx_uint8.trained = True
        return idx_uint8
    else:
        raise TypeError("dtype %s not supported" % dataset_dt)


@auto_sync_handle
@auto_convert_output
def extend(Index index, new_vectors, new_indices, handle=None):
    """
    Extend an existing index with new vectors.

    Parameters
    ----------
    index : ivf_flat.Index
        Trained ivf_flat object.
    new_vectors : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    new_indices : CUDA array interface compliant vector shape (n_samples)
        Supported dtype [int64]
    {handle_docstring}

    Returns
    -------
    index: ivf_flat.Index

    Examples
    --------

    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import ivf_flat
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> handle = DeviceResources()
    >>> index = ivf_flat.build(ivf_flat.IndexParams(), dataset,
    ...                        handle=handle)
    >>> n_rows = 100
    >>> more_data = cp.random.random_sample((n_rows, n_features),
    ...                                     dtype=cp.float32)
    >>> indices = index.size + cp.arange(n_rows, dtype=cp.int64)
    >>> index = ivf_flat.extend(index, more_data, indices)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> distances, neighbors = ivf_flat.search(ivf_flat.SearchParams(),
    ...                                      index, queries,
    ...                                      k, handle=handle)
    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()

    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)
    """
    if not index.trained:
        raise ValueError("Index need to be built before calling extend.")

    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    vecs_cai = cai_wrapper(new_vectors)
    vecs_dt = vecs_cai.dtype
    cdef int64_t n_rows = vecs_cai.shape[0]
    cdef uint32_t dim = vecs_cai.shape[1]

    _check_input_array(vecs_cai, [np.dtype(index.active_index_type)],
                       exp_cols=index.dim)

    idx_cai = cai_wrapper(new_indices)
    _check_input_array(idx_cai, [np.dtype('int64')], exp_rows=n_rows)
    if len(idx_cai.shape)!=1:
        raise ValueError("Indices array is expected to be 1D")

    cdef optional[device_vector_view[int64_t, int64_t]] new_indices_opt

    cdef IndexFloat idx_float
    cdef IndexInt8 idx_int8
    cdef IndexUint8 idx_uint8

    if vecs_dt == np.float32:
        idx_float = index
        if idx_float.index.size() > 0:
            new_indices_opt = make_device_vector_view(
                <int64_t *><uintptr_t>idx_cai.data,
                <int64_t>idx_cai.shape[0])
        with cuda_interruptible():
            c_ivf_flat.extend(deref(handle_),
                              get_dmv_float(vecs_cai, check_shape=True),
                              new_indices_opt,
                              idx_float.index)
    elif vecs_dt == np.int8:
        idx_int8 = index
        if idx_int8.index[0].size() > 0:
            new_indices_opt = make_device_vector_view(
                <int64_t *><uintptr_t>idx_cai.data,
                <int64_t>idx_cai.shape[0])
        with cuda_interruptible():
            c_ivf_flat.extend(deref(handle_),
                              get_dmv_int8(vecs_cai, check_shape=True),
                              new_indices_opt,
                              idx_int8.index)
    elif vecs_dt == np.uint8:
        idx_uint8 = index
        if idx_uint8.index[0].size() > 0:
            new_indices_opt = make_device_vector_view(
                <int64_t *><uintptr_t>idx_cai.data,
                <int64_t>idx_cai.shape[0])
        with cuda_interruptible():
            c_ivf_flat.extend(deref(handle_),
                              get_dmv_uint8(vecs_cai, check_shape=True),
                              new_indices_opt,
                              idx_uint8.index)
    else:
        raise TypeError("query dtype %s not supported" % vecs_dt)

    return index


cdef class SearchParams:
    """
    IVF-FLAT search parameters

    Parameters
    ----------
    n_probes: int, default = 1024
        The number of course clusters to select for the fine search.
    """
    cdef c_ivf_flat.search_params params

    def __init__(self, *, n_probes=20):
        self.params.n_probes = n_probes

    def __repr__(self):
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in ["n_probes"]]
        return "SearchParams(type=IVF-FLAT, " + (", ".join(attr_str)) + ")"

    @property
    def n_probes(self):
        return self.params.n_probes


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
        Trained IVF-FLAT index.
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
    >>> from pylibraft.neighbors import ivf_flat
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> handle = DeviceResources()
    >>> index = ivf_flat.build(ivf_flat.IndexParams(), dataset,
    ...                        handle=handle)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> search_params = ivf_flat.SearchParams(
    ...     n_probes=20
    ... )
    >>> distances, neighbors = ivf_flat.search(search_params, index,
    ...                                        queries, k, handle=handle)
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

    _check_input_array(queries_cai, [np.dtype(index.active_index_type)],
                       exp_cols=index.dim)

    if neighbors is None:
        neighbors = device_ndarray.empty((n_queries, k), dtype='int64')

    neighbors_cai = cai_wrapper(neighbors)
    _check_input_array(neighbors_cai, [np.dtype('int64')],
                       exp_rows=n_queries, exp_cols=k)

    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype='float32')

    distances_cai = cai_wrapper(distances)
    _check_input_array(distances_cai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    cdef c_ivf_flat.search_params params = search_params.params
    cdef IndexFloat idx_float
    cdef IndexInt8 idx_int8
    cdef IndexUint8 idx_uint8

    if queries_dt == np.float32:
        idx_float = index
        with cuda_interruptible():
            c_ivf_flat.search(deref(handle_),
                              params,
                              deref(idx_float.index),
                              get_dmv_float(queries_cai, check_shape=True),
                              get_dmv_int64(neighbors_cai, check_shape=True),
                              get_dmv_float(distances_cai, check_shape=True))
    elif queries_dt == np.byte:
        idx_int8 = index
        with cuda_interruptible():
            c_ivf_flat.search(deref(handle_),
                              params,
                              deref(idx_int8.index),
                              get_dmv_int8(queries_cai, check_shape=True),
                              get_dmv_int64(neighbors_cai, check_shape=True),
                              get_dmv_float(distances_cai, check_shape=True))
    elif queries_dt == np.ubyte:
        idx_uint8 = index
        with cuda_interruptible():
            c_ivf_flat.search(deref(handle_),
                              params,
                              deref(idx_uint8.index),
                              get_dmv_uint8(queries_cai, check_shape=True),
                              get_dmv_int64(neighbors_cai, check_shape=True),
                              get_dmv_float(distances_cai, check_shape=True))
    else:
        raise ValueError("query dtype %s not supported" % queries_dt)

    return (distances, neighbors)


@auto_sync_handle
def save(filename, Index index, handle=None):
    """
    Saves the index to a file.

    Saving / loading the index is experimental. The serialization format is
    subject to change.

    Parameters
    ----------
    filename : string
        Name of the file.
    index : Index
        Trained IVF-Flat index.
    {handle_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import ivf_flat
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> handle = DeviceResources()
    >>> index = ivf_flat.build(ivf_flat.IndexParams(), dataset,
    ...                        handle=handle)
    >>> ivf_flat.save("my_index.bin", index, handle=handle)
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
        c_ivf_flat.serialize_file(
            deref(handle_), c_filename, deref(idx_float.index))
    elif index.active_index_type == "byte":
        idx_int8 = index
        c_ivf_flat.serialize_file(
            deref(handle_), c_filename, deref(idx_int8.index))
    elif index.active_index_type == "ubyte":
        idx_uint8 = index
        c_ivf_flat.serialize_file(
            deref(handle_), c_filename, deref(idx_uint8.index))
    else:
        raise ValueError(
            "Index dtype %s not supported" % index.active_index_type)


@auto_sync_handle
def load(filename, handle=None):
    """
    Loads index from a file.

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

    Examples
    --------
    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import ivf_flat
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build and save index
    >>> handle = DeviceResources()
    >>> index = ivf_flat.build(ivf_flat.IndexParams(), dataset,
    ...                        handle=handle)
    >>> ivf_flat.save("my_index.bin", index, handle=handle)
    >>> del index
    >>> n_queries = 100
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> handle = DeviceResources()
    >>> index = ivf_flat.load("my_index.bin", handle=handle)
    >>> distances, neighbors = ivf_flat.search(ivf_flat.SearchParams(),
    ...                                        index, queries, k=10,
    ...                                        handle=handle)
    """
    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    cdef string c_filename = filename.encode('utf-8')
    cdef IndexFloat idx_float
    cdef IndexInt8 idx_int8
    cdef IndexUint8 idx_uint8

    with open(filename, 'rb') as f:
        type_str = f.read(3).decode('utf-8')

    dataset_dt = np.dtype(type_str)

    if dataset_dt == np.float32:
        idx_float = IndexFloat(handle)
        c_ivf_flat.deserialize_file(
            deref(handle_), c_filename, idx_float.index)
        idx_float.trained = True
        idx_float.active_index_type = 'float32'
        return idx_float
    elif dataset_dt == np.byte:
        idx_int8 = IndexInt8(handle)
        c_ivf_flat.deserialize_file(
            deref(handle_), c_filename, idx_int8.index)
        idx_int8.trained = True
        idx_int8.active_index_type = 'byte'
        return idx_int8
    elif dataset_dt == np.ubyte:
        idx_uint8 = IndexUint8(handle)
        c_ivf_flat.deserialize_file(
            deref(handle_), c_filename, idx_uint8.index)
        idx_uint8.trained = True
        idx_uint8.active_index_type = 'ubyte'
        return idx_uint8
    else:
        raise ValueError("Index dtype %s not supported" % dataset_dt)
