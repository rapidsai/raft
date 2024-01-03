#
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
from libc.stdint cimport int32_t, int64_t, uint32_t, uintptr_t
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
    get_dmv_float,
    get_dmv_int8,
    get_dmv_int64,
    get_dmv_uint8,
    make_optional_view_int64,
)
from pylibraft.neighbors.common cimport _get_metric_string
from pylibraft.neighbors.ivf_pq.cpp.c_ivf_pq cimport (
    index_params,
    search_params,
)


cdef _get_codebook_string(c_ivf_pq.codebook_gen codebook):
    return {c_ivf_pq.codebook_gen.PER_SUBSPACE: "subspace",
            c_ivf_pq.codebook_gen.PER_CLUSTER: "cluster"}[codebook]


cdef _map_dtype_np_to_cuda(dtype, supported_dtypes=None):
    if supported_dtypes is not None and dtype not in supported_dtypes:
        raise TypeError("Type %s is not supported" % str(dtype))
    return {np.float32: c_ivf_pq.cudaDataType_t.CUDA_R_32F,
            np.float16: c_ivf_pq.cudaDataType_t.CUDA_R_16F,
            np.uint8: c_ivf_pq.cudaDataType_t.CUDA_R_8U}[dtype]


cdef _get_dtype_string(dtype):
    return str({c_ivf_pq.cudaDataType_t.CUDA_R_32F: np.float32,
                c_ivf_pq.cudaDataType_t.CUDA_R_16F: np.float16,
                c_ivf_pq.cudaDataType_t.CUDA_R_8U: np.uint8}[dtype])


cdef class IndexParams:
    """
    Parameters to build index for IVF-PQ nearest neighbor search

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
    pq_bits : int, default = 8
        The bit length of the vector element after quantization.
    pq_dim : int, default = 0
        The dimensionality of a the vector after product quantization.
        When zero, an optimal value is selected using a heuristic. Note
        pq_dim * pq_bits must be a multiple of 8. Hint: a smaller 'pq_dim'
        results in a smaller index size and better search performance, but
        lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number,
        but multiple of 8 are desirable for good performance. If 'pq_bits'
        is not 8, 'pq_dim' should be a multiple of 8. For good performance,
        it is desirable that 'pq_dim' is a multiple of 32. Ideally,
        'pq_dim' should be also a divisor of the dataset dim.
    codebook_kind : string, default = "subspace"
        Valid values ["subspace", "cluster"]
    force_random_rotation : bool, default = False
        Apply a random rotation matrix on the input data and queries even
        if `dim % pq_dim == 0`. Note: if `dim` is not multiple of `pq_dim`,
        a random rotation is always applied to the input data and queries
        to transform the working space from `dim` to `rot_dim`, which may
        be slightly larger than the original space and and is a multiple
        of `pq_dim` (`rot_dim % pq_dim == 0`). However, this transform is
        not necessary when `dim` is multiple of `pq_dim` (`dim == rot_dim`,
        hence no need in adding "extra" data columns / features). By
        default, if `dim == rot_dim`, the rotation transform is
        initialized with the identity matrix. When
        `force_random_rotation == True`, a random orthogonal transform
        matrix is generated regardless of the values of `dim` and `pq_dim`.
    add_data_on_build : bool, default = True
        After training the coarse and fine quantizers, we will populate
        the index with the dataset if add_data_on_build == True, otherwise
        the index is left empty, and the extend method can be used
        to add new vectors to the index.
    conservative_memory_allocation : bool, default = True
        By default, the algorithm allocates more space than necessary for
        individual clusters (`list_data`). This allows to amortize the cost
        of memory allocation and reduce the number of data copies during
        repeated calls to `extend` (extending the database).
        To disable this behavior and use as little GPU memory for the
        database as possible, set this flat to `True`.
    random_seed : int, default = 0
        Seed used for random sampling if kmeans_trainset_fraction < 1.
        Value -1 disables random sampling, and results in sampling with a
        fixed stride.
    """
    def __init__(self, *,
                 n_lists=1024,
                 metric="sqeuclidean",
                 kmeans_n_iters=20,
                 kmeans_trainset_fraction=0.5,
                 pq_bits=8,
                 pq_dim=0,
                 codebook_kind="subspace",
                 force_random_rotation=False,
                 add_data_on_build=True,
                 conservative_memory_allocation=False,
                 random_seed=0):
        self.params.n_lists = n_lists
        self.params.metric = _get_metric(metric)
        self.params.metric_arg = 0
        self.params.kmeans_n_iters = kmeans_n_iters
        self.params.kmeans_trainset_fraction = kmeans_trainset_fraction
        self.params.pq_bits = pq_bits
        self.params.pq_dim = pq_dim
        if codebook_kind == "subspace":
            self.params.codebook_kind = c_ivf_pq.codebook_gen.PER_SUBSPACE
        elif codebook_kind == "cluster":
            self.params.codebook_kind = c_ivf_pq.codebook_gen.PER_CLUSTER
        else:
            raise ValueError("Incorrect codebook kind %s" % codebook_kind)
        self.params.force_random_rotation = force_random_rotation
        self.params.add_data_on_build = add_data_on_build
        self.params.conservative_memory_allocation = \
            conservative_memory_allocation
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
    def pq_bits(self):
        return self.params.pq_bits

    @property
    def pq_dim(self):
        return self.params.pq_dim

    @property
    def codebook_kind(self):
        return self.params.codebook_kind

    @property
    def force_random_rotation(self):
        return self.params.force_random_rotation

    @property
    def add_data_on_build(self):
        return self.params.add_data_on_build

    @property
    def conservative_memory_allocation(self):
        return self.params.conservative_memory_allocation

    @property
    def random_seed(self):
        return self.params.random_seed


cdef class Index:
    # We store a pointer to the index because it dose not have a trivial
    # constructor.
    cdef c_ivf_pq.index[int64_t] * index
    cdef readonly bool trained

    def __cinit__(self, handle=None):
        self.trained = False
        self.index = NULL
        if handle is None:
            handle = DeviceResources()
        cdef device_resources* handle_ = \
            <device_resources*><size_t>handle.getHandle()

        # We create a placeholder object. The actual parameter values do
        # not matter, it will be replaced with a built index object later.
        self.index = new c_ivf_pq.index[int64_t](
            deref(handle_), _get_metric("sqeuclidean"),
            c_ivf_pq.codebook_gen.PER_SUBSPACE,
            <uint32_t>1,
            <uint32_t>4,
            <uint32_t>8,
            <uint32_t>0,
            <bool>False)

    def __dealloc__(self):
        if self.index is not NULL:
            del self.index

    def __repr__(self):
        m_str = "metric=" + _get_metric_string(self.index.metric())
        code_str = "codebook=" + _get_codebook_string(
            self.index.codebook_kind())
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in ["size", "dim", "pq_dim", "pq_bits",
                                 "n_lists", "rot_dim"]]
        attr_str = [m_str, code_str] + attr_str
        return "Index(type=IVF-PQ, " + (", ".join(attr_str)) + ")"

    @property
    def dim(self):
        return self.index[0].dim()

    @property
    def size(self):
        return self.index[0].size()

    @property
    def pq_dim(self):
        return self.index[0].pq_dim()

    @property
    def pq_len(self):
        return self.index[0].pq_len()

    @property
    def pq_bits(self):
        return self.index[0].pq_bits()

    @property
    def metric(self):
        return self.index[0].metric()

    @property
    def n_lists(self):
        return self.index[0].n_lists()

    @property
    def rot_dim(self):
        return self.index[0].rot_dim()

    @property
    def codebook_kind(self):
        return self.index[0].codebook_kind()

    @property
    def conservative_memory_allocation(self):
        return self.index[0].conservative_memory_allocation()


@auto_sync_handle
@auto_convert_output
def build(IndexParams index_params, dataset, handle=None):
    """
    Builds an IVF-PQ index that can be later used for nearest neighbor search.

    The input array can be either CUDA array interface compliant matrix or
    array interface compliant matrix in host memory.

    Parameters
    ----------
    index_params : IndexParams object
    dataset : array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    {handle_docstring}

    Returns
    -------
    index: ivf_pq.Index

    Examples
    --------

    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import ivf_pq
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> handle = DeviceResources()
    >>> index_params = ivf_pq.IndexParams(
    ...     n_lists=1024,
    ...     metric="sqeuclidean",
    ...     pq_dim=10)
    >>> index = ivf_pq.build(index_params, dataset, handle=handle)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> distances, neighbors = ivf_pq.search(ivf_pq.SearchParams(), index,
    ...                                      queries, k, handle=handle)
    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)
    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
    """
    dataset_cai = wrap_array(dataset)
    dataset_dt = dataset_cai.dtype
    _check_input_array(dataset_cai, [np.dtype('float32'), np.dtype('byte'),
                                     np.dtype('ubyte')])

    cdef int64_t n_rows = dataset_cai.shape[0]
    cdef uint32_t dim = dataset_cai.shape[1]

    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    idx = Index()

    if dataset_dt == np.float32:
        with cuda_interruptible():
            c_ivf_pq.build(deref(handle_),
                           index_params.params,
                           get_dmv_float(dataset_cai, check_shape=True),
                           idx.index)
        idx.trained = True
    elif dataset_dt == np.byte:
        with cuda_interruptible():
            c_ivf_pq.build(deref(handle_),
                           index_params.params,
                           get_dmv_int8(dataset_cai, check_shape=True),
                           idx.index)
        idx.trained = True
    elif dataset_dt == np.ubyte:
        with cuda_interruptible():
            c_ivf_pq.build(deref(handle_),
                           index_params.params,
                           get_dmv_uint8(dataset_cai, check_shape=True),
                           idx.index)
        idx.trained = True
    else:
        raise TypeError("dtype %s not supported" % dataset_dt)

    return idx


@auto_sync_handle
@auto_convert_output
def extend(Index index, new_vectors, new_indices, handle=None):
    """
    Extend an existing index with new vectors.

    The input array can be either CUDA array interface compliant matrix or
    array interface compliant matrix in host memory.

    Parameters
    ----------
    index : ivf_pq.Index
        Trained ivf_pq object.
    new_vectors : array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    new_indices : array interface compliant vector shape (n_samples)
        Supported dtype [int64]
    {handle_docstring}

    Returns
    -------
    index: ivf_pq.Index

    Examples
    --------

    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import ivf_pq
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> handle = DeviceResources()
    >>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset, handle=handle)
    >>> n_rows = 100
    >>> more_data = cp.random.random_sample((n_rows, n_features),
    ...                                     dtype=cp.float32)
    >>> indices = index.size + cp.arange(n_rows, dtype=cp.int64)
    >>> index = ivf_pq.extend(index, more_data, indices)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> distances, neighbors = ivf_pq.search(ivf_pq.SearchParams(),
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

    vecs_cai = wrap_array(new_vectors)
    vecs_dt = vecs_cai.dtype
    cdef optional[device_vector_view[int64_t, int64_t]] new_indices_opt
    cdef int64_t n_rows = vecs_cai.shape[0]
    cdef uint32_t dim = vecs_cai.shape[1]

    _check_input_array(vecs_cai, [np.dtype('float32'), np.dtype('byte'),
                                  np.dtype('ubyte')],
                       exp_cols=index.dim)

    idx_cai = wrap_array(new_indices)
    _check_input_array(idx_cai, [np.dtype('int64')], exp_rows=n_rows)
    if len(idx_cai.shape)!=1:
        raise ValueError("Indices array is expected to be 1D")

    if index.index.size() > 0:
        new_indices_opt = make_device_vector_view(
            <int64_t *><uintptr_t>idx_cai.data,
            <int64_t>idx_cai.shape[0])

    if vecs_dt == np.float32:
        with cuda_interruptible():
            c_ivf_pq.extend(deref(handle_),
                            get_dmv_float(vecs_cai, check_shape=True),
                            new_indices_opt,
                            index.index)
    elif vecs_dt == np.int8:
        with cuda_interruptible():
            c_ivf_pq.extend(deref(handle_),
                            get_dmv_int8(vecs_cai, check_shape=True),
                            new_indices_opt,
                            index.index)
    elif vecs_dt == np.uint8:
        with cuda_interruptible():
            c_ivf_pq.extend(deref(handle_),
                            get_dmv_uint8(vecs_cai, check_shape=True),
                            new_indices_opt,
                            index.index)
    else:
        raise TypeError("query dtype %s not supported" % vecs_dt)

    return index


cdef class SearchParams:
    """
    IVF-PQ search parameters

    Parameters
    ----------
    n_probes: int, default = 1024
        The number of course clusters to select for the fine search.
    lut_dtype: default = np.float32
        Data type of look up table to be created dynamically at search
        time. The use of low-precision types reduces the amount of shared
        memory required at search time, so fast shared memory kernels can
        be used even for datasets with large dimansionality. Note that
        the recall is slightly degraded when low-precision type is
        selected. Possible values [np.float32, np.float16, np.uint8]
    internal_distance_dtype: default = np.float32
        Storage data type for distance/similarity computation.
        Possible values [np.float32, np.float16]
    """
    def __init__(self, *, n_probes=20,
                 lut_dtype=np.float32,
                 internal_distance_dtype=np.float32):
        self.params.n_probes = n_probes
        self.params.lut_dtype = _map_dtype_np_to_cuda(lut_dtype)
        self.params.internal_distance_dtype = \
            _map_dtype_np_to_cuda(internal_distance_dtype)
        # TODO(tfeher): enable if #926 adds this
        # self.params.shmem_carveout = self.shmem_carveout

    def __repr__(self):
        lut_str = "lut_dtype=" + _get_dtype_string(self.params.lut_dtype)
        idt_str = "internal_distance_dtype=" + \
            _get_dtype_string(self.params.internal_distance_dtype)
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in ["n_probes"]]
        # TODO (tfeher) add "shmem_carveout"
        attr_str = attr_str + [lut_str, idt_str]
        return "SearchParams(type=IVF-PQ, " + (", ".join(attr_str)) + ")"

    @property
    def n_probes(self):
        return self.params.n_probes

    @property
    def lut_dtype(self):
        return self.params.lut_dtype

    @property
    def internal_distance_dtype(self):
        return self.params.internal_distance_dtype


@auto_sync_handle
@auto_convert_output
def search(SearchParams search_params,
           Index index,
           queries,
           k,
           neighbors=None,
           distances=None,
           DeviceMemoryResource memory_resource=None,
           handle=None):
    """
    Find the k nearest neighbors for each query.

    Parameters
    ----------
    search_params : SearchParams
    index : Index
        Trained IVF-PQ index.
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
    memory_resource : RMM DeviceMemoryResource object, optional
        This can be used to explicitly manage the temporary memory
        allocation during search. Passing a pooling allocator can reduce
        memory allocation overhead. If not specified, then the memory
        resource from the raft handle is used.
    {handle_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import ivf_pq
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> handle = DeviceResources()
    >>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset, handle=handle)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> search_params = ivf_pq.SearchParams(
    ...     n_probes=20,
    ...     lut_dtype=cp.float16,
    ...     internal_distance_dtype=cp.float32
    ... )
    >>> # Using a pooling allocator reduces overhead of temporary array
    >>> # creation during search. This is useful if multiple searches
    >>> # are performad with same query size.
    >>> import rmm
    >>> mr = rmm.mr.PoolMemoryResource(
    ...     rmm.mr.CudaMemoryResource(),
    ...     initial_pool_size=2**29,
    ...     maximum_pool_size=2**31
    ... )
    >>> distances, neighbors = ivf_pq.search(search_params, index, queries,
    ...                                      k, memory_resource=mr,
    ...                                      handle=handle)
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
        neighbors = device_ndarray.empty((n_queries, k), dtype='int64')

    neighbors_cai = cai_wrapper(neighbors)
    _check_input_array(neighbors_cai, [np.dtype('int64')],
                       exp_rows=n_queries, exp_cols=k)

    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype='float32')

    distances_cai = cai_wrapper(distances)
    _check_input_array(distances_cai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    cdef c_ivf_pq.search_params params = search_params.params

    cdef uintptr_t neighbors_ptr = neighbors_cai.data
    cdef uintptr_t distances_ptr = distances_cai.data
    # TODO(tfeher) pass mr_ptr arg
    cdef device_memory_resource* mr_ptr = <device_memory_resource*> nullptr
    if memory_resource is not None:
        mr_ptr = memory_resource.get_mr()

    if queries_dt == np.float32:
        with cuda_interruptible():
            c_ivf_pq.search(deref(handle_),
                            params,
                            deref(index.index),
                            get_dmv_float(queries_cai, check_shape=True),
                            get_dmv_int64(neighbors_cai, check_shape=True),
                            get_dmv_float(distances_cai, check_shape=True))
    elif queries_dt == np.byte:
        with cuda_interruptible():
            c_ivf_pq.search(deref(handle_),
                            params,
                            deref(index.index),
                            get_dmv_int8(queries_cai, check_shape=True),
                            get_dmv_int64(neighbors_cai, check_shape=True),
                            get_dmv_float(distances_cai, check_shape=True))
    elif queries_dt == np.ubyte:
        with cuda_interruptible():
            c_ivf_pq.search(deref(handle_),
                            params,
                            deref(index.index),
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
        Trained IVF-PQ index.
    {handle_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import ivf_pq
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> handle = DeviceResources()
    >>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset, handle=handle)
    >>> ivf_pq.save("my_index.bin", index, handle=handle)
    """
    if not index.trained:
        raise ValueError("Index need to be built before saving it.")

    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    cdef string c_filename = filename.encode('utf-8')

    c_ivf_pq.serialize(deref(handle_), c_filename, deref(index.index))


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
    >>> from pylibraft.neighbors import ivf_pq
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build and save index
    >>> handle = DeviceResources()
    >>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset, handle=handle)
    >>> ivf_pq.save("my_index.bin", index, handle=handle)
    >>> del index
    >>> n_queries = 100
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> handle = DeviceResources()
    >>> index = ivf_pq.load("my_index.bin", handle=handle)
    >>> distances, neighbors = ivf_pq.search(ivf_pq.SearchParams(), index,
    ...                                      queries, k=10, handle=handle)
    """
    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    cdef string c_filename = filename.encode('utf-8')
    index = Index()

    c_ivf_pq.deserialize(deref(handle_), c_filename, index.index)
    index.trained = True

    return index
