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
import pylibraft.common.handle

from libc.stdint cimport uintptr_t
from libc.stdint cimport uint32_t, uint8_t, int8_t, int64_t, uint64_t
from cython.operator cimport dereference as deref
from libcpp cimport bool, nullptr

from pylibraft.distance.distance_type cimport DistanceType
from pylibraft.common import Handle
from pylibraft.common.handle cimport handle_t
from rmm._lib.memory_resource cimport device_memory_resource

cimport pylibraft.neighbors.c_ivf_pq as c_ivf_pq

from pylibraft.neighbors.c_ivf_pq cimport index_params
from pylibraft.neighbors.c_ivf_pq cimport search_params

def is_c_cont(cai):
    dt = np.dtype(cai["typestr"])
    return "strides" not in cai or \
        cai["strides"] is None or \
        cai["strides"][1] == dt.itemsize


def _get_metric(metric):
    SUPPORTED_DISTANCES = {
        "l2_expanded": DistanceType.L2Expanded,
        # TODO(tfeher): fix inconsistency: index building for L2SqrtExpanded is only supported by build, not by search.
        # "euclidean": DistanceType.L2SqrtExpanded
        "inner_product": DistanceType.InnerProduct
    }
    if metric not in SUPPORTED_DISTANCES:
        raise ValueError("metric %s is not supported" % metric)
    return SUPPORTED_DISTANCES[metric]


def _check_input_array(cai, exp_dt, exp_rows=None, exp_cols=None):
        if cai["typestr"] not in exp_dt:
            raise TypeError("dtype %s not supported" % cai["typestr"])

        if not is_c_cont(cai):
            raise ValueError("Row major input is expected")

        if exp_cols is not None and cai["shape"][1] != exp_cols:
            raise ValueError("Incorrect number of columns, expected {} got {}" \
                                 .format(exp_cols, cai["shape"][1]))

        if exp_rows is not None and cai["shape"][0] != exp_rows:
            raise ValueError("Incorrect number of rows, expected {} , got {}" \
                                .format(exp_rows, cai["shape"][0]))


# Variables to provide easier access for parameters
PER_SUBSPACE = c_ivf_pq.codebook_gen.PER_SUBSPACE
PER_CLUSTER = c_ivf_pq.codebook_gen.PER_CLUSTER

CUDA_R_32F = c_ivf_pq.cudaDataType_t.CUDA_R_32F
CUDA_R_16F = c_ivf_pq.cudaDataType_t.CUDA_R_16F
CUDA_R_8U = c_ivf_pq.cudaDataType_t.CUDA_R_8U


cdef class IndexParams:
    cdef c_ivf_pq.index_params params

    def __init__(self, *, 
                 n_lists=1024, 
                 metric="l2_expanded",
                 kmeans_n_iters=20, 
                 kmeans_trainset_fraction=0.5,
                 pq_bits=8,
                 pq_dim=0,
                 codebook_kind="per_subspace",
                 force_random_rotation=False,
                 add_data_on_build=True):
        """"
        Parameters to build index for IVF-PQ nearest neighbor search
    
        Parameters
        ----------
        n_list : int, default = 1024
            The number of clusters used in the coarse quantizer.
        metric : string denoting the metric type, default="l2_expanded"
            Valid values for metric: ["l2_expanded", "inner_product"], where
            - l2_expanded is the equclidean distance without the square root operation, 
              i.e.: distance(a,b) = \sum_i (a_i - b_i)^2,
            - inner product distance is defined as distance(a, b) = \sum_i a_i * b_i.
        kmeans_n_iters : int, default = 20
            The number of iterations searching for kmeans centers during index building. 
        kmeans_trainset_fraction : int, default = 0.5
            If kmeans_trainset_fraction is less than 1, then the dataset is subsampled,
            and only n_samples * kmeans_trainset_fraction rows are used for training.
        pq_bits : int, default = 8
            The bit length of the vector element after quantization.
        pq_dim : int, default = 0
            The dimensionality of a the vector after product quantization. When zero, an
            optimal value is selected using a heuristic. Note pq_dim * pq_bits must be a multiple of 8.
            Hint: a smaller 'pq_dim' results in a smaller index size and better search performance, but
            lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number, but multiple of 8 are
            desirable for good performance. If 'pq_bits' is not 8, 'pq_dim' should be a multiple of 8.
            For good performance, it is desirable that 'pq_dim' is a multiple of 32. Ideally, 'pq_dim'
            should be also a divisor of the dataset dim.
        codebook_kind : string, default = "per_subspace"
            Valid values ["per_subspace", "per_cluster"]
        force_random_rotation : bool, default = False
            Apply a random rotation matrix on the input data and queries even if `dim % pq_dim == 0`.
            Note: if `dim` is not multiple of `pq_dim`, a random rotation is always applied to the input
            data and queries to transform the working space from `dim` to `rot_dim`, which may be slightly
            larger than the original space and and is a multiple of `pq_dim` (`rot_dim % pq_dim == 0`).
            However, this transform is not necessary when `dim` is multiple of `pq_dim`
            (`dim == rot_dim`, hence no need in adding "extra" data columns / features).
            By default, if `dim == rot_dim`, the rotation transform is initialized with the identity
            matrix. When `force_random_rotation == True`, a random orthogonal transform matrix is generated
            regardless of the values of `dim` and `pq_dim`.
        add_data_on_build : bool, default = True
            After training the coarse and fine quantizers, we will populate the index with the dataset if
            add_data_on_build == True, otherwise the index is left empty, and the extend method can be used
            to add new vectors to the index.

        """
        self.params.n_lists = n_lists
        self.params.metric = _get_metric(metric)
        self.params.metric_arg = 0
        self.params.kmeans_n_iters = kmeans_n_iters
        self.params.kmeans_trainset_fraction = kmeans_trainset_fraction
        self.params.pq_bits = pq_bits
        self.params.pq_dim = pq_dim
        if codebook_kind == "per_subspace":
            self.params.codebook_kind = c_ivf_pq.codebook_gen.PER_SUBSPACE
        elif codebook_kind == "per_cluster":
            self.params.codebook_kind = c_ivf_pq.codebook_gen.PER_SUBSPACE
        else:
            raise ValueError("Incorrect codebook kind %s" % codebook_kind)
        self.params.force_random_rotation = force_random_rotation
        self.params.add_data_on_build = add_data_on_build

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


cdef class Index:
    # We store a pointer to the index because it dose not have a trivial constructor.
    cdef c_ivf_pq.index[uint64_t] * index
    cdef readonly bool trained

    def __cinit__(self, handle=None):
        self.trained = False
        self.index = NULL
        if handle is None:
            handle = Handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        # We create a placeholder object. The actual parameter values do not matter, it will be
        # replaced with a built index object later.
        self.index = new c_ivf_pq.index[uint64_t](deref(handle_), 
                                 _get_metric("l2_expanded"), 
                                 c_ivf_pq.codebook_gen.PER_SUBSPACE, 
                                 <uint32_t>1,
                                 <uint32_t>4, 
                                 <uint32_t>8,
                                 <uint32_t>0,
                                 <uint32_t>0)

    def __dealloc__(self):
        if self.index is not NULL:
            del self.index

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


def build(IndexParams index_params, dataset, handle=None):
    """
    Builds an IVF-PQ index that can be later used for nearest neighbor search.

    Parameters
    ----------
    index_params : IndexParams object
    dataset : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8] 

    Returns
    -------
    inde x: ivf_pq.Index

    Examples
    --------

    .. code-block:: python

        import cupy as cp

        from pylibraft.common import Handle
        from pylibraft.neighbors import ivf_pq

        n_samples = 5000
        n_features = 50
        n_queries = 1000

        dataset = cp.random.random_sample((n_samples, n_features),
                                              dtype=cp.float32)
        queries = cp.random.random_sample((n_samples, n_features),
                                              dtype=cp.float32)
        out_idx = cp.empty((n_samples, n_samples), dtype=cp.uint64)
        out_dist = cp.empty((n_samples, n_samples), dtype=cp.float32)

        handle = Handle()
        build_params = ivf_pq.IndexParams()
        index = build(index_params, dataset, handle)
        search_params = ivf_pq.SearchParams()
        [out_idx, out_dist] = ivf_pq.search(search_params, queries, handle)

        # pylibraft functions are often asynchronous so the
        # handle needs to be explicitly synchronized
        handle.sync()

    """
    dataset_cai = dataset.__cuda_array_interface__
    dataset_dt = np.dtype(dataset_cai["typestr"])
    _check_input_array(dataset_cai,  [np.dtype('float32'), np.dtype('byte'), np.dtype('ubyte')])
    cdef uintptr_t dataset_ptr = dataset_cai["data"][0]
        
    cdef uint64_t n_rows = dataset_cai["shape"][0]
    cdef uint32_t dim = dataset_cai["shape"][1]
    
    if handle is None:
        handle = Handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        
    idx = Index()
        
    if dataset_dt == np.float32:
        c_ivf_pq.build(deref(handle_),      
                       index_params.params,     
                       <float*> dataset_ptr,               
                       n_rows,
                       dim,
                       idx.index)
        idx.trained = True
    elif dataset_dt == np.byte:
        c_ivf_pq.build(deref(handle_),      
                       index_params.params,     
                       <int8_t*> dataset_ptr,               
                       n_rows,
                       dim,
                       idx.index)
        idx.trained = True
    elif dataset_dt == np.ubyte:
        c_ivf_pq.build(deref(handle_),      
                       index_params.params,     
                       <uint8_t*> dataset_ptr,               
                       n_rows,
                       dim,
                       idx.index) 
        idx.trained = True
    else:
        raise TypeError("dtype %s not supported" % dataset_dt)
 
    handle.sync() 
    return idx


def extend(Index index, new_vectors, new_indices, handle=None):
    """
    Extend an existing index with new vectors.
        
        
    Parameters
    ----------
    index : ivf_pq.Index
        Trained ivf_pq object.
    new_vectors : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8] 
    new_indices : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [uint64] 
    handle: raft Handle

    """
    if not index.trained:
        raise ValueError("Index need to be built before calling extend.")

    if handle is None:
        handle = Handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    vecs_cai = new_vectors.__cuda_array_interface__
    vecs_dt = np.dtype(vecs_cai["typestr"])
    cdef uint64_t n_rows = vecs_cai["shape"][0]
    cdef uint32_t dim = vecs_cai["shape"][1]

    _check_input_array(vecs_cai, [np.dtype('float32'), np.dtype('byte'), np.dtype('ubyte')], 
                       exp_cols=index.dim)

    idx_cai = new_indices.__cuda_array_interface__
    _check_input_array(idx_cai, [np.dtype('uint64')], exp_rows=n_rows)
    if len(idx_cai["shape"])!=1:
        raise ValueError("Indices array is expected to be 1D")


    cdef uintptr_t vecs_ptr = vecs_cai["data"][0]
    cdef uintptr_t idx_ptr = idx_cai["data"][0]

    if vecs_dt == np.float32:
        c_ivf_pq.extend(deref(handle_),
                        index.index,
                        <float*>vecs_ptr,
                        <uint64_t*> idx_ptr,
                        <uint64_t> n_rows)
    elif vecs_dt == np.int8:
        c_ivf_pq.extend(deref(handle_),
                        index.index,
                        <int8_t*>vecs_ptr,
                        <uint64_t*> idx_ptr,
                        <uint64_t> n_rows)
    elif vecs_dt == np.uint8:
        c_ivf_pq.extend(deref(handle_),
                        index.index,
                        <uint8_t*>vecs_ptr,
                        <uint64_t*> idx_ptr,
                        <uint64_t> n_rows)       
    else:
        raise TypeError("query dtype %s not supported" % vecs_dt)

    handle.sync() 
    return index 


cdef class SearchParams:
    cdef c_ivf_pq.search_params params

    def __init__(self, *, n_probes=20, 
                 lut_dtype=CUDA_R_32F, 
                 internal_distance_dtype=CUDA_R_32F):
        """
        IVF-PQ search parameters

        Parameters
        ----------
        n_probes: int, default = 1024
            The number of course clusters to select for the fine search.
        lut_dtype: default = ivf_pq.CUDA_R_32F (float)
            Data type of look up table to be created dynamically at search time. The use of 
            low-precision types reduces the amount of shared memory required at search time, so
            fast shared memory kernels can be used even for datasets with large dimansionality.
            Note that the recall is slightly degraded when low-precision type is selected.
            Possible values [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]
        internal_distance_dtype: default = ivf_q.CUDA_R_32F (float)
            Storage data type for distance/similarity computation.
            Possible values [CUDA_R_32F, CUDA_R_16F]
        
        """

        self.params.n_probes = n_probes
        self.params.lut_dtype = lut_dtype
        self.params.internal_distance_dtype = internal_distance_dtype
        # self.params.shmem_carveout = self.shmem_carveout # TODO(tfeher): enable if #926 adds this

    @property
    def n_probes(self):
        return self.params.n_probes

    @property
    def lut_dtype(self):
        return self.params.lut_dtype

    @property
    def internal_distance_dtype(self):
        return self.params.internal_distance_dtype       

def search(SearchParams search_params, 
           Index index, 
           queries,
           k,
           neighbors, 
           distances,
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
        neighbors : CUDA array interface compliant matrix shape (n_queries, k), dtype uint64_t
            If this parameter is specified, then the neighbor indices will be returned here. Otherwise a
            new array is created.
        distances : CUDA array interface compliant matrix shape (n_queries, k)
            If this parameter is specified, then the distances to the neighbors will be returned here.
            Otherwise a new array is created.
        mr_ptr : pointer to a raft device_memory_resource

        Returns
        -------
        A pair of (neighbors, distances) arrays as defined above.

        """

        if not index.trained:
            raise ValueError("Index need to be built before calling search.")
        
        if handle is None:
            handle = Handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        queries_cai = queries.__cuda_array_interface__
        queries_dt = np.dtype(queries_cai["typestr"])
        cdef uint32_t n_queries = queries_cai["shape"][0]

        _check_input_array(queries_cai, [np.dtype('float32'), np.dtype('byte'), np.dtype('ubyte')], 
                           exp_cols=index.dim)

        neighbors_cai = neighbors.__cuda_array_interface__
        _check_input_array(neighbors_cai, [np.dtype('uint64')], exp_rows=n_queries, exp_cols=k)

        distances_cai = distances.__cuda_array_interface__
        _check_input_array(distances_cai, [np.dtype('float32')], exp_rows=n_queries, exp_cols=k)

        cdef c_ivf_pq.search_params params = search_params.params

        cdef uintptr_t queries_ptr = queries_cai["data"][0]
        cdef uintptr_t neighbors_ptr = neighbors_cai["data"][0]
        cdef uintptr_t distances_ptr = distances_cai["data"][0]
        # TODO(tfeher) pass mr_ptr arg
        cdef device_memory_resource* mr_ptr = <device_memory_resource*> nullptr

        if queries_dt == np.float32:
            c_ivf_pq.search(deref(handle_),
                params,
                deref(index.index),
                <float*>queries_ptr,
                <uint32_t> n_queries,
                <uint32_t> k,
                <uint64_t*> neighbors_ptr,
                <float*> distances_ptr,
                mr_ptr)
        elif queries_dt == np.byte:
            c_ivf_pq.search(deref(handle_),
                params,
                deref(index.index),
                <int8_t*>queries_ptr,
                <uint32_t> n_queries,
                <uint32_t> k,
                <uint64_t*> neighbors_ptr,
                <float*> distances_ptr,
                mr_ptr)
        elif queries_dt == np.ubyte:
            c_ivf_pq.search(deref(handle_),
                params,
                deref(index.index),
                <uint8_t*>queries_ptr,
                <uint32_t> n_queries,
                <uint32_t> k,
                <uint64_t*> neighbors_ptr,
                <float*> distances_ptr,
                mr_ptr)
        else:
            raise ValueError("query dtype %s not supported" % queries_dt)

        handle.sync()      

