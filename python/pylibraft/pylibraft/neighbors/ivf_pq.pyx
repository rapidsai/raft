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
from cuda.ccudart cimport cudaDataType_t

from libcpp cimport bool, nullptr
from pylibraft.distance.distance_type cimport DistanceType
from pylibraft.common.handle cimport handle_t
from rmm._lib.memory_resource cimport device_memory_resource

cdef extern from "raft/neighbors/ann_types.hpp" \
        namespace "raft::neighbors::ann":

    cdef cppclass ann_index "raft::neighbors::index":
        pass

    cdef cppclass ann_index_params "raft::spatial::knn::index_params":
        DistanceType metric
        float metric_arg
        bool add_data_on_build

    cdef cppclass ann_search_params "raft::spatial::knn::search_params":
        pass

cdef extern from "raft/neighbors/ivf_pq_types.hpp" \
        namespace "raft::neighbors::ivf_pq":

    ctypedef enum codebook_gen:
        PER_SUBSPACE "raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE",        
        PER_CLUSTER "raft::neighbors::ivf_pq::codebook_gen::PER_CLUSTER"        


    cdef cppclass index_params(ann_index_params):
        uint32_t n_lists
        uint32_t kmeans_n_iters
        double kmeans_trainset_fraction
        uint32_t pq_bits
        uint32_t pq_dim 
        codebook_gen codebook_kind
        bool force_random_rotation

    cdef cppclass index[IdxT](ann_index):
        index(const handle_t& handle,
              DistanceType metric,
              codebook_gen codebook_kind,
              uint32_t n_lists,
              uint32_t dim,
              uint32_t pq_bits,
              uint32_t pq_dim,
              uint32_t n_nonempty_lists)

    cdef cppclass search_params(ann_search_params):
        uint32_t n_probes
        cudaDataType_t lut_dtype
        cudaDataType_t internal_distance_dtype

cdef extern from "raft/neighbors/specializations/ivf_pq_specialization.hpp" \
        namespace "raft::neighbors::ivf_pq":

    cdef index[uint64_t] build(const handle_t& handle,      
             const index_params& params,     
             const float* dataset,               
             uint64_t n_rows,                    
             uint32_t dim)  

    cdef index[uint64_t] build(const handle_t& handle,      
             const index_params& params,     
             const int8_t* dataset,               
             uint64_t n_rows,                    
             uint32_t dim)

    cdef index[uint64_t] build(const handle_t& handle,      
             const index_params& params,     
             const uint8_t* dataset,               
             uint64_t n_rows,                    
             uint32_t dim)   

    cdef index[uint64_t] extend(const handle_t& handle,        
              const index[uint64_t]& orig_index, 
              const float* new_vectors,          
              const uint64_t* new_indices,       
              uint64_t n_rows)

    cdef index[uint64_t] extend(const handle_t& handle,        
              const index[uint64_t]& orig_index, 
              const int8_t* new_vectors,          
              const uint64_t* new_indices,       
              uint64_t n_rows)    

    cdef index[uint64_t] extend(const handle_t& handle,        
              const index[uint64_t]& orig_index, 
              const uint8_t* new_vectors,          
              const uint64_t* new_indices,       
              uint64_t n_rows)                  

    cdef void search(const handle_t& handle,
                   const search_params& params,
                   const index[uint64_t]& index,
                   const float* queries,
                   uint32_t n_queries,
                   uint32_t k,
                   uint64_t* neighbors,
                   float* distances,
                   device_memory_resource* mr)
    
    cdef void search(const handle_t& handle,
                   const search_params& params,
                   const index[uint64_t]& index,
                   const int8_t* queries,
                   uint32_t n_queries,
                   uint32_t k,
                   uint64_t* neighbors,
                   float* distances,
                   device_memory_resource* mr)
    
    cdef void search(const handle_t& handle,
                   const search_params& params,
                   const index[uint64_t]& index,
                   const uint8_t* queries,
                   uint32_t n_queries,
                   uint32_t k,
                   uint64_t* neighbors,
                   float* distances,
                   device_memory_resource* mr)


def is_c_cont(cai, dt):
    return "strides" not in cai or \
        cai["strides"] is None or \
        cai["strides"][1] == dt.itemsize

    
def _get_codebook_kind(kind):
    return {
        'per_subspace': 0, # codebook_gen.PER_SUBSPACE
        'per_cluster': 1 # codebook_gen.PER_CLUSTER
        }[kind]


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


class IvfPq:
    """
    Nearest neighbors search using IVF-PQ method.
    """

    # Class variables to provide easier access for data type parameters for the search function.
    CUDA_R_32F = cudaDataType_t.CUDA_R_32F
    CUDA_R_16F = cudaDataType_t.CUDA_R_16F
    CUDA_R_8U = cudaDataType_t.CUDA_R_8U

    def __init__(self, *, 
                 handle=None, 
                 n_lists = 1024, 
                 metric="euclidean",
                 kmeans_n_iters=20, 
                 kmeans_trainset_fraction=0.5,
                 pq_bits=8,
                 pq_dim=0,
                 codebook_kind="per_subspace",
                 force_random_rotation=False,
                 add_data_on_build=True):
        """"
        Approximate nearest neighbor search using IVF-PQ method.
    
        Parameters
        ----------
        n_list : int, default = 1024
            The number of clusters used in the coarse quantizer.
        metric : string denoting the metric type, default="euclidean"
            Valid values for metric: ["l2_expanded", "inner_product"],
            where sqeuclidean is the equclidean distance without the square root operation.
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
        
        Examples
        --------

        .. code-block:: python

            import cupy as cp

            from pylibraft.neighbors import IvfPq

            n_samples = 5000
            n_features = 50
            n_queries = 1000

            dataset = cp.random.random_sample((n_samples, n_features),
                                              dtype=cp.float32)
            queries = cp.random.random_sample((n_samples, n_features),
                                              dtype=cp.float32)
            out_idx = cp.empty((n_samples, n_samples), dtype=cp.uint64)
            out_dist = cp.empty((n_samples, n_samples), dtype=cp.float32)

            nn = IvfPQ()
            nn.build(dataset)
            [out_idx, out_dist] = nn.search(queries)
        """
        self.handle = pylibraft.common.handle.Handle() if handle is None \
            else handle

        self._n_lists = n_lists
        self._metric = metric
        self._kmeans_n_iters = kmeans_n_iters
        self._kmeans_trainset_fraction = kmeans_trainset_fraction
        self._pq_bits = pq_bits
        self._pq_dim = pq_dim
        self._codebook_kind = codebook_kind
        self._force_random_rotation = force_random_rotation
        self._add_data_on_build = add_data_on_build

        self._index = None

    def __del__(self):
        self._dealloc()

    def _dealloc(self):
        # deallocate the index
        cdef index[uint64_t] *idx
        if self._index is not None:
            idx = <index[uint64_t]*><uintptr_t>self._index
            del idx

    def build(self, dataset):
        """
        Builds an IVF-PQ index that can be later used for nearest neighbor search.

        Parameters
        ----------
        dataset : CUDA array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float, int8, uint8] 

        """
        # TODO(tfeher): ensure that this works with managed memory as well
        dataset_cai = dataset.__cuda_array_interface__
        dataset_dt = np.dtype(dataset_cai["typestr"])
        if not is_c_cont(dataset_cai, dataset_dt):
            raise ValueError("Row major input is expected")
        
        cdef index_params params
        params.n_lists = self._n_lists
        params.metric = _get_metric(self._metric)
        params.metric_arg = 0
        params.kmeans_n_iters = self._kmeans_n_iters
        params.kmeans_trainset_fraction = self._kmeans_trainset_fraction
        params.pq_bits = self._pq_bits
        params.pq_dim = self._pq_dim
        #params.codebook_kind = _get_codebook_kind(self._codebook_kind)
        params.force_random_rotation = self._force_random_rotation
        params.add_data_on_build = self._add_data_on_build


        # cdef index[uint64_t] *index_ptr
        cdef uint64_t n_rows = dataset_cai["shape"][0] # make it uint32_t
        self._dim = dataset_cai["shape"][1]
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t dataset_ptr = dataset_cai["data"][0]
        
        self._dealloc()

        cdef index[uint64_t] *idx = new index[uint64_t](deref(handle_), 
                                 _get_metric(self._metric), 
                                 codebook_gen.PER_SUBSPACE, 
                                 <uint32_t>self._n_lists,
                                 <uint32_t>self._dim, 
                                 <uint32_t>self._pq_bits,
                                 <uint32_t>self._pq_dim,
                                 <uint32_t>0)
        
        if dataset_dt == np.float32:
            idx[0] = build(deref(handle_),      
                           params,     
                           <float*> dataset_ptr,               
                           n_rows,
                           <uint32_t> self._dim)
        elif dataset_dt == np.byte:
            idx[0] = build(deref(handle_),      
                           params,     
                           <int8_t*> dataset_ptr,               
                           n_rows,
                           <uint32_t> self._dim)
        elif dataset_dt == np.ubyte:
            idx[0] = build(deref(handle_),      
                           params,     
                           <uint8_t*> dataset_ptr,               
                           n_rows,
                           <uint32_t> self._dim)  
        else:
            raise ValueError("dtype %s not supported" % dataset_dt)

        self._index = <uintptr_t>idx 

        self.handle.sync()      

    def extend(self, new_vectors, new_indices):
        """
        Extend an existing index with new vectors.
        
        
        Parameters
        ----------
        new_vectors : CUDA array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float, int8, uint8] 
        new_indices : CUDA array interface compliant matrix shape (n_samples, dim)
            Supported dtype [uint64t] 
        """
        if self._index is None:
            raise ValueError("Index need to be built before calling extend.")

        vecs_cai = new_vectors.__cuda_array_interface__
        vecs_dt = np.dtype(vecs_cai["typestr"])
        cdef uint32_t n_rows = vecs_cai["shape"][0]
        cdef uint32_t dim = vecs_cai["shape"][1]

        assert(vecs_dt in [np.dtype('float32'), np.dtype('byte'), np.dtype('ubyte') ])
        assert(dim == self._dim)

        idx_cai = new_indices.__cuda_array_interface__
        assert(n_rows == idx_cai["shape"][0])
        assert(dim == idx_cai["shape"][1])
        idx_dt = np.dtype(vecs_cai["typestr"])
        assert(idx_dt in [np.dtype('uint64')])

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef index[uint64_t] *idx = <index[uint64_t]*><uintptr_t>self._index
        cdef uintptr_t vecs_ptr = vecs_cai["data"][0]
        cdef uintptr_t idx_ptr = idx_cai["data"][0]

        if vecs_dt == np.float32:
            idx[0] = extend(deref(handle_),
                            deref(idx),
                            <float*>vecs_ptr,
                            <uint64_t*> idx_ptr,
                            <uint64_t> n_rows)
        elif vecs_dt == np.int8:
            idx[0] = extend(deref(handle_),
                            deref(idx),
                            <int8_t*>vecs_ptr,
                            <uint64_t*> idx_ptr,
                            <uint64_t> n_rows)
        elif vecs_dt == np.uint8:
            idx[0] = extend(deref(handle_),
                            deref(idx),
                            <uint8_t*>vecs_ptr,
                            <uint64_t*> idx_ptr,
                            <uint64_t> n_rows)       
        else:
            raise ValueError("query dtype %s not supported" % vecs_dt)

        self.handle.sync()  



    def search(self, 
               queries,
               k,
               neighbors, 
               distances,
               n_probes=20, 
               lut_dtype=cudaDataType_t.CUDA_R_32F, 
               internal_distance_dtype=cudaDataType_t.CUDA_R_32F, 
               preferred_thread_block_size=0):
        """
        Find the k nearest neighbors for each query.

        Parameters
        ----------
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
        n_probes: int, default = 1024
            The number of course clusters to select for the fine search.
        lut_dtype: default = CUDA_R_16F (half)
            Data type of look up table to be created dynamically at search time. The use of 
            low-precision types reduces the amount of shared memory required at search time, so
            fast shared memory kernels can be used even for datasets with large dimansionality.
            Note that the recall is slightly degraded when low-precision type is selected.
            Possible values [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]
        internal_distance_dtype: default = CUDA_R_32F (float)
            Storage data type for distance/similarity computation.
            Possible values [CUDA_R_32F, CUDA_R_16F]

        Returns
        -------
        A pair of [neighbors, distances] arrays as defined above.

        # cudaDataType_t
        CUDA_R_32F = 0  # float
        CUDA_R_16F = 2  # half
        CUDA_R_8U = 8  # uint8
        """

        if self._index is None:
            raise ValueError("Index need to be built before calling search.")

        assert(n_probes > 0)
        assert(k > 0)

        queries_cai = queries.__cuda_array_interface__
        queries_dt = np.dtype(queries_cai["typestr"])
        cdef uint32_t n_queries = queries_cai["shape"][0]
        cdef uint32_t dim_queries = queries_cai["shape"][1]
        assert(n_queries > 0)
        assert(queries_dt in [np.dtype('float32'), np.dtype('byte'), np.dtype('ubyte') ])

        assert(dim_queries == self._dim)

        neighbors_cai = neighbors.__cuda_array_interface__
        neighbors_dt = np.dtype(neighbors_cai["typestr"])
        assert(neighbors_cai["shape"][0] == n_queries)
        assert(neighbors_cai["shape"][1] == k)
        assert(neighbors_dt is np.dtype('uint64'))

        distances_cai = distances.__cuda_array_interface__
        distances_dt = np.dtype(distances_cai["typestr"])
        assert(distances_cai["shape"][0] == n_queries)
        assert(distances_cai["shape"][1] == k)
        assert(distances_dt is np.dtype('float32'))

        cdef search_params params
        params.n_probes = n_probes
        # params.lut_dtype = lut_dtype
        # params.internal_distance_dtype = internal_distance_dtype

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef index[uint64_t] *idx = <index[uint64_t]*><uintptr_t>self._index
        cdef uintptr_t queries_ptr = queries_cai["data"][0]
        cdef uintptr_t neighbors_ptr = neighbors_cai["data"][0]
        cdef uintptr_t distances_ptr = distances_cai["data"][0]
        cdef device_memory_resource* mr_ptr = <device_memory_resource*> nullptr

        if queries_dt == np.float32:
            search(deref(handle_),
                params,
                deref(idx),
                <float*>queries_ptr,
                <uint32_t> n_queries,
                <uint32_t> k,
                <uint64_t*> neighbors_ptr,
                <float*> distances_ptr,
                mr_ptr)
        elif queries_dt == np.byte:
            search(deref(handle_),
                params,
                deref(idx),
                <int8_t*>queries_ptr,
                <uint32_t> n_queries,
                <uint32_t> k,
                <uint64_t*> neighbors_ptr,
                <float*> distances_ptr,
                mr_ptr)
        elif queries_dt == np.ubyte:
            search(deref(handle_),
                params,
                deref(idx),
                <uint8_t*>queries_ptr,
                <uint32_t> n_queries,
                <uint32_t> k,
                <uint64_t*> neighbors_ptr,
                <float*> distances_ptr,
                mr_ptr)
        else:
            raise ValueError("query dtype %s not supported" % queries_dt)

        self.handle.sync()      

        
