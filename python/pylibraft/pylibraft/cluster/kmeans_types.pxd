#    TODO: expose this function from raft
#    cdef void kmeans_fit[ElementType, IndexType](
#        const handle_t & handle, 
#        const KMeansParams& params,
#        device_matrix_view[const ElementType, IndexType] X,
#        optional[device_vector_view[const ElementType, IndexType]] sample_weight,
#        device_matrix_view[ElementType, IndexType] inertia,
#        host_scalar_view[ElementType] inertia,
#        host_scalar_view[IndexType] n_iter) except +
from libcpp cimport bool

from pylibraft.random.rng_state cimport RngState

cdef extern from "raft/cluster/kmeans_types.hpp" \
        namespace "raft::cluster::kmeans":

    ctypedef enum InitMethod 'raft::cluster::KMeansParams::InitMethod':
        KMeansPlusPlus 'raft::cluster::kmeans::KMeansParams::InitMethod::KMeansPlusPlus'
        Random 'raft::cluster::kmeans::KMeansParams::InitMethod::Random'
        Array 'raft::cluster::kmeans::KMeansParams::InitMethod::Array'

    cdef cppclass KMeansParams:
        KMeansParams() except +
        int n_clusters
        InitMethod init
        int max_iter
        double tol
        int verbosity
        RngState rng_state
        int n_init
        double oversampling_factor
        int batch_samples
        int batch_centroids
        bool inertia_check
