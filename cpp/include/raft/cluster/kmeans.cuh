/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <raft/cluster/detail/kmeans.cuh>
#include <raft/cluster/detail/kmeans_auto_find_k.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <optional>

namespace raft::cluster::kmeans {

/**
 * Functor used for sampling centroids
 */
template <typename DataT, typename IndexT>
using SamplingOp = detail::SamplingOp<DataT, IndexT>;

/**
 * Functor used to extract the index from a KeyValue pair
 * storing both index and a distance.
 */
template <typename IndexT, typename DataT>
using KeyValueIndexOp = detail::KeyValueIndexOp<IndexT, DataT>;

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <raft/cluster/kmeans.cuh>
 *   #include <raft/cluster/kmeans_types.hpp>
 *   using namespace raft::cluster;
 *   ...
 *   raft::raft::resources handle;
 *   raft::cluster::KMeansParams params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids,
 *               raft::make_scalar_view(&inertia),
 *               raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 *                              centroids as the initial cluster centers.
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
template <typename DataT, typename IndexT>
void fit(raft::resources const& handle,
         const KMeansParams& params,
         raft::device_matrix_view<const DataT, IndexT> X,
         std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
         raft::device_matrix_view<DataT, IndexT> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<IndexT> n_iter)
{
  detail::kmeans_fit<DataT, IndexT>(handle, params, X, sample_weight, centroids, inertia, n_iter);
}

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <raft/cluster/kmeans.cuh>
 *   #include <raft/cluster/kmeans_types.hpp>
 *   using namespace raft::cluster;
 *   ...
 *   raft::raft::resources handle;
 *   raft::cluster::KMeansParams params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids.view(),
 *               raft::make_scalar_view(&inertia),
 *               raft::make_scalar_view(&n_iter));
 *   ...
 *   auto labels = raft::make_device_vector<int, int>(handle, X.extent(0));
 *
 *   kmeans::predict(handle,
 *                   params,
 *                   X,
 *                   std::nullopt,
 *                   centroids.view(),
 *                   false,
 *                   labels.view(),
 *                   raft::make_scalar_view(&ineratia));
 * @endcode
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 * @param[in]     handle           The raft handle.
 * @param[in]     params           Parameters for KMeans model.
 * @param[in]     X                New data to predict.
 *                                 [dim = n_samples x n_features]
 * @param[in]     sample_weight    Optional weights for each observation in X.
 *                                 [len = n_samples]
 * @param[in]     centroids        Cluster centroids. The data must be in
 *                                 row-major format.
 *                                 [dim = n_clusters x n_features]
 * @param[in]     normalize_weight True if the weights should be normalized
 * @param[out]    labels           Index of the cluster each sample in X
 *                                 belongs to.
 *                                 [len = n_samples]
 * @param[out]    inertia          Sum of squared distances of samples to
 *                                 their closest cluster center.
 */
template <typename DataT, typename IndexT>
void predict(raft::resources const& handle,
             const KMeansParams& params,
             raft::device_matrix_view<const DataT, IndexT> X,
             std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
             raft::device_matrix_view<const DataT, IndexT> centroids,
             raft::device_vector_view<IndexT, IndexT> labels,
             bool normalize_weight,
             raft::host_scalar_view<DataT> inertia)
{
  detail::kmeans_predict<DataT, IndexT>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}

/**
 * @brief Compute k-means clustering and predicts cluster index for each sample
 * in the input.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <raft/cluster/kmeans.cuh>
 *   #include <raft/cluster/kmeans_types.hpp>
 *   using namespace raft::cluster;
 *   ...
 *   raft::raft::resources handle;
 *   raft::cluster::KMeansParams params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *   auto labels = raft::make_device_vector<int, int>(handle, X.extent(0));
 *
 *   kmeans::fit_predict(handle,
 *                       params,
 *                       X,
 *                       std::nullopt,
 *                       centroids.view(),
 *                       labels.view(),
 *                       raft::make_scalar_view(&inertia),
 *                       raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must be
 *                              in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     Optional
 *                              [in] When init is InitMethod::Array, use
 *                              centroids  as the initial cluster centers
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    labels        Index of the cluster each sample in X belongs
 *                              to.
 *                              [len = n_samples]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
template <typename DataT, typename IndexT>
void fit_predict(raft::resources const& handle,
                 const KMeansParams& params,
                 raft::device_matrix_view<const DataT, IndexT> X,
                 std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
                 std::optional<raft::device_matrix_view<DataT, IndexT>> centroids,
                 raft::device_vector_view<IndexT, IndexT> labels,
                 raft::host_scalar_view<DataT> inertia,
                 raft::host_scalar_view<IndexT> n_iter)
{
  detail::kmeans_fit_predict<DataT, IndexT>(
    handle, params, X, sample_weight, centroids, labels, inertia, n_iter);
}

/**
 * @brief Transform X to a cluster-distance space.
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format
 *                              [dim = n_samples x n_features]
 * @param[in]     centroids     Cluster centroids. The data must be in row-major format.
 *                              [dim = n_clusters x n_features]
 * @param[out]    X_new         X transformed in the new space.
 *                              [dim = n_samples x n_features]
 */
template <typename DataT, typename IndexT>
void transform(raft::resources const& handle,
               const KMeansParams& params,
               raft::device_matrix_view<const DataT, IndexT> X,
               raft::device_matrix_view<const DataT, IndexT> centroids,
               raft::device_matrix_view<DataT, IndexT> X_new)
{
  detail::kmeans_transform<DataT, IndexT>(handle, params, X, centroids, X_new);
}

template <typename DataT, typename IndexT>
void transform(raft::resources const& handle,
               const KMeansParams& params,
               const DataT* X,
               const DataT* centroids,
               IndexT n_samples,
               IndexT n_features,
               DataT* X_new)
{
  detail::kmeans_transform<DataT, IndexT>(
    handle, params, X, centroids, n_samples, n_features, X_new);
}

/**
 * Automatically find the optimal value of k using a binary search.
 * This method maximizes the Calinski-Harabasz Index while minimizing the per-cluster inertia.
 *
 *  @code{.cpp}
 *   #include <raft/core/handle.hpp>
 *   #include <raft/cluster/kmeans.cuh>
 *   #include <raft/cluster/kmeans_types.hpp>
 *
 *   #include <raft/random/make_blobs.cuh>
 *
 *   using namespace raft::cluster;
 *
 *   raft::handle_t handle;
 *   int n_samples = 100, n_features = 15, n_clusters = 10;
 *   auto X = raft::make_device_matrix<float, int>(handle, n_samples, n_features);
 *   auto labels = raft::make_device_vector<float, int>(handle, n_samples);
 *
 *   raft::random::make_blobs(handle, X, labels, n_clusters);
 *
 *   auto best_k = raft::make_host_scalar<int>(0);
 *   auto n_iter = raft::make_host_scalar<int>(0);
 *   auto inertia = raft::make_host_scalar<int>(0);
 *
 *   kmeans::find_k(handle, X, best_k.view(), inertia.view(), n_iter.view(), n_clusters+1);
 *
 * @endcode
 *
 * @tparam idx_t indexing type (should be integral)
 * @tparam value_t value type (should be floating point)
 * @param handle raft handle
 * @param X input observations (shape n_samples, n_dims)
 * @param best_k best k found from binary search
 * @param inertia inertia of best k found
 * @param n_iter number of iterations used to find best k
 * @param kmax maximum k to try in search
 * @param kmin minimum k to try in search (should be >= 1)
 * @param maxiter maximum number of iterations to run
 * @param tol tolerance for early stopping convergence
 */
template <typename idx_t, typename value_t>
void find_k(raft::resources const& handle,
            raft::device_matrix_view<const value_t, idx_t> X,
            raft::host_scalar_view<idx_t> best_k,
            raft::host_scalar_view<value_t> inertia,
            raft::host_scalar_view<idx_t> n_iter,
            idx_t kmax,
            idx_t kmin    = 1,
            idx_t maxiter = 100,
            value_t tol   = 1e-3)
{
  detail::find_k(handle, X, best_k, inertia, n_iter, kmax, kmin, maxiter, tol);
}

/**
 * @brief Select centroids according to a sampling operation
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle             The raft handle
 * @param[in]  X                  The data in row-major format
 *                                [dim = n_samples x n_features]
 * @param[in]  minClusterDistance Distance for every sample to it's nearest centroid
 *                                [dim = n_samples]
 * @param[in]  isSampleCentroid   Flag the sample chosen as initial centroid
 *                                [dim = n_samples]
 * @param[in]  select_op          The sampling operation used to select the centroids
 * @param[out] inRankCp           The sampled centroids
 *                                [dim = n_selected_centroids x n_features]
 * @param[in]  workspace          Temporary workspace buffer which can get resized
 *
 */
template <typename DataT, typename IndexT>
void sample_centroids(raft::resources const& handle,
                      raft::device_matrix_view<const DataT, IndexT> X,
                      raft::device_vector_view<DataT, IndexT> minClusterDistance,
                      raft::device_vector_view<std::uint8_t, IndexT> isSampleCentroid,
                      SamplingOp<DataT, IndexT>& select_op,
                      rmm::device_uvector<DataT>& inRankCp,
                      rmm::device_uvector<char>& workspace)
{
  detail::sampleCentroids<DataT, IndexT>(
    handle, X, minClusterDistance, isSampleCentroid, select_op, inRankCp, workspace);
}

/**
 * @brief Compute cluster cost
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam ReductionOpT the type of data used for the reduction operation.
 *
 * @param[in]  handle             The raft handle
 * @param[in]  minClusterDistance Distance for every sample to it's nearest centroid
 *                                [dim = n_samples]
 * @param[in]  workspace          Temporary workspace buffer which can get resized
 * @param[out] clusterCost        Resulting cluster cost
 * @param[in]  reduction_op       The reduction operation used for the cost
 *
 */
template <typename DataT, typename IndexT, typename ReductionOpT>
void cluster_cost(raft::resources const& handle,
                  raft::device_vector_view<DataT, IndexT> minClusterDistance,
                  rmm::device_uvector<char>& workspace,
                  raft::device_scalar_view<DataT> clusterCost,
                  ReductionOpT reduction_op)
{
  detail::computeClusterCost(
    handle, minClusterDistance, workspace, clusterCost, raft::identity_op{}, reduction_op);
}

/**
 * @brief Update centroids given current centroids and number of points assigned to each centroid.
 *  This function also produces a vector of RAFT key/value pairs containing the cluster assignment
 *  for each point and its distance.
 *
 * @tparam DataT
 * @tparam IndexT
 * @param[in] handle: Raft handle to use for managing library resources
 * @param[in] X: input matrix (size n_samples, n_features)
 * @param[in] sample_weights: number of samples currently assigned to each centroid (size n_samples)
 * @param[in] centroids: matrix of current centroids (size n_clusters, n_features)
 * @param[in] labels: Iterator of labels (can also be a raw pointer)
 * @param[out] weight_per_cluster: sum of sample weights per cluster (size n_clusters)
 * @param[out] new_centroids: output matrix of updated centroids (size n_clusters, n_features)
 */
template <typename DataT, typename IndexT, typename LabelsIterator>
void update_centroids(raft::resources const& handle,
                      raft::device_matrix_view<const DataT, IndexT, row_major> X,
                      raft::device_vector_view<const DataT, IndexT> sample_weights,
                      raft::device_matrix_view<const DataT, IndexT, row_major> centroids,
                      LabelsIterator labels,
                      raft::device_vector_view<DataT, IndexT> weight_per_cluster,
                      raft::device_matrix_view<DataT, IndexT, row_major> new_centroids)
{
  // TODO: Passing these into the algorithm doesn't really present much of a benefit
  // because they are being resized anyways.
  // ref https://github.com/rapidsai/raft/issues/930
  rmm::device_uvector<char> workspace(0, resource::get_cuda_stream(handle));

  detail::update_centroids<DataT, IndexT>(
    handle, X, sample_weights, centroids, labels, weight_per_cluster, new_centroids, workspace);
}

/**
 * @brief Compute distance for every sample to it's nearest centroid
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle               The raft handle
 * @param[in]  X                    The data in row-major format
 *                                  [dim = n_samples x n_features]
 * @param[in]  centroids            Centroids data
 *                                  [dim = n_cluster x n_features]
 * @param[out] minClusterDistance   Distance for every sample to it's nearest centroid
 *                                  [dim = n_samples]
 * @param[in]  L2NormX              L2 norm of X : ||x||^2
 *                                  [dim = n_samples]
 * @param[out] L2NormBuf_OR_DistBuf Resizable buffer to store L2 norm of centroids or distance
 *                                  matrix
 * @param[in]  metric               Distance metric to use
 * @param[in]  batch_samples        batch size for input data samples
 * @param[in]  batch_centroids      batch size for input centroids
 * @param[in]  workspace            Temporary workspace buffer which can get resized
 *
 */
template <typename DataT, typename IndexT>
void min_cluster_distance(raft::resources const& handle,
                          raft::device_matrix_view<const DataT, IndexT> X,
                          raft::device_matrix_view<DataT, IndexT> centroids,
                          raft::device_vector_view<DataT, IndexT> minClusterDistance,
                          raft::device_vector_view<DataT, IndexT> L2NormX,
                          rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
                          raft::distance::DistanceType metric,
                          int batch_samples,
                          int batch_centroids,
                          rmm::device_uvector<char>& workspace)
{
  detail::minClusterDistanceCompute<DataT, IndexT>(handle,
                                                   X,
                                                   centroids,
                                                   minClusterDistance,
                                                   L2NormX,
                                                   L2NormBuf_OR_DistBuf,
                                                   metric,
                                                   batch_samples,
                                                   batch_centroids,
                                                   workspace);
}

/**
 * @brief Calculates a <key, value> pair for every sample in input 'X' where key is an
 * index of one of the 'centroids' (index of the nearest centroid) and 'value'
 * is the distance between the sample and the 'centroid[key]'
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle                The raft handle
 * @param[in]  X                     The data in row-major format
 *                                   [dim = n_samples x n_features]
 * @param[in]  centroids             Centroids data
 *                                   [dim = n_cluster x n_features]
 * @param[out] minClusterAndDistance Distance vector that contains for every sample, the nearest
 *                                   centroid and it's distance
 *                                   [dim = n_samples]
 * @param[in]  L2NormX               L2 norm of X : ||x||^2
 *                                   [dim = n_samples]
 * @param[out] L2NormBuf_OR_DistBuf  Resizable buffer to store L2 norm of centroids or distance
 *                                   matrix
 * @param[in] metric                 distance metric
 * @param[in] batch_samples          batch size of data samples
 * @param[in] batch_centroids        batch size of centroids
 * @param[in] workspace              Temporary workspace buffer which can get resized
 *
 */
template <typename DataT, typename IndexT>
void min_cluster_and_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<const DataT, IndexT> centroids,
  raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT> minClusterAndDistance,
  raft::device_vector_view<DataT, IndexT> L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  raft::distance::DistanceType metric,
  int batch_samples,
  int batch_centroids,
  rmm::device_uvector<char>& workspace)
{
  detail::minClusterAndDistanceCompute<DataT, IndexT>(handle,
                                                      X,
                                                      centroids,
                                                      minClusterAndDistance,
                                                      L2NormX,
                                                      L2NormBuf_OR_DistBuf,
                                                      metric,
                                                      batch_samples,
                                                      batch_centroids,
                                                      workspace);
}

/**
 * @brief Shuffle and randomly select 'n_samples_to_gather' from input 'in' and stores
 * in 'out' does not modify the input
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle              The raft handle
 * @param[in]  in                  The data to shuffle and gather
 *                                 [dim = n_samples x n_features]
 * @param[out] out                 The sampled data
 *                                 [dim = n_samples_to_gather x n_features]
 * @param[in]  n_samples_to_gather Number of sample to gather
 * @param[in]  seed                Seed for the shuffle
 *
 */
template <typename DataT, typename IndexT>
void shuffle_and_gather(raft::resources const& handle,
                        raft::device_matrix_view<const DataT, IndexT> in,
                        raft::device_matrix_view<DataT, IndexT> out,
                        uint32_t n_samples_to_gather,
                        uint64_t seed)
{
  detail::shuffleAndGather<DataT, IndexT>(handle, in, out, n_samples_to_gather, seed);
}

/**
 * @brief Count the number of samples in each cluster
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle               The raft handle
 * @param[in]  params               The parameters for KMeans
 * @param[in]  X                    The data in row-major format
 *                                  [dim = n_samples x n_features]
 * @param[in]  L2NormX              L2 norm of X : ||x||^2
 *                                  [dim = n_samples]
 * @param[in]  centroids            Centroids data
 *                                  [dim = n_cluster x n_features]
 * @param[in]  workspace            Temporary workspace buffer which can get resized
 * @param[out] sampleCountInCluster The count for each centroid
 *                                  [dim = n_cluster]
 *
 */
template <typename DataT, typename IndexT>
void count_samples_in_cluster(raft::resources const& handle,
                              const KMeansParams& params,
                              raft::device_matrix_view<const DataT, IndexT> X,
                              raft::device_vector_view<DataT, IndexT> L2NormX,
                              raft::device_matrix_view<DataT, IndexT> centroids,
                              rmm::device_uvector<char>& workspace,
                              raft::device_vector_view<DataT, IndexT> sampleCountInCluster)
{
  detail::countSamplesInCluster<DataT, IndexT>(
    handle, params, X, L2NormX, centroids, workspace, sampleCountInCluster);
}

/**
 * @brief Selects 'n_clusters' samples from the input X using kmeans++ algorithm.
 *
 * @see "k-means++: the advantages of careful seeding". 2007, Arthur, D. and Vassilvitskii, S.
 *        ACM-SIAM symposium on Discrete algorithms.
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle                The raft handle
 * @param[in]  params                The parameters for KMeans
 * @param[in]  X                     The data in row-major format
 *                                   [dim = n_samples x n_features]
 * @param[out] centroids             Centroids data
 *                                   [dim = n_cluster x n_features]
 * @param[in]  workspace             Temporary workspace buffer which can get resized
 */
template <typename DataT, typename IndexT>
void init_plus_plus(raft::resources const& handle,
                    const KMeansParams& params,
                    raft::device_matrix_view<const DataT, IndexT> X,
                    raft::device_matrix_view<DataT, IndexT> centroids,
                    rmm::device_uvector<char>& workspace)
{
  detail::kmeansPlusPlus<DataT, IndexT>(handle, params, X, centroids, workspace);
}

/*
 * @brief Main function used to fit KMeans (after cluster initialization)
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     [in] Initial cluster centers.
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 * @param[in]     workspace     Temporary workspace buffer which can get resized
 */
template <typename DataT, typename IndexT>
void fit_main(raft::resources const& handle,
              const KMeansParams& params,
              raft::device_matrix_view<const DataT, IndexT> X,
              raft::device_vector_view<const DataT, IndexT> sample_weights,
              raft::device_matrix_view<DataT, IndexT> centroids,
              raft::host_scalar_view<DataT> inertia,
              raft::host_scalar_view<IndexT> n_iter,
              rmm::device_uvector<char>& workspace)
{
  detail::kmeans_fit_main<DataT, IndexT>(
    handle, params, X, sample_weights, centroids, inertia, n_iter, workspace);
}

};  // end namespace raft::cluster::kmeans

namespace raft::cluster {

/**
 * Note: All of the functions below in raft::cluster are deprecated and will
 * be removed in a future release. Please use raft::cluster::kmeans instead.
 */

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 *                              centroids as the initial cluster centers.
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
template <typename DataT, typename IndexT = int>
void kmeans_fit(raft::resources const& handle,
                const KMeansParams& params,
                raft::device_matrix_view<const DataT, IndexT> X,
                std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
                raft::device_matrix_view<DataT, IndexT> centroids,
                raft::host_scalar_view<DataT> inertia,
                raft::host_scalar_view<IndexT> n_iter)
{
  kmeans::fit<DataT, IndexT>(handle, params, X, sample_weight, centroids, inertia, n_iter);
}

template <typename DataT, typename IndexT = int>
void kmeans_fit(raft::resources const& handle,
                const KMeansParams& params,
                const DataT* X,
                const DataT* sample_weight,
                DataT* centroids,
                IndexT n_samples,
                IndexT n_features,
                DataT& inertia,
                IndexT& n_iter)
{
  kmeans::fit<DataT, IndexT>(
    handle, params, X, sample_weight, centroids, n_samples, n_features, inertia, n_iter);
}

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 * @param[in]     handle           The raft handle.
 * @param[in]     params           Parameters for KMeans model.
 * @param[in]     X                New data to predict.
 *                                 [dim = n_samples x n_features]
 * @param[in]     sample_weight    Optional weights for each observation in X.
 *                                 [len = n_samples]
 * @param[in]     centroids        Cluster centroids. The data must be in
 *                                 row-major format.
 *                                 [dim = n_clusters x n_features]
 * @param[in]     normalize_weight True if the weights should be normalized
 * @param[out]    labels           Index of the cluster each sample in X
 *                                 belongs to.
 *                                 [len = n_samples]
 * @param[out]    inertia          Sum of squared distances of samples to
 *                                 their closest cluster center.
 */
template <typename DataT, typename IndexT = int>
void kmeans_predict(raft::resources const& handle,
                    const KMeansParams& params,
                    raft::device_matrix_view<const DataT, IndexT> X,
                    std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
                    raft::device_matrix_view<const DataT, IndexT> centroids,
                    raft::device_vector_view<IndexT, IndexT> labels,
                    bool normalize_weight,
                    raft::host_scalar_view<DataT> inertia)
{
  kmeans::predict<DataT, IndexT>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}

template <typename DataT, typename IndexT = int>
void kmeans_predict(raft::resources const& handle,
                    const KMeansParams& params,
                    const DataT* X,
                    const DataT* sample_weight,
                    const DataT* centroids,
                    IndexT n_samples,
                    IndexT n_features,
                    IndexT* labels,
                    bool normalize_weight,
                    DataT& inertia)
{
  kmeans::predict<DataT, IndexT>(handle,
                                 params,
                                 X,
                                 sample_weight,
                                 centroids,
                                 n_samples,
                                 n_features,
                                 labels,
                                 normalize_weight,
                                 inertia);
}

/**
 * @brief Compute k-means clustering and predicts cluster index for each sample
 * in the input.
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must be
 *                              in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     Optional
 *                              [in] When init is InitMethod::Array, use
 *                              centroids  as the initial cluster centers
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    labels        Index of the cluster each sample in X belongs
 *                              to.
 *                              [len = n_samples]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
template <typename DataT, typename IndexT = int>
void kmeans_fit_predict(raft::resources const& handle,
                        const KMeansParams& params,
                        raft::device_matrix_view<const DataT, IndexT> X,
                        std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
                        std::optional<raft::device_matrix_view<DataT, IndexT>> centroids,
                        raft::device_vector_view<IndexT, IndexT> labels,
                        raft::host_scalar_view<DataT> inertia,
                        raft::host_scalar_view<IndexT> n_iter)
{
  kmeans::fit_predict<DataT, IndexT>(
    handle, params, X, sample_weight, centroids, labels, inertia, n_iter);
}

template <typename DataT, typename IndexT = int>
void kmeans_fit_predict(raft::resources const& handle,
                        const KMeansParams& params,
                        const DataT* X,
                        const DataT* sample_weight,
                        DataT* centroids,
                        IndexT n_samples,
                        IndexT n_features,
                        IndexT* labels,
                        DataT& inertia,
                        IndexT& n_iter)
{
  kmeans::fit_predict<DataT, IndexT>(
    handle, params, X, sample_weight, centroids, n_samples, n_features, labels, inertia, n_iter);
}

/**
 * @brief Transform X to a cluster-distance space.
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format
 *                              [dim = n_samples x n_features]
 * @param[in]     centroids     Cluster centroids. The data must be in row-major format.
 *                              [dim = n_clusters x n_features]
 * @param[out]    X_new         X transformed in the new space.
 *                              [dim = n_samples x n_features]
 */
template <typename DataT, typename IndexT = int>
void kmeans_transform(raft::resources const& handle,
                      const KMeansParams& params,
                      raft::device_matrix_view<const DataT, IndexT> X,
                      raft::device_matrix_view<const DataT, IndexT> centroids,
                      raft::device_matrix_view<DataT, IndexT> X_new)
{
  kmeans::transform<DataT, IndexT>(handle, params, X, centroids, X_new);
}

template <typename DataT, typename IndexT = int>
void kmeans_transform(raft::resources const& handle,
                      const KMeansParams& params,
                      const DataT* X,
                      const DataT* centroids,
                      IndexT n_samples,
                      IndexT n_features,
                      DataT* X_new)
{
  kmeans::transform<DataT, IndexT>(handle, params, X, centroids, n_samples, n_features, X_new);
}

template <typename DataT, typename IndexT>
using SamplingOp = kmeans::SamplingOp<DataT, IndexT>;

template <typename IndexT, typename DataT>
using KeyValueIndexOp = kmeans::KeyValueIndexOp<IndexT, DataT>;

/**
 * @brief Select centroids according to a sampling operation
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle             The raft handle
 * @param[in]  X                  The data in row-major format
 *                                [dim = n_samples x n_features]
 * @param[in]  minClusterDistance Distance for every sample to it's nearest centroid
 *                                [dim = n_samples]
 * @param[in]  isSampleCentroid   Flag the sample chosen as initial centroid
 *                                [dim = n_samples]
 * @param[in]  select_op          The sampling operation used to select the centroids
 * @param[out] inRankCp           The sampled centroids
 *                                [dim = n_selected_centroids x n_features]
 * @param[in]  workspace          Temporary workspace buffer which can get resized
 *
 */
template <typename DataT, typename IndexT>
void sampleCentroids(raft::resources const& handle,
                     raft::device_matrix_view<const DataT, IndexT> X,
                     raft::device_vector_view<DataT, IndexT> minClusterDistance,
                     raft::device_vector_view<std::uint8_t, IndexT> isSampleCentroid,
                     SamplingOp<DataT, IndexT>& select_op,
                     rmm::device_uvector<DataT>& inRankCp,
                     rmm::device_uvector<char>& workspace)
{
  kmeans::sample_centroids<DataT, IndexT>(
    handle, X, minClusterDistance, isSampleCentroid, select_op, inRankCp, workspace);
}

/**
 * @brief Compute cluster cost
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam ReductionOpT the type of data used for the reduction operation.
 *
 * @param[in]  handle             The raft handle
 * @param[in]  minClusterDistance Distance for every sample to it's nearest centroid
 *                                [dim = n_samples]
 * @param[in]  workspace          Temporary workspace buffer which can get resized
 * @param[out] clusterCost        Resulting cluster cost
 * @param[in]  reduction_op       The reduction operation used for the cost
 *
 */
template <typename DataT, typename IndexT, typename ReductionOpT>
void computeClusterCost(raft::resources const& handle,
                        raft::device_vector_view<DataT, IndexT> minClusterDistance,
                        rmm::device_uvector<char>& workspace,
                        raft::device_scalar_view<DataT> clusterCost,
                        ReductionOpT reduction_op)
{
  kmeans::cluster_cost(handle, minClusterDistance, workspace, clusterCost, reduction_op);
}

/**
 * @brief Compute distance for every sample to it's nearest centroid
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle               The raft handle
 * @param[in]  params               The parameters for KMeans
 * @param[in]  X                    The data in row-major format
 *                                  [dim = n_samples x n_features]
 * @param[in]  centroids            Centroids data
 *                                  [dim = n_cluster x n_features]
 * @param[out] minClusterDistance   Distance for every sample to it's nearest centroid
 *                                  [dim = n_samples]
 * @param[in]  L2NormX              L2 norm of X : ||x||^2
 *                                  [dim = n_samples]
 * @param[out] L2NormBuf_OR_DistBuf Resizable buffer to store L2 norm of centroids or distance
 *                                  matrix
 * @param[in]  workspace            Temporary workspace buffer which can get resized
 *
 */
template <typename DataT, typename IndexT>
void minClusterDistanceCompute(raft::resources const& handle,
                               const KMeansParams& params,
                               raft::device_matrix_view<const DataT, IndexT> X,
                               raft::device_matrix_view<DataT, IndexT> centroids,
                               raft::device_vector_view<DataT, IndexT> minClusterDistance,
                               raft::device_vector_view<DataT, IndexT> L2NormX,
                               rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
                               rmm::device_uvector<char>& workspace)
{
  kmeans::min_cluster_distance<DataT, IndexT>(handle,
                                              X,
                                              centroids,
                                              minClusterDistance,
                                              L2NormX,
                                              L2NormBuf_OR_DistBuf,
                                              params.metric,
                                              params.batch_samples,
                                              params.batch_centroids,
                                              workspace);
}

/**
 * @brief Calculates a <key, value> pair for every sample in input 'X' where key is an
 * index of one of the 'centroids' (index of the nearest centroid) and 'value'
 * is the distance between the sample and the 'centroid[key]'
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle                The raft handle
 * @param[in]  params                The parameters for KMeans
 * @param[in]  X                     The data in row-major format
 *                                   [dim = n_samples x n_features]
 * @param[in]  centroids             Centroids data
 *                                   [dim = n_cluster x n_features]
 * @param[out] minClusterAndDistance Distance vector that contains for every sample, the nearest
 *                                   centroid and it's distance
 *                                   [dim = n_samples]
 * @param[in]  L2NormX               L2 norm of X : ||x||^2
 *                                   [dim = n_samples]
 * @param[out] L2NormBuf_OR_DistBuf  Resizable buffer to store L2 norm of centroids or distance
 *                                   matrix
 * @param[in]  workspace             Temporary workspace buffer which can get resized
 *
 */
template <typename DataT, typename IndexT>
void minClusterAndDistanceCompute(
  raft::resources const& handle,
  const KMeansParams& params,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<const DataT, IndexT> centroids,
  raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT> minClusterAndDistance,
  raft::device_vector_view<DataT, IndexT> L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  rmm::device_uvector<char>& workspace)
{
  kmeans::min_cluster_and_distance<DataT, IndexT>(handle,
                                                  X,
                                                  centroids,
                                                  minClusterAndDistance,
                                                  L2NormX,
                                                  L2NormBuf_OR_DistBuf,
                                                  params.metric,
                                                  params.batch_samples,
                                                  params.batch_centroids,
                                                  workspace);
}

/**
 * @brief Shuffle and randomly select 'n_samples_to_gather' from input 'in' and stores
 * in 'out' does not modify the input
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle              The raft handle
 * @param[in]  in                  The data to shuffle and gather
 *                                 [dim = n_samples x n_features]
 * @param[out] out                 The sampled data
 *                                 [dim = n_samples_to_gather x n_features]
 * @param[in]  n_samples_to_gather Number of sample to gather
 * @param[in]  seed                Seed for the shuffle
 *
 */
template <typename DataT, typename IndexT>
void shuffleAndGather(raft::resources const& handle,
                      raft::device_matrix_view<const DataT, IndexT> in,
                      raft::device_matrix_view<DataT, IndexT> out,
                      uint32_t n_samples_to_gather,
                      uint64_t seed)
{
  kmeans::shuffle_and_gather<DataT, IndexT>(handle, in, out, n_samples_to_gather, seed);
}

/**
 * @brief Count the number of samples in each cluster
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle               The raft handle
 * @param[in]  params               The parameters for KMeans
 * @param[in]  X                    The data in row-major format
 *                                  [dim = n_samples x n_features]
 * @param[in]  L2NormX              L2 norm of X : ||x||^2
 *                                  [dim = n_samples]
 * @param[in]  centroids            Centroids data
 *                                  [dim = n_cluster x n_features]
 * @param[in]  workspace            Temporary workspace buffer which can get resized
 * @param[out] sampleCountInCluster The count for each centroid
 *                                  [dim = n_cluster]
 *
 */
template <typename DataT, typename IndexT>
void countSamplesInCluster(raft::resources const& handle,
                           const KMeansParams& params,
                           raft::device_matrix_view<const DataT, IndexT> X,
                           raft::device_vector_view<DataT, IndexT> L2NormX,
                           raft::device_matrix_view<DataT, IndexT> centroids,
                           rmm::device_uvector<char>& workspace,
                           raft::device_vector_view<DataT, IndexT> sampleCountInCluster)
{
  kmeans::count_samples_in_cluster<DataT, IndexT>(
    handle, params, X, L2NormX, centroids, workspace, sampleCountInCluster);
}

/*
 * @brief Selects 'n_clusters' samples from the input X using kmeans++ algorithm.

 * @note  This is the algorithm described in
 *        "k-means++: the advantages of careful seeding". 2007, Arthur, D. and Vassilvitskii, S.
 *        ACM-SIAM symposium on Discrete algorithms.
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle                The raft handle
 * @param[in]  params                The parameters for KMeans
 * @param[in]  X                     The data in row-major format
 *                                   [dim = n_samples x n_features]
 * @param[out] centroids             Centroids data
 *                                   [dim = n_cluster x n_features]
 * @param[in]  workspace             Temporary workspace buffer which can get resized
 */
template <typename DataT, typename IndexT>
void kmeansPlusPlus(raft::resources const& handle,
                    const KMeansParams& params,
                    raft::device_matrix_view<const DataT, IndexT> X,
                    raft::device_matrix_view<DataT, IndexT> centroidsRawData,
                    rmm::device_uvector<char>& workspace)
{
  kmeans::init_plus_plus<DataT, IndexT>(handle, params, X, centroidsRawData, workspace);
}

/*
 * @brief Main function used to fit KMeans (after cluster initialization)
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     [in] Initial cluster centers.
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 * @param[in]     workspace     Temporary workspace buffer which can get resized
 */
template <typename DataT, typename IndexT>
void kmeans_fit_main(raft::resources const& handle,
                     const KMeansParams& params,
                     raft::device_matrix_view<const DataT, IndexT> X,
                     raft::device_vector_view<const DataT, IndexT> weight,
                     raft::device_matrix_view<DataT, IndexT> centroidsRawData,
                     raft::host_scalar_view<DataT> inertia,
                     raft::host_scalar_view<IndexT> n_iter,
                     rmm::device_uvector<char>& workspace)
{
  kmeans::fit_main<DataT, IndexT>(
    handle, params, X, weight, centroidsRawData, inertia, n_iter, workspace);
}
};  // namespace raft::cluster
