/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <optional>
#include <raft/cluster/detail/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/mdarray.hpp>

namespace raft::cluster {
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
void kmeans_fit(handle_t const& handle,
                const KMeansParams& params,
                raft::device_matrix_view<const DataT, IndexT> X,
                std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
                raft::device_matrix_view<DataT, IndexT> centroids,
                raft::host_scalar_view<DataT> inertia,
                raft::host_scalar_view<IndexT> n_iter)
{
  detail::kmeans_fit<DataT, IndexT>(handle, params, X, sample_weight, centroids, inertia, n_iter);
}

template <typename DataT, typename IndexT = int>
void kmeans_fit(handle_t const& handle,
                const KMeansParams& params,
                const DataT* X,
                const DataT* sample_weight,
                DataT* centroids,
                IndexT n_samples,
                IndexT n_features,
                DataT& inertia,
                IndexT& n_iter)
{
  detail::kmeans_fit<DataT, IndexT>(
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
void kmeans_predict(handle_t const& handle,
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

template <typename DataT, typename IndexT = int>
void kmeans_predict(handle_t const& handle,
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
  detail::kmeans_predict<DataT, IndexT>(handle,
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
void kmeans_fit_predict(handle_t const& handle,
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

template <typename DataT, typename IndexT = int>
void kmeans_fit_predict(handle_t const& handle,
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
  detail::kmeans_fit_predict<DataT, IndexT>(
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
void kmeans_transform(const raft::handle_t& handle,
                      const KMeansParams& params,
                      raft::device_matrix_view<const DataT, IndexT> X,
                      raft::device_matrix_view<const DataT, IndexT> centroids,
                      raft::device_matrix_view<DataT, IndexT> X_new)
{
  detail::kmeans_transform<DataT, IndexT>(handle, params, X, centroids, X_new);
}

template <typename DataT, typename IndexT = int>
void kmeans_transform(const raft::handle_t& handle,
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

template <typename DataT, typename IndexT = int>
using SamplingOp = detail::SamplingOp<DataT, IndexT>;

template <typename IndexT, typename DataT>
using KeyValueIndexOp = detail::KeyValueIndexOp<IndexT, DataT>;

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
 * @param[in]  isSampleCentroid   Flag the sample choosen as initial centroid
 *                                [dim = n_samples]
 * @param[in]  select_op          The sampling operation used to select the centroids
 * @param[out] inRankCp           The sampled centroids
 *                                [dim = n_selected_centroids x n_features]
 * @param[in]  workspace          Temporary workspace buffer which can get resized
 *
 */
template <typename DataT, typename IndexT = int>
void sampleCentroids(const raft::handle_t& handle,
                     const raft::device_matrix_view<const DataT, IndexT>& X,
                     const raft::device_vector_view<DataT, IndexT>& minClusterDistance,
                     const raft::device_vector_view<IndexT, IndexT>& isSampleCentroid,
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
template <typename DataT, typename ReductionOpT, typename IndexT = int>
void computeClusterCost(const raft::handle_t& handle,
                        const raft::device_vector_view<DataT, IndexT>& minClusterDistance,
                        rmm::device_uvector<char>& workspace,
                        const raft::device_scalar_view<DataT>& clusterCost,
                        ReductionOpT reduction_op)
{
  detail::computeClusterCost<DataT, ReductionOpT, IndexT>(
    handle, minClusterDistance, workspace, clusterCost, reduction_op);
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
void minClusterDistanceCompute(const raft::handle_t& handle,
                               const KMeansParams& params,
                               const raft::device_matrix_view<const DataT, IndexT>& X,
                               const raft::device_matrix_view<DataT, IndexT>& centroids,
                               const raft::device_vector_view<DataT, IndexT>& minClusterDistance,
                               const raft::device_vector_view<DataT, IndexT>& L2NormX,
                               rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
                               rmm::device_uvector<char>& workspace)
{
  detail::minClusterDistanceCompute<DataT, IndexT>(
    handle, params, X, centroids, minClusterDistance, L2NormX, L2NormBuf_OR_DistBuf, workspace);
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
  const raft::handle_t& handle,
  const KMeansParams& params,
  const raft::device_matrix_view<const DataT, IndexT> X,
  const raft::device_matrix_view<const DataT, IndexT> centroids,
  const raft::device_vector_view<cub::KeyValuePair<IndexT, DataT>, IndexT>& minClusterAndDistance,
  const raft::device_vector_view<DataT, IndexT>& L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  rmm::device_uvector<char>& workspace)
{
  detail::minClusterAndDistanceCompute<DataT, IndexT>(
    handle, params, X, centroids, minClusterAndDistance, L2NormX, L2NormBuf_OR_DistBuf, workspace);
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
 * @param[in]  workspace           Temporary workspace buffer which can get resized
 *
 */
template <typename DataT, typename IndexT>
void shuffleAndGather(const raft::handle_t& handle,
                      const raft::device_matrix_view<const DataT, IndexT>& in,
                      const raft::device_matrix_view<DataT, IndexT>& out,
                      uint32_t n_samples_to_gather,
                      uint64_t seed,
                      rmm::device_uvector<char>* workspace = nullptr)
{
  detail::shuffleAndGather<DataT, IndexT>(handle, in, out, n_samples_to_gather, seed, workspace);
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
void countSamplesInCluster(const raft::handle_t& handle,
                           const KMeansParams& params,
                           const raft::device_matrix_view<const DataT, IndexT>& X,
                           const raft::device_vector_view<DataT, IndexT>& L2NormX,
                           const raft::device_matrix_view<DataT, IndexT>& centroids,
                           rmm::device_uvector<char>& workspace,
                           const raft::device_vector_view<DataT, IndexT>& sampleCountInCluster)
{
  detail::countSamplesInCluster<DataT, IndexT>(
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
void kmeansPlusPlus(const raft::handle_t& handle,
                    const KMeansParams& params,
                    const raft::device_matrix_view<const DataT, IndexT>& X,
                    const raft::device_matrix_view<DataT, IndexT>& centroidsRawData,
                    rmm::device_uvector<char>& workspace)
{
  detail::kmeansPlusPlus<DataT, IndexT>(handle, params, X, centroidsRawData, workspace);
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
void kmeans_fit_main(const raft::handle_t& handle,
                     const KMeansParams& params,
                     const raft::device_matrix_view<const DataT, IndexT>& X,
                     const raft::device_vector_view<const DataT, IndexT>& weight,
                     const raft::device_matrix_view<DataT, IndexT>& centroidsRawData,
                     const raft::host_scalar_view<DataT>& inertia,
                     const raft::host_scalar_view<IndexT>& n_iter,
                     rmm::device_uvector<char>& workspace)
{
  detail::kmeans_fit_main<DataT, IndexT>(
    handle, params, X, weight, centroidsRawData, inertia, n_iter, workspace);
}
}  // namespace raft::cluster
