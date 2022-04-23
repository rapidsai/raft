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
#include <raft/cluster/kmeans_params.hpp>
#include <raft/core/mdarray.hpp>

namespace raft {
namespace cluster {

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IdxT the type of data used for indexing.
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 * be in row-major format
 * @param[in]     sample_weight Optional weights for each observation in X.
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 * centroids as the initial cluster centers
 *                              [out] Otherwise, generated centroids from the
 * kmeans algorithm is stored at the address pointed by 'centroids'.
 * @param[out]    inertia       Sum of squared distances of samples to their
 * closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
template <typename DataT, typename IndexT = int>
void kmeans_fit(handle_t const& handle,
                const KMeansParams& params,
                raft::device_matrix_view<const DataT> X,
                std::optional<raft::device_vector_view<const DataT>> sample_weight,
                raft::device_matrix_view<DataT> centroids,
                DataT& inertia,
                IndexT& n_iter)
{
  detail::kmeans_fit<DataT, IndexT>(handle, params, X, sample_weight, centroids, inertia, n_iter);
}

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @param[in]     handle            The handle to the cuML library context
 * that manages the CUDA resources.
 * @param[in]     params            Parameters for KMeans model.
 * @param[in]     centroids         Cluster centroids. The data must be in
 * row-major format.
 * @param[in]     X                 New data to predict.
 * @param[in]     sample_weight     The weights for each observation in X.
 * @param[in]     normalize_weight  True if the weights should be normalized
 * @param[out]    labels            Index of the cluster each sample in X
 * belongs to.
 * @param[out]    inertia           Sum of squared distances of samples to
 * their closest cluster center.
 */
template <typename DataT, typename IndexT = int>
void kmeans_predict(handle_t const& handle,
                    const KMeansParams& params,
                    raft::device_matrix_view<const DataT> X,
                    std::optional<raft::device_vector_view<const DataT>> sample_weight,
                    raft::device_matrix_view<const DataT> centroids,
                    raft::device_vector_view<IndexT> labels,
                    bool normalize_weight,
                    DataT& inertia)
{
  detail::kmeans_predict<DataT, IndexT>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}

/**
 * @brief Compute k-means clustering and predicts cluster index for each sample
 * in the input.
 *
 * @param[in]     handle        The handle to the cuML library context that
 * manages the CUDA resources.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must be
 * in row-major format
 * @param[in]     sample_weight The weights for each observation in X.
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 * centroids  as the initial cluster centers
 *                              [out] Otherwise, generated centroids from the
 * kmeans algorithm is stored at the address pointed by 'centroids'.
 * @param[out]    labels        Index of the cluster each sample in X belongs
 * to.
 * @param[out]    inertia       Sum of squared distances of samples to their
 * closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
template <typename DataT, typename IndexT = int>
void kmeans_fit_predict(handle_t const& handle,
                        const KMeansParams& params,
                        raft::device_matrix_view<const DataT> X,
                        std::optional<raft::device_vector_view<const DataT>> sample_weight,
                        raft::device_matrix_view<DataT> centroids,
                        raft::device_vector_view<IndexT> labels,
                        DataT& inertia,
                        IndexT& n_iter)
{
  kmeans_fit<DataT, IndexT>(handle, params, X, sample_weight, centroids, inertia, n_iter);
  kmeans_predict<DataT, IndexT>(handle, params, X, sample_weight, centroids, labels, true, inertia);
}

/**
 * @brief Transform X to a cluster-distance space.
 *
 * @param[in]     handle        The handle to the cuML library context that
 * manages the CUDA resources.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 * be in row-major format
 * @param[in]     centroids     Cluster centroids. The data must be in row-major format.
 * @param[out]    X_new         X transformed in the new space..
 */
template <typename DataT, typename IndexT = int>
void kmeans_transform(const raft::handle_t& handle,
                      const KMeansParams& params,
                      raft::device_matrix_view<const DataT> X,
                      raft::device_matrix_view<const DataT> centroids,
                      raft::device_matrix_view<DataT> X_new)
{
  detail::kmeans_transform<DataT, IndexT>(handle, params, X, centroids, X_new);
}
}  // namespace cluster
}  // namespace raft
