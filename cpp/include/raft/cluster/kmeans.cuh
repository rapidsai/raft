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

#include <raft/cluster/kmeans_params.hpp>
#include <raft/cluster/detail/kmeans.cuh>
#include <optional>
#include <raft/mdarray.hpp>

namespace raft {
namespace cluster {

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IdxT the type of data used for indexing.
 * @tparam layout the layout of the data (row or column).
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. It must be noted
 * that the data must be in row-major format and stored in device accessible
 * location.
 * @param[in]     n_samples     Number of samples in the input X.
 * @param[in]     n_features    Number of features or the dimensions of each
 * sample.
 * @param[in]     sample_weight Optional weights for each observation in X.
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 * centroids as the initial cluster centers
 *                              [out] Otherwise, generated centroids from the
 * kmeans algorithm is stored at the address pointed by 'centroids'.
 * @param[out]    inertia       Sum of squared distances of samples to their
 * closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
template <typename DataT, typename IndexT, typename layout>
void kmeans_fit(handle_t const& handle,
                const KMeansParams& params,
                const raft::device_matrix_view<DataT, layout> X,
                const std::optional<raft::device_vector_view<DataT>>& sample_weight,
                std::optional<raft::device_matrix_view<DataT, layout>>& centroids,
                DataT& inertia,
                IndexT& n_iter)
{
  detail::kmeans_fit<DataT, IndexT, layout>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

template <typename DataT, typename IndexT, typename layout>
void kmeans_predict(handle_t const& handle,
                const KMeansParams& params,
                const raft::device_matrix_view<DataT, layout> X,
                const std::optional<raft::device_vector_view<DataT>>& sample_weight,
                raft::device_matrix_view<DataT, layout> centroids,
                raft::device_vector_view<IndexT> labels,
                bool normalize_weight,
                DataT& inertia)
{
  detail::kmeans_predict<DataT, IndexT, layout>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}

template <typename DataT, typename IndexT, typename layout>
void kmeans_fit_predict(handle_t const& handle,
                const KMeansParams& params,
                const raft::device_matrix_view<DataT, layout> X,
                const std::optional<raft::device_vector_view<DataT>>& sample_weight,
                std::optional<raft::device_matrix_view<DataT, layout>>& centroids,
                raft::device_vector_view<IndexT> labels,
                DataT& inertia,
                IndexT& n_iter)
{
  kmeans_fit<DataT, IndexT, layout>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
  kmeans_predict<DataT, IndexT, layout>(
    handle, params, X, sample_weight, centroids.value(), labels, true, inertia);
}
}  // namespace cluster
}  // namespace raft
