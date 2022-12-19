/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/cluster/detail/kmeans_balanced.cuh>
#include <raft/core/mdarray.hpp>
#include <raft/util/cuda_utils.cuh>

namespace raft::cluster::kmeans_balanced {

// todo: remove old interface and call this one instead

/**
 * @brief Find clusters of balanced sizes with a hierarchical k-means algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/handle.hpp>
 *   #include <raft/cluster/kmeans_balanced.cuh>
 *   #include <raft/cluster/kmeans_balanced_types.hpp>
 *   ...
 *   raft::handle_t handle;
 *   raft::cluster::KMeansBalancedParams params;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, n_clusters, n_features);
 *   raft::cluster::kmeans_balanced::fit(handle, params, X, centroids);
 * @endcode
 *
 * @tparam DataT Type of the input data.
 * @tparam MathT Type of the centroids and mapped data.
 * @tparam IndexT Type used for indexing.
 * @tparam MappingOpT Type of the mapping function.
 * @param[in]  handle     The raft handle
 * @param[in]  params     Structure containing the hyper-parameters
 * @param[in]  X          Training instances to cluster. The data must be in row-major format.
 *                        [dim = n_samples x n_features]
 * @param[out] centroids  The generated centroids [dim = n_clusters x n_features]
 * @param[in]  mapping_op (optional) Functor to convert from the input datatype to the arithmetic
 *                        datatype. If DataT and MathT are the same, this must be the identity.
 */
template <typename DataT, typename MathT, typename IndexT, typename MappingOpT = raft::identity_op>
void fit(handle_t const& handle,
         KMeansBalancedParams const& params,
         raft::device_matrix_view<const DataT, IndexT> X,
         raft::device_matrix_view<MathT, IndexT> centroids,
         MappingOpT mapping_op = raft::identity_op())
{
  logger::get(RAFT_NAME).set_level(params.verbosity);
  RAFT_EXPECTS(X.extent(1) == centroids.extent(1),
               "Number of features in dataset and centroids are different");
  RAFT_EXPECTS(static_cast<uint64_t>(X.extent(0)) * static_cast<uint64_t>(X.extent(1)) <=
                 static_cast<uint64_t>(std::numeric_limits<IndexT>::max()),
               "The chosen index type cannot represent all indices for the given dataset");

  detail::build_hierarchical(handle,
                             params,
                             X.extent(1),
                             X.data_handle(),
                             X.extent(0),
                             centroids.data_handle(),
                             centroids.extent(0),
                             mapping_op);
}

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/handle.hpp>
 *   #include <raft/cluster/kmeans_balanced.cuh>
 *   #include <raft/cluster/kmeans_balanced_types.hpp>
 *   ...
 *   raft::handle_t handle;
 *   raft::cluster::KMeansBalancedParams params;
 *   auto labels = raft::make_device_vector<float, int>(handle, n_rows);
 *   raft::cluster::kmeans_balanced::fit(handle, params, X, centroids, labels);
 * @endcode
 *
 * @tparam DataT Type of the input data.
 * @tparam MathT Type of the centroids and mapped data.
 * @tparam IndexT Type used for indexing.
 * @tparam LabelT Type of the output labels.
 * @tparam MappingOpT Type of the mapping function.
 * @param[in]  handle     The raft handle
 * @param[in]  params     Structure containing the hyper-parameters
 * @param[in]  X          Training instances to cluster. The data must be in row-major format.
 *                        [dim = n_samples x n_features]
 * @param[in]  centroids  The input centroids [dim = n_clusters x n_features]
 * @param[out] labels     The output labels [dim = n_rows]
 * @param[in]  mapping_op (optional) Functor to convert from the input datatype to the arithmetic
 *                        datatype. If DataT and MathT are the same, this must be the identity.
 */
template <typename DataT,
          typename MathT,
          typename IndexT,
          typename LabelT,
          typename MappingOpT = raft::identity_op>
void predict(handle_t const& handle,
             KMeansBalancedParams const& params,
             raft::device_matrix_view<const DataT, IndexT> X,
             raft::device_matrix_view<const MathT, IndexT> centroids,
             raft::device_vector_view<LabelT, IndexT> labels,
             MappingOpT mapping_op = raft::identity_op())
{
  logger::get(RAFT_NAME).set_level(params.verbosity);
  RAFT_EXPECTS(X.extent(0) == labels.extent(0),
               "Number of rows in dataset and labels are different");
  RAFT_EXPECTS(X.extent(1) == centroids.extent(1),
               "Number of features in dataset and centroids are different");
  RAFT_EXPECTS(static_cast<uint64_t>(X.extent(0)) * static_cast<uint64_t>(X.extent(1)) <=
                 static_cast<uint64_t>(std::numeric_limits<IndexT>::max()),
               "The chosen index type cannot represent all indices for the given dataset");
  RAFT_EXPECTS(static_cast<uint64_t>(centroids.extent(0)) <=
                 static_cast<uint64_t>(std::numeric_limits<LabelT>::max()),
               "The chosen label type cannot represent all cluster labels");

  detail::predict(handle,
                  params,
                  centroids.data_handle(),
                  centroids.extent(0),
                  X.extent(1),
                  X.data_handle(),
                  X.extent(0),
                  labels.data_handle(),
                  mapping_op);
}

/**
 * @brief Compute k-means clustering and predict cluster index for each sample in the input.
 *
 * @code{.cpp}
 *   #include <raft/core/handle.hpp>
 *   #include <raft/cluster/kmeans_balanced.cuh>
 *   #include <raft/cluster/kmeans_balanced_types.hpp>
 *   ...
 *   raft::handle_t handle;
 *   raft::cluster::KMeansBalancedParams params;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, n_clusters, n_features);
 *   auto labels = raft::make_device_vector<float, int>(handle, n_rows);
 *   raft::cluster::kmeans_balanced::fit_predict(handle, params, X, centroids, labels);
 * @endcode
 *
 * @tparam DataT Type of the input data.
 * @tparam MathT Type of the centroids and mapped data.
 * @tparam IndexT Type used for indexing.
 * @tparam LabelT Type of the output labels.
 * @tparam MappingOpT Type of the mapping function.
 * @param[in]  handle     The raft handle
 * @param[in]  params     Structure containing the hyper-parameters
 * @param[in]  X          Training instances to cluster. The data must be in row-major format.
 *                        [dim = n_samples x n_features]
 * @param[in]  centroids  The input centroids [dim = n_clusters x n_features]
 * @param[out] labels     The output labels [dim = n_rows]
 * @param[in]  mapping_op (optional) Functor to convert from the input datatype to the arithmetic
 *                        datatype. If DataT and MathT are the same, this must be the identity.
 */
template <typename DataT,
          typename MathT,
          typename IndexT,
          typename LabelT,
          typename MappingOpT = raft::identity_op>
void fit_predict(handle_t const& handle,
                 KMeansBalancedParams const& params,
                 raft::device_matrix_view<const DataT, IndexT> X,
                 raft::device_matrix_view<MathT, IndexT> centroids,
                 raft::device_vector_view<LabelT, IndexT> labels,
                 MappingOpT mapping_op = raft::identity_op())
{
  auto centroids_const = raft::make_device_matrix_view<const MathT, IndexT>(
    centroids.data_handle(), centroids.extent(0), centroids.extent(1));
  raft::cluster::kmeans_balanced::fit(handle, params, X, centroids, mapping_op);
  raft::cluster::kmeans_balanced::predict(handle, params, X, centroids_const, labels, mapping_op);
}

}  // namespace raft::cluster::kmeans_balanced
