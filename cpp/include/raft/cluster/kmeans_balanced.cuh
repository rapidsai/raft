/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/util/cuda_utils.cuh>

#include <utility>

namespace raft::cluster::kmeans_balanced {

/**
 * @brief Find clusters of balanced sizes with a hierarchical k-means algorithm.
 *
 * This variant of the k-means algorithm first clusters the dataset in mesoclusters, then clusters
 * the subsets associated to each mesocluster into fine clusters, and finally runs a few k-means
 * iterations over the whole dataset and with all the centroids to obtain the final clusters.
 *
 * Each k-means iteration applies expectation-maximization-balancing:
 *  - Balancing: adjust centers for clusters that have a small number of entries. If the size of a
 *    cluster is below a threshold, the center is moved towards a bigger cluster.
 *  - Expectation: predict the labels (i.e find closest cluster centroid to each point)
 *  - Maximization: calculate optimal centroids (i.e find the center of gravity of each cluster)
 *
 * The number of mesoclusters is chosen by rounding the square root of the number of clusters. E.g
 * for 512 clusters, we would have 23 mesoclusters. The number of fine clusters per mesocluster is
 * chosen proportionally to the number of points in each mesocluster.
 *
 * This variant of k-means uses random initialization and a fixed number of iterations, though
 * iterations can be repeated if the balancing step moved the centroids.
 *
 * Additionally, this algorithm supports quantized datasets in arbitrary types but the core part of
 * the algorithm will work with a floating-point type, hence a conversion function can be provided
 * to map the data type to the math type.
 *
 * @code{.cpp}
 *   #include <raft/core/handle.hpp>
 *   #include <raft/cluster/kmeans_balanced.cuh>
 *   #include <raft/cluster/kmeans_balanced_types.hpp>
 *   ...
 *   raft::handle_t handle;
 *   raft::cluster::kmeans_balanced_params params;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, n_clusters, n_features);
 *   raft::cluster::kmeans_balanced::fit(handle, params, X, centroids.view());
 * @endcode
 *
 * @tparam DataT Type of the input data.
 * @tparam MathT Type of the centroids and mapped data.
 * @tparam IndexT Type used for indexing.
 * @tparam MappingOpT Type of the mapping function.
 * @param[in]  handle     The raft resources
 * @param[in]  params     Structure containing the hyper-parameters
 * @param[in]  X          Training instances to cluster. The data must be in row-major format.
 *                        [dim = n_samples x n_features]
 * @param[out] centroids  The generated centroids [dim = n_clusters x n_features]
 * @param[in]  mapping_op (optional) Functor to convert from the input datatype to the arithmetic
 *                        datatype. If DataT == MathT, this must be the identity.
 */
template <typename DataT, typename MathT, typename IndexT, typename MappingOpT = raft::identity_op>
void fit(const raft::resources& handle,
         kmeans_balanced_params const& params,
         raft::device_matrix_view<const DataT, IndexT> X,
         raft::device_matrix_view<MathT, IndexT> centroids,
         MappingOpT mapping_op = raft::identity_op())
{
  RAFT_EXPECTS(X.extent(1) == centroids.extent(1),
               "Number of features in dataset and centroids are different");
  RAFT_EXPECTS(static_cast<uint64_t>(X.extent(0)) * static_cast<uint64_t>(X.extent(1)) <=
                 static_cast<uint64_t>(std::numeric_limits<IndexT>::max()),
               "The chosen index type cannot represent all indices for the given dataset");
  RAFT_EXPECTS(centroids.extent(0) > IndexT{0} && centroids.extent(0) <= X.extent(0),
               "The number of centroids must be strictly positive and cannot exceed the number of "
               "points in the training dataset.");

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
 *   raft::cluster::kmeans_balanced_params params;
 *   auto labels = raft::make_device_vector<float, int>(handle, n_rows);
 *   raft::cluster::kmeans_balanced::predict(handle, params, X, centroids, labels);
 * @endcode
 *
 * @tparam DataT Type of the input data.
 * @tparam MathT Type of the centroids and mapped data.
 * @tparam IndexT Type used for indexing.
 * @tparam LabelT Type of the output labels.
 * @tparam MappingOpT Type of the mapping function.
 * @param[in]  handle     The raft resources
 * @param[in]  params     Structure containing the hyper-parameters
 * @param[in]  X          Dataset for which to infer the closest clusters.
 *                        [dim = n_samples x n_features]
 * @param[in]  centroids  The input centroids [dim = n_clusters x n_features]
 * @param[out] labels     The output labels [dim = n_samples]
 * @param[in]  mapping_op (optional) Functor to convert from the input datatype to the arithmetic
 *                        datatype. If DataT == MathT, this must be the identity.
 */
template <typename DataT,
          typename MathT,
          typename IndexT,
          typename LabelT,
          typename MappingOpT = raft::identity_op>
void predict(const raft::resources& handle,
             kmeans_balanced_params const& params,
             raft::device_matrix_view<const DataT, IndexT> X,
             raft::device_matrix_view<const MathT, IndexT> centroids,
             raft::device_vector_view<LabelT, IndexT> labels,
             MappingOpT mapping_op = raft::identity_op())
{
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
 * @brief Compute hierarchical balanced k-means clustering and predict cluster index for each sample
 * in the input.
 *
 * @code{.cpp}
 *   #include <raft/core/handle.hpp>
 *   #include <raft/cluster/kmeans_balanced.cuh>
 *   #include <raft/cluster/kmeans_balanced_types.hpp>
 *   ...
 *   raft::handle_t handle;
 *   raft::cluster::kmeans_balanced_params params;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, n_clusters, n_features);
 *   auto labels = raft::make_device_vector<float, int>(handle, n_rows);
 *   raft::cluster::kmeans_balanced::fit_predict(
 *       handle, params, X, centroids.view(), labels.view());
 * @endcode
 *
 * @tparam DataT Type of the input data.
 * @tparam MathT Type of the centroids and mapped data.
 * @tparam IndexT Type used for indexing.
 * @tparam LabelT Type of the output labels.
 * @tparam MappingOpT Type of the mapping function.
 * @param[in]  handle     The raft resources
 * @param[in]  params     Structure containing the hyper-parameters
 * @param[in]  X          Training instances to cluster. The data must be in row-major format.
 *                        [dim = n_samples x n_features]
 * @param[out] centroids  The output centroids [dim = n_clusters x n_features]
 * @param[out] labels     The output labels [dim = n_samples]
 * @param[in]  mapping_op (optional) Functor to convert from the input datatype to the arithmetic
 *                        datatype. If DataT and MathT are the same, this must be the identity.
 */
template <typename DataT,
          typename MathT,
          typename IndexT,
          typename LabelT,
          typename MappingOpT = raft::identity_op>
void fit_predict(const raft::resources& handle,
                 kmeans_balanced_params const& params,
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

namespace helpers {

/**
 * @brief Randomly initialize centers and apply expectation-maximization-balancing iterations
 *
 * This is essentially the non-hierarchical balanced k-means algorithm which is used by the
 * hierarchical algorithm once to build the mesoclusters and once per mesocluster to build the fine
 * clusters.
 *
 * @code{.cpp}
 *   #include <raft/core/handle.hpp>
 *   #include <raft/cluster/kmeans_balanced.cuh>
 *   #include <raft/cluster/kmeans_balanced_types.hpp>
 *   ...
 *   raft::handle_t handle;
 *   raft::cluster::kmeans_balanced_params params;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, n_clusters, n_features);
 *   auto labels = raft::make_device_vector<int, int>(handle, n_samples);
 *   auto sizes = raft::make_device_vector<int, int>(handle, n_clusters);
 *   raft::cluster::kmeans_balanced::build_clusters(
 *       handle, params, X, centroids.view(), labels.view(), sizes.view());
 * @endcode
 *
 * @tparam DataT Type of the input data.
 * @tparam MathT Type of the centroids and mapped data.
 * @tparam IndexT Type used for indexing.
 * @tparam LabelT Type of the output labels.
 * @tparam CounterT Counter type supported by CUDA's native atomicAdd.
 * @tparam MappingOpT Type of the mapping function.
 * @param[in]  handle        The raft resources
 * @param[in]  params        Structure containing the hyper-parameters
 * @param[in]  X             Training instances to cluster. The data must be in row-major format.
 *                           [dim = n_samples x n_features]
 * @param[out] centroids     The output centroids [dim = n_clusters x n_features]
 * @param[out] labels        The output labels [dim = n_samples]
 * @param[out] cluster_sizes Size of each cluster [dim = n_clusters]
 * @param[in]  mapping_op    (optional) Functor to convert from the input datatype to the
 *                           arithmetic datatype. If DataT == MathT, this must be the identity.
 * @param[in]  X_norm        (optional) Dataset's row norms [dim = n_samples]
 */
template <typename DataT,
          typename MathT,
          typename IndexT,
          typename LabelT,
          typename CounterT,
          typename MappingOpT>
void build_clusters(const raft::resources& handle,
                    const kmeans_balanced_params& params,
                    raft::device_matrix_view<const DataT, IndexT> X,
                    raft::device_matrix_view<MathT, IndexT> centroids,
                    raft::device_vector_view<LabelT, IndexT> labels,
                    raft::device_vector_view<CounterT, IndexT> cluster_sizes,
                    MappingOpT mapping_op = raft::identity_op(),
                    std::optional<raft::device_vector_view<const MathT>> X_norm = std::nullopt)
{
  RAFT_EXPECTS(X.extent(0) == labels.extent(0),
               "Number of rows in dataset and labels are different");
  RAFT_EXPECTS(X.extent(1) == centroids.extent(1),
               "Number of features in dataset and centroids are different");
  RAFT_EXPECTS(centroids.extent(0) == cluster_sizes.extent(0),
               "Number of rows in centroids and clusyer_sizes are different");

  detail::build_clusters(handle,
                         params,
                         X.extent(1),
                         X.data_handle(),
                         X.extent(0),
                         centroids.extent(0),
                         centroids.data_handle(),
                         labels.data_handle(),
                         cluster_sizes.data_handle(),
                         mapping_op,
                         resource::get_workspace_resource(handle),
                         X_norm.has_value() ? X_norm.value().data_handle() : nullptr);
}

/**
 * @brief Given the data and labels, calculate cluster centers and sizes in one sweep.
 *
 * Let `S_i = {x_k | x_k \in X & labels[k] == i}` be the vectors in the dataset with label i.
 *
 * On exit,
 *   `centers_i = (\sum_{x \in S_i} x + w_i * center_i) / (|S_i| + w_i)`,
 *     where  `w_i = reset_counters ?  0 : cluster_size[i]`.
 *
 * In other words, the updated cluster centers are a weighted average of the existing cluster
 * center, and the coordinates of the points labeled with i. _This allows calling this function
 * multiple times with different datasets with the same effect as if calling this function once
 * on the combined dataset_.
 *
 * @code{.cpp}
 *   #include <raft/core/handle.hpp>
 *   #include <raft/cluster/kmeans_balanced.cuh>
 *   ...
 *   raft::handle_t handle;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, n_clusters, n_features);
 *   auto sizes = raft::make_device_vector<int, int>(handle, n_clusters);
 *   raft::cluster::kmeans_balanced::calc_centers_and_sizes(
 *       handle, X, labels, centroids.view(), sizes.view(), true);
 * @endcode
 *
 * @tparam DataT Type of the input data.
 * @tparam MathT Type of the centroids and mapped data.
 * @tparam IndexT Type used for indexing.
 * @tparam LabelT Type of the output labels.
 * @tparam CounterT Counter type supported by CUDA's native atomicAdd.
 * @tparam MappingOpT Type of the mapping function.
 * @param[in]  handle         The raft resources
 * @param[in]  X              Dataset for which to calculate cluster centers. The data must be in
 *                            row-major format. [dim = n_samples x n_features]
 * @param[in]  labels         The input labels [dim = n_samples]
 * @param[out] centroids      The output centroids [dim = n_clusters x n_features]
 * @param[out] cluster_sizes  Size of each cluster [dim = n_clusters]
 * @param[in]  reset_counters Whether to clear the output arrays before calculating.
 *                            When set to `false`, this function may be used to update existing
 *                            centers and sizes using the weighted average principle.
 * @param[in]  mapping_op     (optional) Functor to convert from the input datatype to the
 *                            arithmetic datatype. If DataT == MathT, this must be the identity.
 */
template <typename DataT,
          typename MathT,
          typename IndexT,
          typename LabelT,
          typename CounterT,
          typename MappingOpT = raft::identity_op>
void calc_centers_and_sizes(const raft::resources& handle,
                            raft::device_matrix_view<const DataT, IndexT> X,
                            raft::device_vector_view<const LabelT, IndexT> labels,
                            raft::device_matrix_view<MathT, IndexT> centroids,
                            raft::device_vector_view<CounterT, IndexT> cluster_sizes,
                            bool reset_counters   = true,
                            MappingOpT mapping_op = raft::identity_op())
{
  RAFT_EXPECTS(X.extent(0) == labels.extent(0),
               "Number of rows in dataset and labels are different");
  RAFT_EXPECTS(X.extent(1) == centroids.extent(1),
               "Number of features in dataset and centroids are different");
  RAFT_EXPECTS(centroids.extent(0) == cluster_sizes.extent(0),
               "Number of rows in centroids and clusyer_sizes are different");

  detail::calc_centers_and_sizes(handle,
                                 centroids.data_handle(),
                                 cluster_sizes.data_handle(),
                                 centroids.extent(0),
                                 X.extent(1),
                                 X.data_handle(),
                                 X.extent(0),
                                 labels.data_handle(),
                                 reset_counters,
                                 mapping_op,
                                 resource::get_workspace_resource(handle));
}

}  // namespace helpers

}  // namespace raft::cluster::kmeans_balanced
