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
// todo: document this API

template <typename DataT, typename MathT, typename IndexT, typename MappingOpT = raft::identity_op>
void fit(handle_t const& handle,
         KMeansBalancedParams const& params,
         raft::device_matrix_view<const DataT, IndexT> X,
         raft::device_matrix_view<MathT, IndexT> centroids,
         MappingOpT mapping_op = raft::identity_op())
{
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
