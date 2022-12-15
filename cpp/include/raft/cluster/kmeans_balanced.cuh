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
#include <raft/distance/distance_types.hpp>
#include <raft/util/cuda_utils.cuh>

namespace raft::cluster::kmeans_balanced {

// todo: wrap n_iter, metric, etc in parameter structure?
// todo: remove stream arg when there is a handle arg
// todo: C++-style casts
// todo: remove old interface and call this one instead
// todo: if mapping_op has same input and output types, is it assumed to be identity?
// todo: document this API
// todo: IdxT everywhere (except labels)
// todo: check IdxT and LabelT are big enough

template <typename DataT,
          typename MathT,
          typename IndexT,
          typename MappingOpT = raft::Nop<MathT, IndexT>>
void fit(handle_t const& handle,
         raft::device_matrix_view<const DataT, IndexT> X,
         raft::device_matrix_view<MathT, IndexT> centroids,
         uint32_t n_iter,
         raft::distance::DistanceType metric = raft::distance::DistanceType::L2Expanded,
         MappingOpT mapping_op               = raft::Nop<MathT, IndexT>())
{
  detail::build_hierarchical(handle,
                             n_iter,
                             X.extent(1),
                             X.data_handle(),
                             X.extent(0),
                             centroids.data_handle(),
                             centroids.extent(0),
                             metric,
                             mapping_op);
}

template <typename DataT,
          typename MathT,
          typename IndexT,
          typename LabelT,
          typename MappingOpT = raft::Nop<MathT, IndexT>>
void predict(handle_t const& handle,
             raft::device_matrix_view<const DataT, IndexT> X,
             raft::device_matrix_view<const MathT, IndexT> centroids,
             raft::device_vector_view<LabelT, IndexT> labels,
             raft::distance::DistanceType metric = raft::distance::DistanceType::L2Expanded,
             MappingOpT mapping_op               = raft::Nop<MathT, IndexT>())
{
  ASSERT(X.extent(0) == labels.extent(0), "Number of rows in dataset and labels are different");
  ASSERT(X.extent(1) == centroids.extent(1),
         "Number of features in dataset and centroids are different");

  detail::predict(handle,
                  centroids.data_handle(),
                  centroids.extent(0),
                  X.extent(1),
                  X.data_handle(),
                  X.extent(0),
                  labels.data_handle(),
                  metric,
                  mapping_op);
}

template <typename DataT,
          typename MathT,
          typename IndexT,
          typename LabelT,
          typename MappingOpT = raft::Nop<MathT, IndexT>>
void fit_predict(handle_t const& handle,
                 raft::device_matrix_view<const DataT, IndexT> X,
                 raft::device_matrix_view<MathT, IndexT> centroids,
                 raft::device_vector_view<LabelT, IndexT> labels,
                 uint32_t n_iter,
                 raft::distance::DistanceType metric = raft::distance::DistanceType::L2Expanded,
                 MappingOpT mapping_op               = raft::Nop<MathT, IndexT>())
{
  auto centroids_const = raft::make_device_matrix_view<const MathT, int>(
    centroids.data_handle(), centroids.extent(0), centroids.extent(1));
  raft::cluster::kmeans_balanced::fit(handle, X, centroids, n_iter, metric, mapping_op);
  raft::cluster::kmeans_balanced::predict(handle, X, centroids_const, labels, metric, mapping_op);
}

}  // namespace raft::cluster::kmeans_balanced
