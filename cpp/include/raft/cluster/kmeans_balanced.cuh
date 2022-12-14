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
// todo: proper double support: MathT != DataT

template <typename DataT, typename IndexT, typename MappingOpT = raft::Nop<DataT, IndexT>>
void fit(handle_t const& handle,
         raft::device_matrix_view<const DataT, IndexT> X,
         raft::device_matrix_view<DataT, IndexT> centroids,
         uint32_t n_iter,
         raft::distance::DistanceType metric = raft::distance::DistanceType::L2Expanded,
         MappingOpT mapping_op               = raft::Nop<DataT, IndexT>())
{
  detail::build_hierarchical(handle,
                             n_iter,
                             X.extent(1),
                             X.data_handle(),
                             X.extent(0),
                             centroids.data_handle(),
                             centroids.extent(0),
                             metric,
                             mapping_op,
                             handle.get_stream());
}

}  // namespace raft::cluster::kmeans_balanced
