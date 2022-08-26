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

#include "../ivf_pq_types.hpp"
#include "ann_kmeans_balanced.cuh"
#include "ann_utils.cuh"
#include "ivf_pq_legacy.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/pow2_utils.cuh>

#include <rmm/cuda_stream_view.hpp>

namespace raft::spatial::knn::ivf_pq::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

/** See raft::spatial::knn::ivf_pq::extend docs */
template <typename T, typename IdxT>
inline auto extend(const handle_t& handle,
                   const index<IdxT>& orig_index,
                   const T* new_vectors,
                   const IdxT* new_indices,
                   IdxT n_rows) -> index<IdxT>
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::extend(%zu, %u)", size_t(n_rows), orig_index.dim());

  if (new_indices != nullptr) {
    RAFT_LOG_WARN("Index input is ignored at the moment (non-null new_indices given).");
  }

  return ivf_pq::detail::cuannIvfPqCreateNewIndexByAddingVectorsToOldIndex(
    handle, const_cast<index<IdxT>&>(orig_index), new_vectors, n_rows);
}

/** See raft::spatial::knn::ivf_pq::build docs */
template <typename T, typename IdxT>
inline auto build(
  const handle_t& handle, const index_params& params, const T* dataset, IdxT n_rows, uint32_t dim)
  -> index<IdxT>
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "unsupported data type");
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");

  ivf_pq::index<IdxT> index(handle, params, dim);
  index.desc().numDataset = n_rows;

  // Build index
  ivf_pq::detail::cuannIvfPqBuildIndex(handle,
                                       index,
                                       dataset,
                                       params.kmeans_trainset_fraction,
                                       params.kmeans_n_iters,
                                       params.random_rotation);

  return index;
}

}  // namespace raft::spatial::knn::ivf_pq::detail
