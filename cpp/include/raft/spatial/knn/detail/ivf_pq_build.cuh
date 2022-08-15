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
                   const index<T, IdxT>& orig_index,
                   const T* new_vectors,
                   const IdxT* new_indices,
                   IdxT n_rows) -> index<T, IdxT>
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::extend(%zu, %u)", size_t(n_rows), orig_index.dim());

  if (new_indices != nullptr) {
    RAFT_LOG_WARN("Index input is ignored at the moment (non-null new_indices given).");
  }

  ivf_pq::index<T, IdxT> new_index(
    handle, orig_index.metric(), orig_index.n_lists(), orig_index.dim(), orig_index.pq_dim());
  new_index.desc() = ivf_pq::detail::cuannIvfPqCreateNewIndexByAddingVectorsToOldIndex(
    handle, const_cast<cuannIvfPqDescriptor_t&>(orig_index.desc()), new_vectors, n_rows);

  return new_index;
}

/** See raft::spatial::knn::ivf_pq::build docs */
template <typename T, typename IdxT>
inline auto build(
  const handle_t& handle, const index_params& params, const T* dataset, IdxT n_rows, uint32_t dim)
  -> index<T, IdxT>
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "unsupported data type");
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");

  ivf_pq::index<T, IdxT> index(handle, params.metric, params.n_lists, dim, params.pq_dim);

  ivf_pq::detail::cuannIvfPqSetIndexParameters(
    index.desc(),
    index.n_lists(),  /* Number of clusters */
    (uint32_t)n_rows, /* Number of dataset entries */
    index.dim(),      /* Dimension of each entry */
    index.pq_dim(),   /* Dimension of each entry after product quantization */
    params.pq_bits,   /* Bit length of PQ */
    index.metric(),
    params.codebook_kind);

  // Build index
  ivf_pq::detail::cuannIvfPqBuildIndex(
    handle,
    index.desc(),
    dataset,                                          // dataset
    dataset,                                          // ?kmeans? trainset
    uint32_t(params.add_data_on_build ? n_rows : 0),  // size of the trainset (I guess for kmeans)
    params.kmeans_n_iters,
    params.random_rotation,
    true  // hierarchialClustering: always true in raft
  );

  return index;
}

}  // namespace raft::spatial::knn::ivf_pq::detail
