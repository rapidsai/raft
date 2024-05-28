/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/neighbors/ann_mg_types.hpp>
#include <raft/neighbors/detail/ann_mg.cuh>

namespace raft::neighbors::mg {

template <typename T, typename IdxT>
auto build(const raft::resources& handle,
           const raft::neighbors::mg::nccl_clique& clique,
           const ivf_flat::mg_index_params& index_params,
           raft::host_matrix_view<const T, IdxT, row_major> index_dataset)
  -> detail::ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>
{
  return mg::detail::build<T, IdxT>(handle, clique, index_params, index_dataset);
}

template <typename T, typename IdxT>
void extend(const raft::resources& handle,
            const raft::neighbors::mg::nccl_clique& clique,
            detail::ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,
            raft::host_matrix_view<const T, IdxT, row_major> new_vectors,
            std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices)
{
  mg::detail::extend<T, IdxT>(handle, clique, index, new_vectors, new_indices);
}

template <typename T, typename IdxT>
void search(const raft::resources& handle,
            const raft::neighbors::mg::nccl_clique& clique,
            const detail::ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,
            const ivf_flat::search_params& search_params,
            raft::host_matrix_view<const T, IdxT, row_major> query_dataset,
            raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,
            raft::host_matrix_view<float, IdxT, row_major> distances,
            uint64_t n_rows_per_batch = 1 << 20) // 2^20
{
  mg::detail::search<T, IdxT>(handle, clique, index, search_params, query_dataset, neighbors, distances, n_rows_per_batch);
}

}  // namespace raft::neighbors::mg