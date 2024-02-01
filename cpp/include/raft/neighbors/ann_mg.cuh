/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <raft/neighbors/detail/ann_mg.cuh>

namespace raft::neighbors::mg {

template <typename T, typename IdxT>
auto build(const std::vector<int> device_ids,
           raft::neighbors::mg::dist_mode mode,
           const ivf_flat::index_params& index_params,
           raft::host_matrix_view<const T, IdxT, row_major> index_dataset)
  -> detail::ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>
{
  return mg::detail::build<T, IdxT>(device_ids, mode, index_params, index_dataset);
}

template <typename T, typename IdxT>
auto build(const std::vector<int> device_ids,
           raft::neighbors::mg::dist_mode mode,
           const ivf_pq::index_params& index_params,
           raft::host_matrix_view<const T, IdxT, row_major> index_dataset)
  -> detail::ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>
{
  return mg::detail::build<T>(device_ids, mode, index_params, index_dataset);
}

template <typename T, typename IdxT>
auto build(const std::vector<int> device_ids,
           raft::neighbors::mg::dist_mode mode,
           const cagra::index_params& index_params,
           raft::host_matrix_view<const T, IdxT, row_major> index_dataset)
  -> detail::ann_mg_index<cagra::index<T, IdxT>, T, IdxT>
{
  return mg::detail::build<T, IdxT>(device_ids, mode, index_params, index_dataset);
}

template <typename T, typename IdxT>
void extend(detail::ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,
            raft::host_matrix_view<const T, IdxT, row_major> new_vectors,
            std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices)
{
  mg::detail::extend<T, IdxT>(index, new_vectors, new_indices);
}

template <typename T, typename IdxT>
void extend(detail::ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,
            raft::host_matrix_view<const T, IdxT, row_major> new_vectors,
            std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices)
{
  mg::detail::extend<T>(index, new_vectors, new_indices);
}

template <typename T, typename IdxT>
void search(const detail::ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,
            const ivf_flat::search_params& search_params,
            raft::host_matrix_view<const T, IdxT, row_major> query_dataset,
            raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,
            raft::host_matrix_view<float, IdxT, row_major> distances)
{
  mg::detail::search<T, IdxT>(index, search_params, query_dataset, neighbors, distances);
}

template <typename T, typename IdxT>
void search(const detail::ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,
            const ivf_pq::search_params& search_params,
            raft::host_matrix_view<const T, IdxT, row_major> query_dataset,
            raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,
            raft::host_matrix_view<float, IdxT, row_major> distances)
{
  mg::detail::search<T>(index, search_params, query_dataset, neighbors, distances);
}

template <typename T, typename IdxT>
void search(const detail::ann_mg_index<cagra::index<T, IdxT>, T, IdxT>& index,
            const cagra::search_params& search_params,
            raft::host_matrix_view<const T, IdxT, row_major> query_dataset,
            raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,
            raft::host_matrix_view<float, IdxT, row_major> distances)
{
  mg::detail::search<T, IdxT>(index, search_params, query_dataset, neighbors, distances);
}

template <typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const detail::ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,
               const std::string& filename)
{
  mg::detail::serialize(handle, index, filename);
}

template <typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const detail::ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,
               const std::string& filename)
{
  mg::detail::serialize(handle, index, filename);
}

template <typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const detail::ann_mg_index<cagra::index<T, IdxT>, T, IdxT>& index,
               const std::string& filename)
{
  mg::detail::serialize(handle, index, filename);
}

template <typename T, typename IdxT>
detail::ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT> deserialize_flat(const raft::resources& handle,
                                                                         const std::string& filename)
{
  return mg::detail::deserialize_flat<T, IdxT>(handle, filename);
}

template <typename T, typename IdxT>
detail::ann_mg_index<ivf_pq::index<IdxT>, T, IdxT> deserialize_pq(const raft::resources& handle,
                                                                  const std::string& filename)
{
  return mg::detail::deserialize_pq<T, IdxT>(handle, filename);
}

template <typename T, typename IdxT>
detail::ann_mg_index<cagra::index<T, IdxT>, T, IdxT> deserialize_cagra(const raft::resources& handle,
                                                                       const std::string& filename)
{
  return mg::detail::deserialize_cagra<T, IdxT>(handle, filename);
}
}  // namespace raft::neighbors::mg