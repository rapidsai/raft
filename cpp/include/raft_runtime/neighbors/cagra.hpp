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

#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <string>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>

namespace raft::runtime::neighbors::experimental::cagra {

// Using device and host_matrix_view avoids needing to typedef mutltiple mdspans based on accessors
#define RAFT_INST_CAGRA_FUNCS(T, IdxT)                                                       \
  auto build(raft::resources const& handle,                                                  \
             const raft::neighbors::experimental::cagra::index_params& params,               \
             raft::device_matrix_view<const T, IdxT, row_major> dataset)                     \
    ->raft::neighbors::experimental::cagra::index<T, IdxT>;                                  \
                                                                                             \
  auto build(raft::resources const& handle,                                                  \
             const raft::neighbors::experimental::cagra::index_params& params,               \
             raft::host_matrix_view<const T, IdxT, row_major> dataset)                       \
    ->raft::neighbors::experimental::cagra::index<T, IdxT>;                                  \
                                                                                             \
  void build(raft::resources const& handle,                                                  \
             const raft::neighbors::experimental::cagra::index_params& params,               \
             raft::device_matrix_view<const T, IdxT, row_major> dataset,                     \
             raft::neighbors::experimental::cagra::index<T, IdxT>& idx);                     \
                                                                                             \
  void build(raft::resources const& handle,                                                  \
             const raft::neighbors::experimental::cagra::index_params& params,               \
             raft::host_matrix_view<const T, IdxT, row_major> dataset,                       \
             raft::neighbors::experimental::cagra::index<T, IdxT>& idx);                     \
                                                                                             \
  void build_knn_graph(raft::resources const& handle,                                        \
                       raft::device_matrix_view<const T, IdxT, row_major> dataset,           \
                       raft::host_matrix_view<IdxT, IdxT, row_major> knn_graph,              \
                       std::optional<float> refine_rate,                                     \
                       std::optional<raft::neighbors::ivf_pq::index_params> build_params,    \
                       std::optional<raft::neighbors::ivf_pq::search_params> search_params); \
                                                                                             \
  void build_knn_graph(raft::resources const& handle,                                        \
                       raft::host_matrix_view<const T, IdxT, row_major> dataset,             \
                       raft::host_matrix_view<IdxT, IdxT, row_major> knn_graph,              \
                       std::optional<float> refine_rate,                                     \
                       std::optional<raft::neighbors::ivf_pq::index_params> build_params,    \
                       std::optional<raft::neighbors::ivf_pq::search_params> search_params); \
                                                                                             \
  void sort_knn_graph(raft::resources const& handle,                                         \
                      raft::device_matrix_view<const T, IdxT, row_major> dataset,            \
                      raft::host_matrix_view<IdxT, IdxT, row_major> knn_graph);              \
                                                                                             \
  void sort_knn_graph(raft::resources const& handle,                                         \
                      raft::host_matrix_view<const T, IdxT, row_major> dataset,              \
                      raft::host_matrix_view<IdxT, IdxT, row_major> knn_graph);              \
                                                                                             \
  void search(raft::resources const& handle,                                                 \
              raft::neighbors::experimental::cagra::search_params const& params,             \
              const raft::neighbors::experimental::cagra::index<T, IdxT>& index,             \
              raft::device_matrix_view<const T, IdxT, row_major> queries,                    \
              raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,                     \
              raft::device_matrix_view<float, IdxT, row_major> distances);                   \
  void serialize_file(raft::resources const& handle,                                         \
                      const std::string& filename,                                           \
                      const raft::neighbors::experimental::cagra::index<T, IdxT>& index);    \
                                                                                             \
  void deserialize_file(raft::resources const& handle,                                       \
                        const std::string& filename,                                         \
                        raft::neighbors::experimental::cagra::index<T, IdxT>* index);        \
  void serialize(raft::resources const& handle,                                              \
                 std::string& str,                                                           \
                 const raft::neighbors::experimental::cagra::index<T, IdxT>& index);         \
                                                                                             \
  void deserialize(raft::resources const& handle,                                            \
                   const std::string& str,                                                   \
                   raft::neighbors::experimental::cagra::index<T, IdxT>* index);

RAFT_INST_CAGRA_FUNCS(float, uint32_t);
RAFT_INST_CAGRA_FUNCS(int8_t, uint32_t);
RAFT_INST_CAGRA_FUNCS(uint8_t, uint32_t);

#undef RAFT_INST_CAGRA_FUNCS

#define RAFT_INST_CAGRA_OPTIMIZE(IdxT)                                   \
  void optimize(raft::resources const& res,                              \
                raft::host_matrix_view<IdxT, IdxT, row_major> knn_graph, \
                raft::host_matrix_view<IdxT, IdxT, row_major> new_graph);

RAFT_INST_CAGRA_OPTIMIZE(uint32_t);

#undef RAFT_INST_CAGRA_OPTIMIZE

}  // namespace raft::runtime::neighbors::experimental::cagra
