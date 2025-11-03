/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/ivf_pq_types.hpp>

#include <string>

namespace raft::runtime::neighbors::cagra {

// Using device and host_matrix_view avoids needing to typedef mutltiple mdspans based on accessors
#define RAFT_INST_CAGRA_FUNCS(T, IdxT)                                                 \
  auto build(raft::resources const& handle,                                            \
             const raft::neighbors::cagra::index_params& params,                       \
             raft::device_matrix_view<const T, int64_t, row_major> dataset)            \
    -> raft::neighbors::cagra::index<T, IdxT>;                                         \
                                                                                       \
  auto build(raft::resources const& handle,                                            \
             const raft::neighbors::cagra::index_params& params,                       \
             raft::host_matrix_view<const T, int64_t, row_major> dataset)              \
    -> raft::neighbors::cagra::index<T, IdxT>;                                         \
                                                                                       \
  void build_device(raft::resources const& handle,                                     \
                    const raft::neighbors::cagra::index_params& params,                \
                    raft::device_matrix_view<const T, int64_t, row_major> dataset,     \
                    raft::neighbors::cagra::index<T, IdxT>& idx);                      \
                                                                                       \
  void build_host(raft::resources const& handle,                                       \
                  const raft::neighbors::cagra::index_params& params,                  \
                  raft::host_matrix_view<const T, int64_t, row_major> dataset,         \
                  raft::neighbors::cagra::index<T, IdxT>& idx);                        \
                                                                                       \
  void search(raft::resources const& handle,                                           \
              raft::neighbors::cagra::search_params const& params,                     \
              const raft::neighbors::cagra::index<T, IdxT>& index,                     \
              raft::device_matrix_view<const T, int64_t, row_major> queries,           \
              raft::device_matrix_view<IdxT, int64_t, row_major> neighbors,            \
              raft::device_matrix_view<float, int64_t, row_major> distances);          \
  void serialize_file(raft::resources const& handle,                                   \
                      const std::string& filename,                                     \
                      const raft::neighbors::cagra::index<T, IdxT>& index,             \
                      bool include_dataset = true);                                    \
                                                                                       \
  void deserialize_file(raft::resources const& handle,                                 \
                        const std::string& filename,                                   \
                        raft::neighbors::cagra::index<T, IdxT>* index);                \
  void serialize(raft::resources const& handle,                                        \
                 std::string& str,                                                     \
                 const raft::neighbors::cagra::index<T, IdxT>& index,                  \
                 bool include_dataset = true);                                         \
  void serialize_to_hnswlib(raft::resources const& handle,                             \
                            std::string& str,                                          \
                            const raft::neighbors::cagra::index<T, IdxT>& index);      \
  void serialize_to_hnswlib_file(raft::resources const& handle,                        \
                                 const std::string& filename,                          \
                                 const raft::neighbors::cagra::index<T, IdxT>& index); \
  void deserialize(raft::resources const& handle,                                      \
                   const std::string& str,                                             \
                   raft::neighbors::cagra::index<T, IdxT>* index);

RAFT_INST_CAGRA_FUNCS(float, uint32_t);
RAFT_INST_CAGRA_FUNCS(int8_t, uint32_t);
RAFT_INST_CAGRA_FUNCS(uint8_t, uint32_t);

#undef RAFT_INST_CAGRA_FUNCS

#define RAFT_INST_CAGRA_OPTIMIZE(IdxT)                                               \
  void optimize_device(raft::resources const& res,                                   \
                       raft::device_matrix_view<IdxT, int64_t, row_major> knn_graph, \
                       raft::host_matrix_view<IdxT, int64_t, row_major> new_graph);  \
                                                                                     \
  void optimize_host(raft::resources const& res,                                     \
                     raft::host_matrix_view<IdxT, int64_t, row_major> knn_graph,     \
                     raft::host_matrix_view<IdxT, int64_t, row_major> new_graph);

RAFT_INST_CAGRA_OPTIMIZE(uint32_t);

#undef RAFT_INST_CAGRA_OPTIMIZE

}  // namespace raft::runtime::neighbors::cagra
