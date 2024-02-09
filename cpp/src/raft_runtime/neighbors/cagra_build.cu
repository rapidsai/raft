/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft_runtime/neighbors/cagra.hpp>

namespace raft::runtime::neighbors::cagra {

#define RAFT_INST_CAGRA_BUILD(T, IdxT)                                             \
  auto build(raft::resources const& handle,                                        \
             const raft::neighbors::cagra::index_params& params,                   \
             raft::device_matrix_view<const T, int64_t, row_major> dataset)        \
    ->raft::neighbors::cagra::index<T, IdxT>                                       \
  {                                                                                \
    return raft::neighbors::cagra::build<T, IdxT>(handle, params, dataset);        \
  }                                                                                \
                                                                                   \
  auto build(raft::resources const& handle,                                        \
             const raft::neighbors::cagra::index_params& params,                   \
             raft::host_matrix_view<const T, int64_t, row_major> dataset)          \
    ->raft::neighbors::cagra::index<T, IdxT>                                       \
  {                                                                                \
    return raft::neighbors::cagra::build<T, IdxT>(handle, params, dataset);        \
  }                                                                                \
                                                                                   \
  void build_device(raft::resources const& handle,                                 \
                    const raft::neighbors::cagra::index_params& params,            \
                    raft::device_matrix_view<const T, int64_t, row_major> dataset, \
                    raft::neighbors::cagra::index<T, IdxT>& idx)                   \
  {                                                                                \
    idx = build(handle, params, dataset);                                          \
  }                                                                                \
                                                                                   \
  void build_host(raft::resources const& handle,                                   \
                  const raft::neighbors::cagra::index_params& params,              \
                  raft::host_matrix_view<const T, int64_t, row_major> dataset,     \
                  raft::neighbors::cagra::index<T, IdxT>& idx)                     \
  {                                                                                \
    idx = build(handle, params, dataset);                                          \
  }

RAFT_INST_CAGRA_BUILD(float, uint32_t);
RAFT_INST_CAGRA_BUILD(int8_t, uint32_t);
RAFT_INST_CAGRA_BUILD(uint8_t, uint32_t);

#undef RAFT_INST_CAGRA_BUILD

#define RAFT_INST_CAGRA_OPTIMIZE(IdxT)                                               \
  void optimize_device(raft::resources const& handle,                                \
                       raft::device_matrix_view<IdxT, int64_t, row_major> knn_graph, \
                       raft::host_matrix_view<IdxT, int64_t, row_major> new_graph)   \
  {                                                                                  \
    raft::neighbors::cagra::optimize(handle, knn_graph, new_graph);                  \
  }                                                                                  \
  void optimize_host(raft::resources const& handle,                                  \
                     raft::host_matrix_view<IdxT, int64_t, row_major> knn_graph,     \
                     raft::host_matrix_view<IdxT, int64_t, row_major> new_graph)     \
  {                                                                                  \
    raft::neighbors::cagra::optimize(handle, knn_graph, new_graph);                  \
  }

RAFT_INST_CAGRA_OPTIMIZE(uint32_t);

#undef RAFT_INST_CAGRA_OPTIMIZE

}  // namespace raft::runtime::neighbors::cagra
