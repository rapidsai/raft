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

#pragma once

#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/hnsw_types.hpp>

namespace raft::runtime::neighbors::hnsw {

#define RAFT_INST_HNSW_FUNCS(T, IdxT)                                         \
  std::unique_ptr<raft::neighbors::hnsw::index<T>> from_cagra(                \
    raft::resources const& res, raft::neighbors::cagra::index<T, IdxT>);      \
  void search(raft::resources const& handle,                                  \
              raft::neighbors::hnsw::search_params const& params,             \
              raft::neighbors::hnsw::index<T> const& index,                   \
              raft::host_matrix_view<const T, int64_t, row_major> queries,    \
              raft::host_matrix_view<uint64_t, int64_t, row_major> neighbors, \
              raft::host_matrix_view<float, int64_t, row_major> distances);   \
  template <typename DType>                                                   \
  std::unique_ptr<raft::neighbors::hnsw::index<DType>> deserialize_file(      \
    raft::resources const& handle,                                            \
    const std::string& filename,                                              \
    int dim,                                                                  \
    raft::distance::DistanceType metric);                                     \
  template <>                                                                 \
  std::unique_ptr<raft::neighbors::hnsw::index<T>> deserialize_file(          \
    raft::resources const& handle,                                            \
    const std::string& filename,                                              \
    int dim,                                                                  \
    raft::distance::DistanceType metric);

RAFT_INST_HNSW_FUNCS(float, uint32_t);
RAFT_INST_HNSW_FUNCS(int8_t, uint32_t);
RAFT_INST_HNSW_FUNCS(uint8_t, uint32_t);

}  // namespace raft::runtime::neighbors::hnsw
