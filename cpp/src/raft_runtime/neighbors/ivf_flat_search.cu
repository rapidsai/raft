/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/neighbors/ivf_flat.cuh>

#include <raft_runtime/neighbors/ivf_flat.hpp>

namespace raft::runtime::neighbors::ivf_flat {

#define RAFT_INST_SEARCH(T, IdxT)                                         \
  void search(raft::resources const& handle,                              \
              raft::neighbors::ivf_flat::search_params const& params,     \
              const raft::neighbors::ivf_flat::index<T, IdxT>& index,     \
              raft::device_matrix_view<const T, IdxT, row_major> queries, \
              raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,  \
              raft::device_matrix_view<float, IdxT, row_major> distances) \
  {                                                                       \
    raft::neighbors::ivf_flat::search<T, IdxT>(                           \
      handle, params, index, queries, neighbors, distances);              \
  }

RAFT_INST_SEARCH(float, int64_t);
RAFT_INST_SEARCH(int8_t, int64_t);
RAFT_INST_SEARCH(uint8_t, int64_t);

#undef RAFT_INST_SEARCH

}  // namespace raft::runtime::neighbors::ivf_flat
