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

#include <raft/neighbors/cagra.cuh>
#include <raft_runtime/neighbors/cagra.hpp>

namespace raft::runtime::neighbors::cagra {

#define RAFT_INST_CAGRA_SEARCH(T, IdxT)                                                            \
  void search(raft::resources const& handle,                                                       \
              raft::neighbors::cagra::search_params const& params,                                 \
              const raft::neighbors::cagra::index<T, IdxT>& index,                                 \
              raft::device_matrix_view<const T, int64_t, row_major> queries,                       \
              raft::device_matrix_view<IdxT, int64_t, row_major> neighbors,                        \
              raft::device_matrix_view<float, int64_t, row_major> distances)                       \
  {                                                                                                \
    raft::neighbors::cagra::search<T, IdxT>(handle, params, index, queries, neighbors, distances); \
  }

RAFT_INST_CAGRA_SEARCH(float, uint32_t);
RAFT_INST_CAGRA_SEARCH(int8_t, uint32_t);
RAFT_INST_CAGRA_SEARCH(uint8_t, uint32_t);

#undef RAFT_INST_CAGRA_SEARCH

}  // namespace raft::runtime::neighbors::cagra
