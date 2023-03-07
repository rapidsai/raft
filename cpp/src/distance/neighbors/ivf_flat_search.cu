/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <raft/neighbors/ivf_pq.cuh>
// #include <raft/neighbors/specializations/detail/ivf_flat_search.cuh>
#include <raft_runtime/neighbors/ivf_flat.hpp>

namespace raft::runtime::neighbors::ivf_flat {

#define RAFT_INST_SEARCH(T, IdxT)                                         \
  void search(raft::device_resources const&,                              \
              const raft::neighbors::ivf_flat::search_params&,            \
              const raft::neighbors::ivf_flat::index<T, IdxT>&,           \
              raft::device_matrix_view<const T, IdxT, row_major> queries, \
              raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,  \
              raft::device_matrix_view<T, IdxT, row_major> distances,     \
              uint32_t k);

RAFT_INST_SEARCH(float, uint64_t);
RAFT_INST_SEARCH(int8_t, uint64_t);
RAFT_INST_SEARCH(uint8_t, uint64_t);

#undef RAFT_INST_SEARCH

}  // namespace raft::runtime::neighbors::ivf_flat
