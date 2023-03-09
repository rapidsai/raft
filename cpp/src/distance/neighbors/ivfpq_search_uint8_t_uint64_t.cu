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
#include <raft/neighbors/specializations/ivf_pq.cuh>
#include <raft_runtime/neighbors/ivf_pq.hpp>

namespace raft::runtime::neighbors::ivf_pq {

#define RAFT_SEARCH_INST(T, IdxT)                                         \
  void search(raft::device_resources const& handle,                       \
              const raft::neighbors::ivf_pq::search_params& params,       \
              const raft::neighbors::ivf_pq::index<IdxT>& idx,            \
              raft::device_matrix_view<const T, IdxT, row_major> queries, \
              uint32_t k,                                                 \
              raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,  \
              raft::device_matrix_view<float, IdxT, row_major> distances) \
  {                                                                       \
    raft::neighbors::ivf_pq::search<T, IdxT>(                             \
      handle, params, idx, queries, k, neighbors, distances);             \
  }

RAFT_SEARCH_INST(uint8_t, uint64_t);

#undef RAFT_INST_SEARCH

}  // namespace raft::runtime::neighbors::ivf_pq
