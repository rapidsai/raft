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

namespace raft::neighbors::ivf_pq {

#define RAFT_INST_BUILD_EXTEND(T, IdxT)                               \
  template auto extend<T, IdxT>(raft::device_resources const& handle, \
                                const index<IdxT>& orig_index,        \
                                const T* new_vectors,                 \
                                const IdxT* new_indices,              \
                                IdxT n_rows)                          \
    ->index<IdxT>;                                                    \
  template void extend<T, IdxT>(raft::device_resources const& handle, \
                                index<IdxT>* index,                   \
                                const T* new_vectors,                 \
                                const IdxT* new_indices,              \
                                IdxT n_rows);

RAFT_INST_BUILD_EXTEND(int8_t, uint64_t);

#undef RAFT_INST_BUILD_EXTEND

}  // namespace raft::neighbors::ivf_pq
