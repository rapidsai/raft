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

#include <raft/neighbors/specializations/ivf_flat.cuh>

namespace raft::neighbors::ivf_flat {

#define RAFT_MAKE_INSTANCE(T, IdxT)                                                   \
  template auto build(raft::device_resources const& handle,                           \
                      raft::device_matrix_view<const T, uint64_t, row_major> dataset, \
                      const index_params& params)                                     \
    ->index<T, IdxT>;                                                                 \
                                                                                      \
  template void build(raft::device_resources const& handle,                           \
                      raft::device_matrix_view<const T, uint64_t, row_major> dataset, \
                      const index_params& params,                                     \
                      index<T, IdxT>* idx);

RAFT_MAKE_INSTANCE(float, uint64_t);

#undef RAFT_MAKE_INSTANCE

}  // namespace raft::neighbors::ivf_flat
