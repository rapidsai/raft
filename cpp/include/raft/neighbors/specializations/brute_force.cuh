/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <raft/neighbors/brute_force.cuh>

// also define the detail api, which is used by raft::neighbors::brute_force
// (not doing the public api, since has extra template params on index_layout, matrix_index,
// search_layout etc - and isn't clear what the defaults here should be)
namespace raft::neighbors::detail {
#define RAFT_INST(IdxT, T, IntT)                                                                 \
  extern template void brute_force_knn_impl<IntT, IdxT, T>(raft::device_resources const& handle, \
                                                           std::vector<T*>& input,               \
                                                           std::vector<IntT>& sizes,             \
                                                           IntT D,                               \
                                                           T* search_items,                      \
                                                           IntT n,                               \
                                                           IdxT* res_I,                          \
                                                           T* res_D,                             \
                                                           IntT k,                               \
                                                           bool rowMajorIndex,                   \
                                                           bool rowMajorQuery,                   \
                                                           std::vector<IdxT>* translations,      \
                                                           raft::distance::DistanceType metric,  \
                                                           float metricArg,                      \
                                                           raft::identity_op);
RAFT_INST(long, float, int);
RAFT_INST(long, float, unsigned int);
RAFT_INST(uint32_t, float, int);
RAFT_INST(uint32_t, float, unsigned int);
#undef RAFT_INST
}  // namespace raft::neighbors::detail
