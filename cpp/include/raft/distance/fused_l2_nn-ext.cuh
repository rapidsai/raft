/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <raft/core/kvp.hpp>                            // raft::KeyValuePair
#include <raft/core/resources.hpp>                      // raft::resources
#include <raft/distance/fused_distance_nn_helpers.cuh>  // include initialize and reduce operations
#include <raft/util/raft_explicit.hpp>                  // RAFT_EXPLICIT

#include <cstdint>  // int64_t

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

namespace raft {
namespace distance {

template <typename DataT, typename OutT, typename IdxT>
void fusedL2NNMinReduce(OutT* min,
                        const DataT* x,
                        const DataT* y,
                        const DataT* xn,
                        const DataT* yn,
                        IdxT m,
                        IdxT n,
                        IdxT k,
                        void* workspace,
                        bool sqrt,
                        bool initOutBuffer,
                        cudaStream_t stream) RAFT_EXPLICIT;

}  // namespace distance
}  // namespace raft

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_distance_fusedL2NNMinReduce(DataT, OutT, IdxT)                          \
  extern template void raft::distance::fusedL2NNMinReduce<DataT, OutT, IdxT>(OutT * min,         \
                                                                             const DataT* x,     \
                                                                             const DataT* y,     \
                                                                             const DataT* xn,    \
                                                                             const DataT* yn,    \
                                                                             IdxT m,             \
                                                                             IdxT n,             \
                                                                             IdxT k,             \
                                                                             void* workspace,    \
                                                                             bool sqrt,          \
                                                                             bool initOutBuffer, \
                                                                             cudaStream_t stream)

instantiate_raft_distance_fusedL2NNMinReduce(double, double, int);
instantiate_raft_distance_fusedL2NNMinReduce(double, double, int64_t);
instantiate_raft_distance_fusedL2NNMinReduce(float, float, int);
instantiate_raft_distance_fusedL2NNMinReduce(float, float, int64_t);

// We can't have comma's in the macro expansion, so we use the COMMA macro:
#define COMMA ,

instantiate_raft_distance_fusedL2NNMinReduce(double, raft::KeyValuePair<int COMMA double>, int);
instantiate_raft_distance_fusedL2NNMinReduce(double,
                                             raft::KeyValuePair<int64_t COMMA double>,
                                             int64_t);
instantiate_raft_distance_fusedL2NNMinReduce(float, raft::KeyValuePair<int COMMA float>, int);
instantiate_raft_distance_fusedL2NNMinReduce(float,
                                             raft::KeyValuePair<int64_t COMMA float>,
                                             int64_t);

#undef COMMA

#undef instantiate_raft_distance_fusedL2NNMinReduce
