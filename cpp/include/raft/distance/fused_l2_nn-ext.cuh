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

#include <cstdint>  // int64_t
#include <raft/core/kvp.hpp>
#include <raft/util/raft_explicit.hpp>  // RAFT_EXPLICIT

#ifdef RAFT_EXPLICIT_INSTANTIATE

namespace raft {
namespace distance {
/**
 * \defgroup fused_l2_nn Fused 1-nearest neighbors
 * @{
 * @}
 */

/**
 * @brief Wrapper around fusedL2NN with minimum reduction operators.
 *
 * fusedL2NN cannot be compiled in the distance library due to the lambda
 * operators, so this wrapper covers the most common case (minimum).
 * This should be preferred to the more generic API when possible, in order to
 * reduce compilation times for users of the shared library.
 *
 * @tparam DataT     data type
 * @tparam OutT      output type to either store 1-NN indices and their minimum
 *                   distances (e.g. raft::KeyValuePair<int, float>) or store only the min
 * distances.
 * @tparam IdxT      indexing arithmetic type
 * @param[out] min           will contain the reduced output (Length = `m`)
 *                           (on device)
 * @param[in]  x             first matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  xn            L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in]  yn            L2 squared norm of `y`. Length = `n`. (on device)
 * @param[in]  m             gemm m
 * @param[in]  n             gemm n
 * @param[in]  k             gemm k
 * @param[in]  workspace     temp workspace. Size = sizeof(int)*m. (on device)
 * @param[in]  sqrt          Whether the output `minDist` should contain L2-sqrt
 * @param[in]  initOutBuffer whether to initialize the output buffer before the
 *                           main kernel launch
 * @param[in]  stream        cuda stream
 */
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

/** @} */

}  // namespace distance
}  // namespace raft

#endif  // RAFT_EXPLICIT_INSTANTIATE

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
