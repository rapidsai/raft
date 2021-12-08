/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <stdint.h>
#include <cub/cub.cuh>
#include <limits>
#include <raft/cuda_utils.cuh>
#include <raft/distance/detail/fused_l2_nn.cuh>
#include <raft/handle.hpp>

namespace raft {
namespace distance {

template <typename LabelT, typename DataT>
using KVPMinReduce = detail::KVPMinReduceImpl<LabelT, DataT>;

template <typename LabelT, typename DataT>
using MinAndDistanceReduceOp = detail::MinAndDistanceReduceOpImpl<LabelT, DataT>;

template <typename LabelT, typename DataT>
using MinReduceOp = detail::MinReduceOpImpl<LabelT, DataT>;

/**
 * Initialize array using init value from reduction op
 */
template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
void initialize(const raft::handle_t& handle, OutT* min, IdxT m, DataT maxVal, ReduceOpT redOp)
{
  detail::initialize<DataT, OutT, IdxT, ReduceOpT>(min, m, maxVal, redOp, handle.get_stream());
}

/**
 * @brief Fused L2 distance and 1-nearest-neighbor computation in a single call.
 *
 * The benefits of such a call are 2-fold: 1) eliminate the need for an
 * intermediate buffer to store the output of gemm 2) reduce the memory read
 * traffic on this intermediate buffer, otherwise needed during the reduction
 * phase for 1-NN.
 *
 * @tparam DataT     data type
 * @tparam OutT      output type to either store 1-NN indices and their minimum
 *                   distances or store only the min distances. Accordingly, one
 *                   has to pass an appropriate `ReduceOpT`
 * @tparam IdxT      indexing arithmetic type
 * @tparam ReduceOpT A struct to perform the final needed reduction operation
 *                   and also to initialize the output array elements with the
 *                   appropriate initial value needed for reduction.
 *
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
 * @param[in]  redOp         reduction operator in the epilogue
 * @param[in] pairRedOp reduction operation on key value pairs
 * @param[in]  sqrt          Whether the output `minDist` should contain L2-sqrt
 * @param[in]  initOutBuffer whether to initialize the output buffer before the
 *                           main kernel launch
 * @param[in]  stream        cuda stream
 */
template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT, typename KVPReduceOpT>
void fusedL2NN(OutT* min,
               const DataT* x,
               const DataT* y,
               const DataT* xn,
               const DataT* yn,
               IdxT m,
               IdxT n,
               IdxT k,
               void* workspace,
               ReduceOpT redOp,
               KVPReduceOpT pairRedOp,
               bool sqrt,
               bool initOutBuffer,
               cudaStream_t stream)
{
  size_t bytes = sizeof(DataT) * k;
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0) {
    detail::fusedL2NNImpl<DataT, OutT, IdxT, 16 / sizeof(DataT), ReduceOpT>(
      min, x, y, xn, yn, m, n, k, (int*)workspace, redOp, pairRedOp, sqrt, initOutBuffer, stream);
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0) {
    detail::fusedL2NNImpl<DataT, OutT, IdxT, 8 / sizeof(DataT), ReduceOpT>(
      min, x, y, xn, yn, m, n, k, (int*)workspace, redOp, pairRedOp, sqrt, initOutBuffer, stream);
  } else {
    detail::fusedL2NNImpl<DataT, OutT, IdxT, 1, ReduceOpT>(
      min, x, y, xn, yn, m, n, k, (int*)workspace, redOp, pairRedOp, sqrt, initOutBuffer, stream);
  }
}

}  // namespace distance
}  // namespace raft
