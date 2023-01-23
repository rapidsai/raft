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

#ifndef __MASKED_L2_NN_H
#define __MASKED_L2_NN_H

#pragma once

#include <limits>
#include <raft/distance/detail/masked_nn.cuh>
#include <raft/distance/fused_l2_nn.cuh>
#include <raft/handle.hpp>
#include <raft/util/cuda_utils.cuh>
#include <stdint.h>

namespace raft {
namespace distance {

/**
 * @brief Parameters for maskedL2NN function
 *
 * Prescribes how to reduce a distance to an intermediate type (`redOp`), and
 * how to reduce two intermediate types (`pairRedOp`). Typically, a distance is
 * mapped to an (index, value) pair and (index, value) pair with the lowest
 * value (distance) is selected.
 *
 * In addition, prescribes whether to compute the square root of the distance
 * (`sqrt`) and whether to initialize the output buffer (`initOutBuffer`).
  */
template <typename ReduceOpT, typename KVPReduceOpT>
struct MaskedL2NNParams {
  /** Reduction operator in the epilogue */
  ReduceOpT redOp;
  /** Reduction operation on key value pairs */
  KVPReduceOpT pairRedOp;
  /** Whether the output `minDist` should contain L2-sqrt */
  bool sqrt;
  /** Whether to initialize the output buffer before the main kernel launch */
  bool initOutBuffer;
};

/**
 * @brief Masked L2 distance and 1-nearest-neighbor computation in a single call.
 *
 * This function enables faster computation of nearest neighbors if the
 * computation of distances between certain point pairs can be skipped.
 *
 * To avoid using a full adjacency matrix between all points in `x` and `y`, the
 * points in `y` are divided into groups. An adjacency matrix describes for each
 * point in `x` and each group whether to compute the distance.
 *
 * **Performance considerations**
 *
 * The points in `x` are grouped into tiles of `M` points (`M` is currently 64,
 * but may change in the future). As a result, the largest compute time
 * reduction occurs if all `M` points can skip a group. If only part of the `M`
 * points can skip a group, then at most a minor compute time reduction and a
 * modest energy use reduction can be expected.
 *
 * The points in `y` are also grouped into tiles of `N` points (`N` is currently
 * 64, but may change in the future). As a result, group sizes should be larger
 * than `N` to avoid wasting computational resources. If the group sizes are
 * evenly divisible by `N`, then the computation is most efficient, although for
 * larger group sizes this effect is minor.
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
 * @param handle             RAFT handle for managing expensive resources
 * @param params             Parameter struct specifying the reduction operations.
 * @param[in]  x             first matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  xn            L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in]  yn            L2 squared norm of `y`. Length = `n`. (on device)
 * @param[in]  adj           A boolean adjacency matrix indicating for each
 *                           row of `x` and each group in `y` whether to compute the
 *                           distance. Dim = `m x num_groups`.
 * @param[in]  group_idxs    An array containing the *end* indices of each group
 *                           in `y`. The value of group_idxs[j] indicates the
 *                           start of group j + 1, i.e., it is the inclusive
 *                           scan of the group lengths. The first group is
 *                           always assumed to start at index 0 and the last
 *                           group typically ends at index `n`. Length =
 *                           `num_groups`.
 * @param[out] min           will contain the reduced output (Length = `m`)
 *                           (on device)
 */
template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT, typename KVPReduceOpT>
void maskedL2NN(raft::handle_t& handle,
                MaskedL2NNParams<ReduceOpT, KVPReduceOpT> params,
                raft::device_matrix_view<DataT, IdxT, raft::layout_c_contiguous> const x,
                raft::device_matrix_view<DataT, IdxT, raft::layout_c_contiguous> const y,
                raft::device_vector_view<DataT, IdxT, raft::layout_c_contiguous> const x_norm,
                raft::device_vector_view<DataT, IdxT, raft::layout_c_contiguous> const y_norm,
                raft::device_matrix_view<bool, IdxT, raft::layout_c_contiguous> const adj,
                raft::device_vector_view<IdxT, IdxT, raft::layout_c_contiguous> const group_idxs,
                raft::device_vector_view<OutT, IdxT, raft::layout_c_contiguous> out)
{
  // TODO: add more assertions.
  RAFT_EXPECTS(x.extent(1) == y.extent(1), "Dimension of vectors in x and y must be equal.");

  RAFT_EXPECTS(x.is_exhaustive(), "Input x must be contiguous.");
  RAFT_EXPECTS(y.is_exhaustive(), "Input y must be contiguous.");

  IdxT m          = x.extent(0);
  IdxT n          = y.extent(0);
  IdxT k          = x.extent(1);
  IdxT num_groups = group_idxs.extent(0);

  detail::maskedL2NNImpl<DataT, OutT, IdxT, ReduceOpT>(handle,
                                                       out.data_handle(),
                                                       x.data_handle(),
                                                       y.data_handle(),
                                                       x_norm.data_handle(),
                                                       y_norm.data_handle(),
                                                       adj.data_handle(),
                                                       group_idxs.data_handle(),
                                                       num_groups,
                                                       m,
                                                       n,
                                                       k,
                                                       params.redOp,
                                                       params.pairRedOp,
                                                       params.sqrt,
                                                       params.initOutBuffer);
}

}  // namespace distance
}  // namespace raft

#endif
