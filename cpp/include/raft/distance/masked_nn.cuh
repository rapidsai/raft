/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <raft/distance/detail/masked_nn.cuh>
#include <raft/distance/fused_l2_nn.cuh>
#include <raft/util/cuda_utils.cuh>

#include <stdint.h>

#include <limits>

namespace raft {
namespace distance {
/**
 * \defgroup masked_nn Masked 1-nearest neighbors
 * @{
 */

/**
 * @brief Parameter struct for masked_l2_nn function
 *
 * @tparam ReduceOpT    Type of reduction operator in the epilogue.
 * @tparam KVPReduceOpT Type of Reduction operation on key value pairs.
 *
 * Usage example:
 * @code{.cpp}
 * #include <raft/distance/masked_nn.cuh>
 *
 * using IdxT        = int;
 * using DataT       = float;
 * using RedOpT      = raft::distance::MinAndDistanceReduceOp<IdxT, DataT>;
 * using PairRedOpT  = raft::distance::KVPMinReduce<IdxT, DataT>;
 * using ParamT      = raft::distance::masked_l2_nn_params<RedOpT, PairRedOpT>;
 *
 * bool init_out = true;
 * bool sqrt     = false;
 *
 * ParamT masked_l2_params{RedOpT{}, PairRedOpT{}, sqrt, init_out};
 * @endcode
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
struct masked_l2_nn_params {
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
 * We use an adjacency matrix that describes which distances to calculate. The
 * points in `y` are divided into groups, and the adjacency matrix indicates
 * whether to compute distances between points in `x` and groups in `y`. In other
 * words, if `adj[i,k]` is true then distance between point `x_i`, and points in
 * `group_k` will be calculated.
 *
 * **Performance considerations**
 *
 * The points in `x` are processed in tiles of `M` points (`M` is currently 64,
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
 *
 * **Comparison to SDDM**
 *
 * [SDDMM](https://ieeexplore.ieee.org/document/8638042) (sampled dense-dense
 * matrix multiplication) is a matrix-matrix multiplication where only part of
 * the output is computed. Compared to masked_l2_nn, there are a few differences:
 *
 * - The output of masked_l2_nn is a single vector (of nearest neighbors) and not
 *   a sparse matrix.
 *
 * - The sampling in masked_l2_nn is expressed through intermediate "groups"
     rather than a CSR format.
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
 * @param[in]  x             First matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             Second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  x_norm        L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in]  y_norm        L2 squared norm of `y`. Length = `n`. (on device)
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
 * @param[out] out           will contain the reduced output (Length = `m`)
 *                           (on device)
 */
template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT, typename KVPReduceOpT>
void masked_l2_nn(raft::resources const& handle,
                  raft::distance::masked_l2_nn_params<ReduceOpT, KVPReduceOpT> params,
                  raft::device_matrix_view<const DataT, IdxT, raft::layout_c_contiguous> x,
                  raft::device_matrix_view<const DataT, IdxT, raft::layout_c_contiguous> y,
                  raft::device_vector_view<const DataT, IdxT, raft::layout_c_contiguous> x_norm,
                  raft::device_vector_view<const DataT, IdxT, raft::layout_c_contiguous> y_norm,
                  raft::device_matrix_view<const bool, IdxT, raft::layout_c_contiguous> adj,
                  raft::device_vector_view<const IdxT, IdxT, raft::layout_c_contiguous> group_idxs,
                  raft::device_vector_view<OutT, IdxT, raft::layout_c_contiguous> out)
{
  IdxT m          = x.extent(0);
  IdxT n          = y.extent(0);
  IdxT k          = x.extent(1);
  IdxT num_groups = group_idxs.extent(0);

  // Match k dimension of x, y
  RAFT_EXPECTS(x.extent(1) == y.extent(1), "Dimension of vectors in x and y must be equal.");
  // Match x, x_norm and y, y_norm
  RAFT_EXPECTS(m == x_norm.extent(0), "Length of `x_norm` must match input `x`.");
  RAFT_EXPECTS(n == y_norm.extent(0), "Length of `y_norm` must match input `y` ");
  // Match adj to x and group_idxs
  RAFT_EXPECTS(m == adj.extent(0), "#rows in `adj` must match input `x`.");
  RAFT_EXPECTS(num_groups == adj.extent(1), "#cols in `adj` must match length of `group_idxs`.");
  // NOTE: We do not check if all indices in group_idxs actually points *inside* y.

  // If there is no work to be done, return immediately.
  if (m == 0 || n == 0 || k == 0 || num_groups == 0) { return; }

  detail::masked_l2_nn_impl<DataT, OutT, IdxT, ReduceOpT>(handle,
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

/** @} */

}  // namespace distance
}  // namespace raft

#endif
