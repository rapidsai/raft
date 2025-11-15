/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace raft::distance::detail {

template <typename IdxT, typename DataT, typename OutT, typename FinOpT>
struct pairwise_matrix_params {
  IdxT m;
  IdxT n;
  IdxT k;
  IdxT ldx;
  IdxT ldy;
  IdxT ld_out;
  const DataT* x;
  const DataT* y;
  const DataT* x_norm;
  const DataT* y_norm;
  OutT* out;
  FinOpT fin_op;
  bool is_row_major;

  /// @brief: Flips the x and y input and corresponding sizes
  void flip_x_and_y()
  {
    // Flip m, n; ldx, ldy; x, y; x_norm, y_norm.
    std::swap(m, n);
    std::swap(ldx, ldy);
    std::swap(x, y);
    std::swap(x_norm, y_norm);
  }
};

}  // namespace raft::distance::detail
