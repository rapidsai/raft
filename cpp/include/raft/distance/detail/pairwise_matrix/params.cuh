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

  //
  [[nodiscard]] pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> flip_x_and_y()
  {
    // Flip m, n; ldx, ldy; x, y; x_norm, y_norm.
    return pairwise_matrix_params<IdxT, DataT, OutT, FinOpT>{
      n, m, k, ldy, ldx, ld_out, y, x, y_norm, x_norm, out, fin_op, is_row_major};
  }
};

}  // namespace raft::distance::detail
