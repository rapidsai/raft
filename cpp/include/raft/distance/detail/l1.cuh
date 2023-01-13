/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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
#include "distance_ops/l1.cuh"
#include "pairwise_matrix/dispatch.cuh"

namespace raft {
namespace distance {
namespace detail {

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void l1Impl(int m,
            int n,
            int k,
            const DataT* x,
            const DataT* y,
            OutT* out,
            FinOpT fin_op,
            cudaStream_t stream,
            bool is_row_major)
{
  ops::l1_distance_op distance_op{};

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  distance_matrix_dispatch<ops::l1_distance_op, DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

}  // namespace detail
}  // namespace distance
}  // namespace raft
