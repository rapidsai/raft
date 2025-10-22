/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/map_then_reduce.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <typename math_t, int TPB = 256>
void meanSquaredError(
  math_t* out, const math_t* A, const math_t* B, size_t len, math_t weight, cudaStream_t stream)
{
  auto sq_diff = [len, weight] __device__(const math_t a, const math_t b) {
    math_t diff = a - b;
    return diff * diff * weight / len;
  };
  raft::linalg::mapThenSumReduce<math_t, decltype(sq_diff)>(out, len, sq_diff, stream, A, B);
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
