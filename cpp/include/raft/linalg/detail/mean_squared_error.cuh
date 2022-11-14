/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
