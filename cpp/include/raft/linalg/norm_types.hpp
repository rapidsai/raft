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

namespace raft {
namespace linalg {

/** Enum to tell how to compute a norm */
enum NormType : unsigned short {
  /** L0 (actually not a norm): sum((x_i != 0 ? 1 : 0)) */
  L0PseudoNorm = 0,
  /** L1 norm or Manhattan: sum(abs(x_i)) */
  L1Norm = 1,
  /** L2 norm or Euclidean: sqrt(sum(x_i^2)). Note that in some prims the square root is optional,
     in which case it can be specified using a boolean or a functor final_op */
  L2Norm = 2,
  /** Linf norm or Chebyshev: max(abs(x_i)) */
  LinfNorm
};

}  // namespace linalg
}  // namespace raft
