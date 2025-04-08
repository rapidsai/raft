/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cstdint>

namespace raft::sparse::solver {

enum LANCZOS_WHICH { LA, LM, SA, SM };

template <typename ValueTypeT>
struct lanczos_solver_config {
  /** The number of eigenvalues and eigenvectors to compute. Must be 1 <= k < n.*/
  int n_components;
  /** Maximum number of iteration. */
  int max_iterations;
  /** The number of Lanczos vectors generated. Must be k + 1 < ncv < n. */
  int ncv;
  /** Tolerance for residuals ``||Ax - wx||`` */
  ValueTypeT tolerance;
  /** which=**/
  LANCZOS_WHICH which;
  /** random seed */
  uint64_t seed;
};

}  // namespace raft::sparse::solver
