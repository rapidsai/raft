/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * Copyright 2020 KETAN DATE & RAKESH NAGI
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
 *
 *      CUDA Implementation of O(n^3) alternating tree Hungarian Algorithm
 *      Authors: Ketan Date and Rakesh Nagi
 *
 *      Article reference:
 *          Date, Ketan, and Rakesh Nagi. "GPU-accelerated Hungarian algorithms
 *          for the Linear Assignment Problem." Parallel Computing 57 (2016): 52-72.
 *
 */
#pragma once

namespace raft::solver {
template <typename vertex_t, typename weight_t>
struct Vertices {
  vertex_t* row_assignments;
  vertex_t* col_assignments;
  int* row_covers;
  int* col_covers;
  weight_t* row_duals;
  weight_t* col_duals;
  weight_t* col_slacks;
};

template <typename vertex_t>
struct VertexData {
  vertex_t* parents;
  vertex_t* children;
  int* is_visited;
};
}  // namespace raft::solver
