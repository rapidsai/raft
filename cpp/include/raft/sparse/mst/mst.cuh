
/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "mst_solver.cuh"

namespace raft {
namespace mst {

template <typename vertex_t, typename edge_t, typename weight_t,
          typename alteration_t = weight_t>
raft::Graph_COO<vertex_t, edge_t, weight_t> mst(
  const raft::handle_t& handle, edge_t const* offsets, vertex_t const* indices,
  weight_t const* weights, vertex_t const v, edge_t const e, vertex_t* color,
  cudaStream_t stream, bool symmetrize_output = true,
  bool initialize_colors = true, int iterations = 0) {
  MST_solver<vertex_t, edge_t, weight_t, alteration_t> mst_solver(
    handle, offsets, indices, weights, v, e, color, stream, symmetrize_output,
    initialize_colors, iterations);
  return mst_solver.solve();
}

}  // namespace mst
}  // namespace raft
