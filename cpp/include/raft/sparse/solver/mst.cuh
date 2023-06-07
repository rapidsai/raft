
/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <raft/sparse/solver/mst_solver.cuh>

namespace raft::sparse::solver {

/**
 * Compute the minimum spanning tree (MST) or minimum spanning forest (MSF) depending on
 * the connected components of the given graph.
 *
 * @tparam vertex_t integral type for precision of vertex indexing
 * @tparam edge_t integral type for precision of edge indexing
 * @tparam weight_t type of weights array
 * @tparam alteration_t type to use for random alteration
 *
 * @param handle
 * @param offsets csr inptr array of row offsets (size v+1)
 * @param indices csr array of column indices (size e)
 * @param weights csr array of weights (size e)
 * @param v number of vertices in graph
 * @param e number of edges in graph
 * @param color array to store resulting colors for MSF
 * @param stream cuda stream for ordering operations
 * @param symmetrize_output should the resulting output edge list should be symmetrized?
 * @param initialize_colors should the colors array be initialized inside the MST?
 * @param iterations maximum number of iterations to perform
 * @return a list of edges containing the mst (or a subset of the edges guaranteed to be in the mst
 * when an msf is encountered)
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t = weight_t>
Graph_COO<vertex_t, edge_t, weight_t> mst(raft::resources const& handle,
                                          edge_t const* offsets,
                                          vertex_t const* indices,
                                          weight_t const* weights,
                                          vertex_t const v,
                                          edge_t const e,
                                          vertex_t* color,
                                          cudaStream_t stream,
                                          bool symmetrize_output = true,
                                          bool initialize_colors = true,
                                          int iterations         = 0)
{
  MST_solver<vertex_t, edge_t, weight_t, alteration_t> mst_solver(handle,
                                                                  offsets,
                                                                  indices,
                                                                  weights,
                                                                  v,
                                                                  e,
                                                                  color,
                                                                  stream,
                                                                  symmetrize_output,
                                                                  initialize_colors,
                                                                  iterations);
  return mst_solver.solve();
}

}  // end namespace raft::sparse::solver
