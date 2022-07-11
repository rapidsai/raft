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

#include "detail/rmat_rectangular_generator.cuh"

namespace raft::random {

/**
 * @brief Generate RMAT for a rectangular shaped adjacency matrices (useful when
 *        graphs to be generated are bipartite)
 *
 * @tparam IdxT  node indices type
 * @tparam ProbT data type used for probability distributions (either fp32 or fp64)
 *
 * @param[out] out     generated edgelist [on device] [dim = n_edges x 2]. On each row
 *                     the first element corresponds to the source node id while the
 *                     second, the destination node id. If you don't need this output
 *                     then pass a `nullptr` in its place.
 * @param[out] out_src list of source node id's [on device] [len = n_edges]. If you
 *                     don't need this output then pass a `nullptr` in its place.
 * @param[out] out_dst list of destination node id's [on device] [len = n_edges]. If
 *                     you don't need this output then pass a `nullptr` in its place.
 * @param[in]  theta   distribution of each quadrant at each level of resolution.
 *                     Since these are probabilities, each of the 2x2 matrix for
 *                     each level of the RMAT must sum to one. [on device]
 *                     [dim = max(r_scale, c_scale) x 2 x 2]. Of course, it is assumed
 *                     that each of the group of 2 x 2 numbers all sum up to 1.
 * @param[in]  r_scale 2^r_scale represents the number of source nodes
 * @param[in]  c_scale 2^c_scale represents the number of destination nodes
 * @param[in]  n_edges number of edges to generate
 * @param[in]  stream  cuda stream to schedule the work on
 * @param[in]  r       underlying state of the random generator. Especially useful when
 *                     one wants to call this API for multiple times in order to generate
 *                     a larger graph. For that case, just create this object with the
 *                     initial seed once and after every call continue to pass the same
 *                     object for the successive calls.
 *
 * @note This can generate duplicate edges and self-loops. It is the responsibility of the
 *       caller to clean them up accordingly.

 * @note This also only generates directed graphs. If undirected graphs are needed, then a
 *       separate post-processing step is expected to be done by the caller.
 *
 * @{
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen(IdxT* out,
			  IdxT* out_src,
			  IdxT* out_dst,
			  const ProbT* theta,
			  IdxT r_scale,
			  IdxT c_scale,
			  IdxT n_edges,
			  cudaStream_t stream,
			  raft::random::RngState& r)
{
  detail::rmat_rectangular_gen_caller(out,
				      out_src,
				      out_dst,
				      theta,
				      r_scale,
				      c_scale,
				      n_edges,
				      stream,
				      r);
}

/**
 * This is the same as the previous method but assumes the same a, b, c, d probability
 * distributions across all the scales
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen(IdxT* out,
			  IdxT* out_src,
			  IdxT* out_dst,
			  ProbT a,
			  ProbT b,
			  ProbT c,
			  IdxT r_scale,
			  IdxT c_scale,
			  IdxT n_edges,
			  cudaStream_t stream,
			  raft::random::RngState& r)
{
  detail::rmat_rectangular_gen_caller(out,
				      out_src,
				      out_dst,
				      a,
				      b,
				      c,
				      r_scale,
				      c_scale,
				      n_edges,
				      stream,
				      r);
}
/** @} */

}  // end namespace raft::random
