/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/resources.hpp>

namespace raft::random {

/**
 * @defgroup rmat RMAT Rectangular Generator
 * @{
 */

/**
 * @brief Generate a bipartite RMAT graph for a rectangular adjacency matrix.
 *
 * This is the most general of several overloads of `rmat_rectangular_gen`
 * in this file, and thus has the most detailed documentation.
 *
 * @tparam IdxT  Type of each node index
 * @tparam ProbT Data type used for probability distributions (either fp32 or fp64)
 *
 * @param[in]  handle  RAFT handle, containing the CUDA stream on which to schedule work
 * @param[in]  r       underlying state of the random generator. Especially useful when
 *                     one wants to call this API for multiple times in order to generate
 *                     a larger graph. For that case, just create this object with the
 *                     initial seed once and after every call continue to pass the same
 *                     object for the successive calls.
 * @param[out] out     Generated edgelist [on device], packed in array-of-structs fashion.
 *                     In each row, the first element is the source node id,
 *                     and the second element is the destination node id.
 * @param[out] out_src Source node id's [on device].
 * @param[out] out_dst Destination node id's [on device].  `out_src` and `out_dst`
 *                     together form the struct-of-arrays representation of the same
 *                     output data as `out`.
 * @param[in]  theta   distribution of each quadrant at each level of resolution.
 *                     Since these are probabilities, each of the 2x2 matrices for
 *                     each level of the RMAT must sum to one. [on device]
 *                     [dim = max(r_scale, c_scale) x 2 x 2]. Of course, it is assumed
 *                     that each of the group of 2 x 2 numbers all sum up to 1.
 * @param[in]  r_scale 2^r_scale represents the number of source nodes
 * @param[in]  c_scale 2^c_scale represents the number of destination nodes
 *
 * @pre `out.extent(0) == 2 * `out_src.extent(0)` is `true`
 * @pre `out_src.extent(0) == out_dst.extent(0)` is `true`
 *
 * We call the `r_scale != c_scale` case the "rectangular adjacency matrix" case
 * (in other words, generating bipartite graphs). In this case, at `depth >= r_scale`,
 * the distribution is assumed to be:
 *
 * `[theta[4 * depth] + theta[4 * depth + 2], theta[4 * depth + 1] + theta[4 * depth + 3]; 0, 0]`.
 *
 * Then for `depth >= c_scale`, the distribution is assumed to be:
 *
 * `[theta[4 * depth] + theta[4 * depth + 1], 0; theta[4 * depth + 2] + theta[4 * depth + 3], 0]`.
 *
 * @note This can generate duplicate edges and self-loops. It is the responsibility of the
 *       caller to clean them up accordingly.
 *
 * @note This also only generates directed graphs. If undirected graphs are needed, then a
 *       separate post-processing step is expected to be done by the caller.
 *
 * @{
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen(
  raft::resources const& handle,
  raft::random::RngState& r,
  raft::device_vector_view<const ProbT, IdxT> theta,
  raft::device_mdspan<IdxT, raft::extents<IdxT, raft::dynamic_extent, 2>, raft::row_major> out,
  raft::device_vector_view<IdxT, IdxT> out_src,
  raft::device_vector_view<IdxT, IdxT> out_dst,
  IdxT r_scale,
  IdxT c_scale)
{
  detail::rmat_rectangular_gen_output<IdxT> output(out, out_src, out_dst);
  detail::rmat_rectangular_gen_impl(handle, r, theta, output, r_scale, c_scale);
}

/**
 * @brief Overload of `rmat_rectangular_gen` that only generates
 *   the struct-of-arrays (two vectors) output representation.
 *
 * This overload only generates the struct-of-arrays (two vectors)
 * output representation: output vector `out_src` of source node id's,
 * and output vector `out_dst` of destination node id's.
 *
 * @pre `out_src.extent(0) == out_dst.extent(0)` is `true`
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen(raft::resources const& handle,
                          raft::random::RngState& r,
                          raft::device_vector_view<const ProbT, IdxT> theta,
                          raft::device_vector_view<IdxT, IdxT> out_src,
                          raft::device_vector_view<IdxT, IdxT> out_dst,
                          IdxT r_scale,
                          IdxT c_scale)
{
  detail::rmat_rectangular_gen_output<IdxT> output(out_src, out_dst);
  detail::rmat_rectangular_gen_impl(handle, r, theta, output, r_scale, c_scale);
}

/**
 * @brief Overload of `rmat_rectangular_gen` that only generates
 *   the array-of-structs (one vector) output representation.
 *
 * This overload only generates the array-of-structs (one vector)
 * output representation: a single output vector `out`,
 * where in each row, the first element is the source node id,
 * and the second element is the destination node id.
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen(
  raft::resources const& handle,
  raft::random::RngState& r,
  raft::device_vector_view<const ProbT, IdxT> theta,
  raft::device_mdspan<IdxT, raft::extents<IdxT, raft::dynamic_extent, 2>, raft::row_major> out,
  IdxT r_scale,
  IdxT c_scale)
{
  detail::rmat_rectangular_gen_output<IdxT> output(out);
  detail::rmat_rectangular_gen_impl(handle, r, theta, output, r_scale, c_scale);
}

/**
 * @brief Overload of `rmat_rectangular_gen` that assumes the same
 *   a, b, c, d probability distributions across all the scales,
 *   and takes all three output vectors
 *   (`out` with the array-of-structs output representation,
 *   and `out_src` and `out_dst` with the struct-of-arrays
 *   output representation).
 *
 * `a`, `b, and `c` effectively replace the above overloads'
 * `theta` parameter.
 *
 * @pre `out.extent(0) == 2 * `out_src.extent(0)` is `true`
 * @pre `out_src.extent(0) == out_dst.extent(0)` is `true`
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen(
  raft::resources const& handle,
  raft::random::RngState& r,
  raft::device_mdspan<IdxT, raft::extents<IdxT, raft::dynamic_extent, 2>, raft::row_major> out,
  raft::device_vector_view<IdxT, IdxT> out_src,
  raft::device_vector_view<IdxT, IdxT> out_dst,
  ProbT a,
  ProbT b,
  ProbT c,
  IdxT r_scale,
  IdxT c_scale)
{
  detail::rmat_rectangular_gen_output<IdxT> output(out, out_src, out_dst);
  detail::rmat_rectangular_gen_impl(handle, r, output, a, b, c, r_scale, c_scale);
}

/**
 * @brief Overload of `rmat_rectangular_gen` that assumes the same
 *   a, b, c, d probability distributions across all the scales,
 *   and takes only two output vectors
 *   (the struct-of-arrays output representation).
 *
 * `a`, `b, and `c` effectively replace the above overloads'
 * `theta` parameter.
 *
 * @pre `out_src.extent(0) == out_dst.extent(0)` is `true`
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen(raft::resources const& handle,
                          raft::random::RngState& r,
                          raft::device_vector_view<IdxT, IdxT> out_src,
                          raft::device_vector_view<IdxT, IdxT> out_dst,
                          ProbT a,
                          ProbT b,
                          ProbT c,
                          IdxT r_scale,
                          IdxT c_scale)
{
  detail::rmat_rectangular_gen_output<IdxT> output(out_src, out_dst);
  detail::rmat_rectangular_gen_impl(handle, r, output, a, b, c, r_scale, c_scale);
}

/**
 * @brief Overload of `rmat_rectangular_gen` that assumes the same
 *   a, b, c, d probability distributions across all the scales,
 *   and takes only one output vector
 *   (the array-of-structs output representation).
 *
 * `a`, `b, and `c` effectively replace the above overloads'
 * `theta` parameter.
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen(
  raft::resources const& handle,
  raft::random::RngState& r,
  raft::device_mdspan<IdxT, raft::extents<IdxT, raft::dynamic_extent, 2>, raft::row_major> out,
  ProbT a,
  ProbT b,
  ProbT c,
  IdxT r_scale,
  IdxT c_scale)
{
  detail::rmat_rectangular_gen_output<IdxT> output(out);
  detail::rmat_rectangular_gen_impl(handle, r, output, a, b, c, r_scale, c_scale);
}

/** @} */  // end group rmat

/**
 * @brief Legacy overload of `rmat_rectangular_gen`
 *   taking raw arrays instead of mdspan.
 *
 * @tparam IdxT  type of each node index
 * @tparam ProbT data type used for probability distributions (either fp32 or fp64)
 *
 * @param[out] out     generated edgelist [on device] [dim = n_edges x 2]. In each row
 *                     the first element is the source node id, and the second element
 *                     is the destination node id. If you don't need this output
 *                     then pass a `nullptr` in its place.
 * @param[out] out_src list of source node id's [on device] [len = n_edges]. If you
 *                     don't need this output then pass a `nullptr` in its place.
 * @param[out] out_dst list of destination node id's [on device] [len = n_edges]. If
 *                     you don't need this output then pass a `nullptr` in its place.
 * @param[in]  theta   distribution of each quadrant at each level of resolution.
 *                     Since these are probabilities, each of the 2x2 matrices for
 *                     each level of the RMAT must sum to one. [on device]
 *                     [dim = max(r_scale, c_scale) x 2 x 2]. Of course, it is assumed
 *                     that each of the group of 2 x 2 numbers all sum up to 1.
 * @param[in]  r_scale 2^r_scale represents the number of source nodes
 * @param[in]  c_scale 2^c_scale represents the number of destination nodes
 * @param[in]  n_edges number of edges to generate
 * @param[in]  stream  cuda stream on which to schedule the work
 * @param[in]  r       underlying state of the random generator. Especially useful when
 *                     one wants to call this API for multiple times in order to generate
 *                     a larger graph. For that case, just create this object with the
 *                     initial seed once and after every call continue to pass the same
 *                     object for the successive calls.
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
  detail::rmat_rectangular_gen_caller(
    out, out_src, out_dst, theta, r_scale, c_scale, n_edges, stream, r);
}

/**
 * @brief Legacy overload of `rmat_rectangular_gen`
 *   taking raw arrays instead of mdspan.
 *   This overload assumes the same a, b, c, d probability distributions
 *   across all the scales.
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
  detail::rmat_rectangular_gen_caller(
    out, out_src, out_dst, a, b, c, r_scale, c_scale, n_edges, stream, r);
}

/** @} */

}  // end namespace raft::random
