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

#include <optional>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <variant>

namespace raft::random {

/**
 * @brief Type of the output vector(s) parameter for `rmat_rectangular_gen`
 *   (see below).
 *
 * @tparam IdxT Type of each node index; must be integral.
 *
 * Users can provide either `out` by itself, (`out_src` and `out_dst`)
 * together, or all three (`out`, `out_src`, and `out_dst`).
 * This class prevents users from doing anything other than that.
 * It also checks compatibility of dimensions at run time.
 *
 * The following examples show how to create an output parameter.
 *
 * @code
 * rmat_rectangular_gen_output<size_t> output1(out);
 * rmat_rectangular_gen_output<size_t> output2(out_src, out_dst);
 * rmat_rectangular_gen_output<size_t> output3(out, out_src, out_dst);
 * @endcode
 *
 * @{
 */
template <typename IdxT>
class rmat_rectangular_gen_output {
 public:
  using out_view_type =
    raft::device_mdspan<IdxT, raft::extents<IdxT, raft::dynamic_extent, 2>, raft::row_major>;
  using out_src_view_type = raft::device_vector_view<IdxT, IdxT>;
  using out_dst_view_type = raft::device_vector_view<IdxT, IdxT>;

 private:
  class output_pair {
   public:
    output_pair(const out_src_view_type& src, const out_dst_view_type& dst) : src_(src), dst_(dst)
    {
      RAFT_EXPECTS(src.extent(0) == dst.extent(0),
                   "rmat_rectangular_gen: "
                   "out_src.extent(0) = %d != out_dst.extent(0) = %d",
                   static_cast<int>(src.extent(0)),
                   static_cast<int>(dst.extent(0)));
    }

    out_src_view_type out_src_view() const { return src_; }

    out_dst_view_type out_dst_view() const { return dst_; }

    IdxT number_of_edges() const { return src_.extent(0); }

   private:
    out_src_view_type src_;
    out_dst_view_type dst_;
  };

  class output_triple {
   public:
    output_triple(const out_view_type& out,
                  const out_src_view_type& src,
                  const out_dst_view_type& dst)
      : out_(out), pair_(src, dst)
    {
      RAFT_EXPECTS(out.extent(0) == IdxT(2) * dst.extent(0),
                   "rmat_rectangular_gen: "
                   "out.extent(0) = %d != 2 * out_dst.extent(0) = %d",
                   static_cast<int>(out.extent(0)),
                   static_cast<int>(IdxT(2) * dst.extent(0)));
    }

    out_view_type out_view() const { return out_; }

    out_src_view_type out_src_view() const { return pair_.out_src_view(); }

    out_dst_view_type out_dst_view() const { return pair_.out_dst_view(); }

    IdxT number_of_edges() const { return pair_.number_of_edges(); }

   private:
    out_view_type out_;
    output_pair pair_;
  };

 public:
  /**
   * @brief Constructor taking no vectors,
   *   that effectively makes all the vectors length zero.
   */
  rmat_rectangular_gen_output() = default;

  /**
   * @brief Constructor taking a single vector, that packs the source
   *   node ids and destination node ids in array-of-structs fashion.
   *
   * @param[out] out Generated edgelist [on device].  In each row, the
   *   first element is the source node id, and the second element is
   *   the destination node id.
   */
  rmat_rectangular_gen_output(const out_view_type& out) : data_(out) {}

  /**
   * @brief Constructor taking two vectors, that store the source node
   *   ids and the destination node ids separately, in
   *   struct-of-arrays fashion.
   *
   * @param[out] out_src Source node id's [on device] [len = n_edges].
   *
   * @param[out] out_dst Destination node id's [on device] [len = n_edges].
   */
  rmat_rectangular_gen_output(const out_src_view_type& src, const out_dst_view_type& dst)
    : data_(output_pair(src, dst))
  {
  }

  /**
   * @brief Constructor taking all three vectors.
   *
   * @param[out] out Generated edgelist [on device].  In each row, the
   *   first element is the source node id, and the second element is
   *   the destination node id.
   *
   * @param[out] out_src Source node id's [on device] [len = n_edges].
   *
   * @param[out] out_dst Destination node id's [on device] [len = n_edges].
   */
  rmat_rectangular_gen_output(const out_view_type& out,
                              const out_src_view_type& src,
                              const out_dst_view_type& dst)
    : data_(output_triple(out, src, dst))
  {
  }

  /**
   * @brief Whether this object was created with a constructor
   *   taking more than zero arguments.
   */
  bool has_value() const { return not std::holds_alternative<std::nullopt_t>(data_); }

  /**
   * @brief Vector for the output single edgelist; the argument given
   *   to the one-argument constructor, or the first argument of the
   *   three-argument constructor; `std::nullopt` if not provided.
   */
  std::optional<out_view_type> out_view() const
  {
    if (std::holds_alternative<out_view_type>(data_)) {
      return std::get<out_view_type>(data_);
    } else if (std::holds_alternative<output_triple>(data_)) {
      return std::get<output_triple>(data_).out_view();
    } else {
      return std::nullopt;
    }
  }

  /**
   * @brief Vector for the output source edgelist; the first argument
   *   given to the two-argument constructor, or the second argument
   *   of the three-argument constructor; `std::nullopt` if not provided.
   */
  std::optional<out_src_view_type> out_src_view() const
  {
    if (std::holds_alternative<output_pair>(data_)) {
      return std::get<output_pair>(data_).out_src_view();
    } else if (std::holds_alternative<output_triple>(data_)) {
      return std::get<output_triple>(data_).out_src_view();
    } else {
      return std::nullopt;
    }
  }

  /**
   * @brief Vector for the output destination edgelist; the second
   *   argument given to the two-argument constructor, or the third
   *   argument of the three-argument constructor;
   *   `std::nullopt` if not provided.
   */
  std::optional<out_dst_view_type> out_dst_view() const
  {
    if (std::holds_alternative<output_pair>(data_)) {
      return std::get<output_pair>(data_).out_dst_view();
    } else if (std::holds_alternative<output_triple>(data_)) {
      return std::get<output_triple>(data_).out_dst_view();
    } else {
      return std::nullopt;
    }
  }

  /**
   * @brief Number of edges in the graph; zero if no output vector
   *   was provided to the constructor.
   */
  IdxT number_of_edges() const
  {
    if (std::holds_alternative<out_view_type>(data_)) {
      return std::get<out_view_type>(data_).extent(0);
    } else if (std::holds_alternative<output_pair>(data_)) {
      return std::get<output_pair>(data_).number_of_edges();
    } else if (std::holds_alternative<output_triple>(data_)) {
      return std::get<output_triple>(data_).number_of_edges();
    } else {
      return IdxT(0);
    }
  }

 private:
  // Defaults to std::nullopt.
  std::variant<std::nullopt_t, out_view_type, output_pair, output_triple> data_;
};

/**
 * @brief Generate RMAT for a rectangular adjacency matrix (useful when
 *        graphs to be generated are bipartite)
 *
 * @tparam IdxT  type of each node index
 * @tparam ProbT data type used for probability distributions (either fp32 or fp64)
 *
 * @param[in]  handle  RAFT handle, containing the CUDA stream on which to schedule work
 * @param[in]  r       underlying state of the random generator. Especially useful when
 *                     one wants to call this API for multiple times in order to generate
 *                     a larger graph. For that case, just create this object with the
 *                     initial seed once and after every call continue to pass the same
 *                     object for the successive calls.
 * @param[in]  theta   distribution of each quadrant at each level of resolution.
 *                     Since these are probabilities, each of the 2x2 matrices for
 *                     each level of the RMAT must sum to one. [on device]
 *                     [dim = max(r_scale, c_scale) x 2 x 2]. Of course, it is assumed
 *                     that each of the group of 2 x 2 numbers all sum up to 1.
 * @param[out] output  generated edgelist [on device]
 * @param[in]  r_scale 2^r_scale represents the number of source nodes
 * @param[in]  c_scale 2^c_scale represents the number of destination nodes
 *
 * We call the `r_scale != c_scale` case the "rectangular adjacency matrix" case (IOW generating
 * bipartite graphs). In this case, at `depth >= r_scale`, the distribution is assumed to be:
 * `[theta[4 * depth] + theta[4 * depth + 2], theta[4 * depth + 1] + theta[4 * depth + 3]; 0, 0]`.
 * Then for `depth >= c_scale`, the distribution is assumed to be:
 * `[theta[4 * depth] + theta[4 * depth + 1], 0; theta[4 * depth + 2] + theta[4 * depth + 3], 0]`.
 *
 * @note This can generate duplicate edges and self-loops. It is the responsibility of the
 *       caller to clean them up accordingly.

 * @note This also only generates directed graphs. If undirected graphs are needed, then a
 *       separate post-processing step is expected to be done by the caller.
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen(const raft::handle_t& handle,
                          raft::random::RngState& r,
                          raft::device_vector_view<const ProbT, IdxT> theta,
                          rmat_rectangular_gen_output<IdxT> output,
                          IdxT r_scale,
                          IdxT c_scale)
{
  static_assert(std::is_integral_v<IdxT>,
                "rmat_rectangular_gen: "
                "Template parameter IdxT must be an integral type");
  if (not output.has_value()) {
    return;  // nowhere to write output, so nothing to do
  }

  const IdxT expected_theta_len = IdxT(4) * (r_scale >= c_scale ? r_scale : c_scale);
  RAFT_EXPECTS(theta.extent(0) == expected_theta_len,
               "rmat_rectangular_gen: "
               "theta.extent(0) = %d != 2 * 2 * max(r_scale = %d, c_scale = %d) = %d",
               static_cast<int>(theta.extent(0)),
               static_cast<int>(r_scale),
               static_cast<int>(c_scale),
               static_cast<int>(expected_theta_len));

  auto out                     = output.out_view();
  auto out_src                 = output.out_src_view();
  auto out_dst                 = output.out_dst_view();
  const bool out_has_value     = out.has_value();
  const bool out_src_has_value = out_src.has_value();
  const bool out_dst_has_value = out_dst.has_value();
  IdxT* out_ptr                = out_has_value ? (*out).data_handle() : nullptr;
  IdxT* out_src_ptr            = out_src_has_value ? (*out_src).data_handle() : nullptr;
  IdxT* out_dst_ptr            = out_dst_has_value ? (*out_dst).data_handle() : nullptr;
  const IdxT n_edges           = output.number_of_edges();

  detail::rmat_rectangular_gen_caller(out_ptr,
                                      out_src_ptr,
                                      out_dst_ptr,
                                      theta.data_handle(),
                                      r_scale,
                                      c_scale,
                                      n_edges,
                                      handle.get_stream(),
                                      r);
}

/**
 * @brief Overload of `rmat_rectangular_gen` that assumes the same
 *   a, b, c, d probability distributions across all the scales.
 *
 * `a`, `b, and `c` effectively replace the above overload's
 * `theta` parameter.
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen(const raft::handle_t& handle,
                          raft::random::RngState& r,
                          rmat_rectangular_gen_output<IdxT> output,
                          ProbT a,
                          ProbT b,
                          ProbT c,
                          IdxT r_scale,
                          IdxT c_scale)
{
  static_assert(std::is_integral_v<IdxT>,
                "rmat_rectangular_gen: "
                "Template parameter IdxT must be an integral type");
  if (not output.has_value()) {
    return;  // nowhere to write output, so nothing to do
  }

  auto out                     = output.out_view();
  auto out_src                 = output.out_src_view();
  auto out_dst                 = output.out_dst_view();
  const bool out_has_value     = out.has_value();
  const bool out_src_has_value = out_src.has_value();
  const bool out_dst_has_value = out_dst.has_value();
  IdxT* out_ptr                = out_has_value ? (*out).data_handle() : nullptr;
  IdxT* out_src_ptr            = out_src_has_value ? (*out_src).data_handle() : nullptr;
  IdxT* out_dst_ptr            = out_dst_has_value ? (*out_dst).data_handle() : nullptr;
  const IdxT n_edges           = output.number_of_edges();

  detail::rmat_rectangular_gen_caller(
    out_ptr, out_src_ptr, out_dst_ptr, a, b, c, r_scale, c_scale, n_edges, handle.get_stream(), r);
}

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
