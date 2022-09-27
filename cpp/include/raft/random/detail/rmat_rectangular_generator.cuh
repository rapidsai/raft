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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_device.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <optional>
#include <variant>

namespace raft {
namespace random {
namespace detail {

template <typename IdxT, typename ProbT>
DI void gen_and_update_bits(IdxT& src_id,
                            IdxT& dst_id,
                            ProbT a,
                            ProbT ab,
                            ProbT abc,
                            IdxT r_scale,
                            IdxT c_scale,
                            IdxT curr_depth,
                            raft::random::PCGenerator& gen)
{
  bool src_bit, dst_bit;
  ProbT val;
  gen.next(val);
  if (val <= a) {
    src_bit = dst_bit = false;
  } else if (val <= ab) {
    src_bit = false;
    dst_bit = true;
  } else if (val <= abc) {
    src_bit = true;
    dst_bit = false;
  } else {
    src_bit = dst_bit = true;
  }
  if (curr_depth < r_scale) { src_id |= (IdxT(src_bit) << (r_scale - curr_depth - 1)); }
  if (curr_depth < c_scale) { dst_id |= (IdxT(dst_bit) << (c_scale - curr_depth - 1)); }
}

template <typename IdxT>
DI void store_ids(
  IdxT* out, IdxT* out_src, IdxT* out_dst, IdxT src_id, IdxT dst_id, IdxT idx, IdxT n_edges)
{
  if (idx < n_edges) {
    if (out != nullptr) {
      // uncoalesced gmem accesses!
      out[idx * 2]     = src_id;
      out[idx * 2 + 1] = dst_id;
    }
    if (out_src != nullptr) { out_src[idx] = src_id; }
    if (out_dst != nullptr) { out_dst[idx] = dst_id; }
  }
}

template <typename IdxT, typename ProbT>
__global__ void rmat_gen_kernel(IdxT* out,
                                IdxT* out_src,
                                IdxT* out_dst,
                                const ProbT* theta,
                                IdxT r_scale,
                                IdxT c_scale,
                                IdxT n_edges,
                                IdxT max_scale,
                                raft::random::RngState r)
{
  IdxT idx = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  extern __shared__ ProbT s_theta[];
  auto theta_len = max_scale * 2 * 2;
  // load the probabilities into shared memory and then convert them into cdf's
  // currently there are smem bank conflicts due to the way these are accessed
  for (int i = threadIdx.x; i < theta_len; i += blockDim.x) {
    s_theta[i] = theta[i];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < max_scale; i += blockDim.x) {
    auto a             = s_theta[4 * i];
    auto b             = s_theta[4 * i + 1];
    auto c             = s_theta[4 * i + 2];
    s_theta[4 * i + 1] = a + b;
    s_theta[4 * i + 2] = a + b + c;
    s_theta[4 * i + 3] += a + b + c;
  }
  __syncthreads();
  IdxT src_id{0}, dst_id{0};
  raft::random::PCGenerator gen{r.seed, r.base_subsequence + idx, 0};
  for (IdxT i = 0; i < max_scale; ++i) {
    auto a = s_theta[i * 4], ab = s_theta[i * 4 + 1], abc = s_theta[i * 4 + 2];
    gen_and_update_bits(src_id, dst_id, a, ab, abc, r_scale, c_scale, i, gen);
  }
  store_ids(out, out_src, out_dst, src_id, dst_id, idx, n_edges);
}

template <typename IdxT, typename ProbT>
void rmat_rectangular_gen_caller(IdxT* out,
                                 IdxT* out_src,
                                 IdxT* out_dst,
                                 const ProbT* theta,
                                 IdxT r_scale,
                                 IdxT c_scale,
                                 IdxT n_edges,
                                 cudaStream_t stream,
                                 raft::random::RngState& r)
{
  if (n_edges <= 0) return;
  static constexpr int N_THREADS = 512;
  auto max_scale                 = max(r_scale, c_scale);
  size_t smem_size               = sizeof(ProbT) * max_scale * 2 * 2;
  auto n_blks                    = raft::ceildiv<IdxT>(n_edges, N_THREADS);
  rmat_gen_kernel<<<n_blks, N_THREADS, smem_size, stream>>>(
    out, out_src, out_dst, theta, r_scale, c_scale, n_edges, max_scale, r);
  RAFT_CUDA_TRY(cudaGetLastError());
  r.advance(n_edges, max_scale);
}

template <typename IdxT, typename ProbT>
__global__ void rmat_gen_kernel(IdxT* out,
                                IdxT* out_src,
                                IdxT* out_dst,
                                ProbT a,
                                ProbT b,
                                ProbT c,
                                IdxT r_scale,
                                IdxT c_scale,
                                IdxT n_edges,
                                IdxT max_scale,
                                raft::random::RngState r)
{
  IdxT idx = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  IdxT src_id{0}, dst_id{0};
  raft::random::PCGenerator gen{r.seed, r.base_subsequence + idx, 0};
  auto min_scale = min(r_scale, c_scale);
  IdxT i         = 0;
  for (; i < min_scale; ++i) {
    gen_and_update_bits(src_id, dst_id, a, a + b, a + b + c, r_scale, c_scale, i, gen);
  }
  for (; i < r_scale; ++i) {
    gen_and_update_bits(src_id, dst_id, a + b, a + b, ProbT(1), r_scale, c_scale, i, gen);
  }
  for (; i < c_scale; ++i) {
    gen_and_update_bits(src_id, dst_id, a + c, ProbT(1), ProbT(1), r_scale, c_scale, i, gen);
  }
  store_ids(out, out_src, out_dst, src_id, dst_id, idx, n_edges);
}

template <typename IdxT, typename ProbT>
void rmat_rectangular_gen_caller(IdxT* out,
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
  if (n_edges <= 0) return;
  static constexpr int N_THREADS = 512;
  auto max_scale                 = max(r_scale, c_scale);
  auto n_blks                    = raft::ceildiv<IdxT>(n_edges, N_THREADS);
  rmat_gen_kernel<<<n_blks, N_THREADS, 0, stream>>>(
    out, out_src, out_dst, a, b, c, r_scale, c_scale, n_edges, max_scale, r);
  RAFT_CUDA_TRY(cudaGetLastError());
  r.advance(n_edges, max_scale);
}

/**
 * @brief Implementation detail for checking output vector parameter(s)
 *   of `raft::random::rmat_rectangular_gen`.
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
                   "out_src.extent(0) = %zu != out_dst.extent(0) = %zu",
                   static_cast<std::size_t>(src.extent(0)),
                   static_cast<std::size_t>(dst.extent(0)));
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
                   "out.extent(0) = %zu != 2 * out_dst.extent(0) = %zu",
                   static_cast<std::size_t>(out.extent(0)),
                   static_cast<std::size_t>(IdxT(2) * dst.extent(0)));
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
 * @brief Implementation of `raft::random::rmat_rectangular_gen_impl`.
 *
 * @tparam IdxT  type of each node index
 * @tparam ProbT data type used for probability distributions (either fp32 or fp64)
 * @param[in]  handle  RAFT handle, containing the CUDA stream on which to schedule work
 * @param[in]  r       underlying state of the random generator. Especially useful when
 *                     one wants to call this API for multiple times in order to generate
 *                     a larger graph. For that case, just create this object with the
 *                     initial seed once and after every call continue to pass the same
 *                     object for the successive calls.
 * @param[out] output  Encapsulation of one, two, or three output vectors.
 * @param[in]  theta   distribution of each quadrant at each level of resolution.
 *                     Since these are probabilities, each of the 2x2 matrices for
 *                     each level of the RMAT must sum to one. [on device]
 *                     [dim = max(r_scale, c_scale) x 2 x 2]. Of course, it is assumed
 *                     that each of the group of 2 x 2 numbers all sum up to 1.
 * @param[in]  r_scale 2^r_scale represents the number of source nodes
 * @param[in]  c_scale 2^c_scale represents the number of destination nodes
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen_impl(const raft::handle_t& handle,
                               raft::random::RngState& r,
                               raft::device_vector_view<const ProbT, IdxT> theta,
                               raft::random::detail::rmat_rectangular_gen_output<IdxT> output,
                               IdxT r_scale,
                               IdxT c_scale)
{
  static_assert(std::is_integral_v<IdxT>,
                "rmat_rectangular_gen: "
                "Template parameter IdxT must be an integral type");
  if (not output.has_value()) {
    return;  // nothing to do; not an error
  }

  const IdxT expected_theta_len = IdxT(4) * (r_scale >= c_scale ? r_scale : c_scale);
  RAFT_EXPECTS(theta.extent(0) == expected_theta_len,
               "rmat_rectangular_gen: "
               "theta.extent(0) = %zu != 2 * 2 * max(r_scale = %zu, c_scale = %zu) = %zu",
               static_cast<std::size_t>(theta.extent(0)),
               static_cast<std::size_t>(r_scale),
               static_cast<std::size_t>(c_scale),
               static_cast<std::size_t>(expected_theta_len));

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

  rmat_rectangular_gen_caller(out_ptr,
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
void rmat_rectangular_gen_impl(const raft::handle_t& handle,
                               raft::random::RngState& r,
                               raft::random::detail::rmat_rectangular_gen_output<IdxT> output,
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

}  // end namespace detail
}  // end namespace random
}  // end namespace raft
