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
#include <raft/random/rng_device.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <optional>
#include <variant>

namespace raft {
namespace random {
namespace detail {

/**
 * @brief Implementation detail for checking output vector parameter(s)
 *   of `raft::random::rmat_rectangular_gen`.
 *
 * `raft::random::rmat_rectangular_gen` lets users specify
 * output vector(s) in three different ways.
 *
 * 1. One vector: `out`, an "array-of-structs" representation
 *    of the edge list.
 *
 * 2. Two vectors: `out_src` and `out_dst`, together forming
 *    a "struct of arrays" representation of the edge list.
 *
 * 3. Three vectors: `out`, `out_src`, and `out_dst`.
 *    `out` is as in (1),
 *    and `out_src` and `out_dst` are as in (2).
 *
 * This class prevents users from doing anything other than that,
 * and makes it easier for the three cases to share a common implementation.
 * It also prevents duplication of run-time vector length checking
 * (`out` must have twice the number of elements as `out_src` and `out_dst`,
 * and `out_src` and `out_dst` must have the same length).
 *
 * @tparam IdxT Type of each node index; must be integral.
 *
 * The following examples show how to create an output parameter.
 *
 * @code
 * rmat_rectangular_gen_output<IdxT> output1(out);
 * rmat_rectangular_gen_output<IdxT> output2(out_src, out_dst);
 * rmat_rectangular_gen_output<IdxT> output3(out, out_src, out_dst);
 * @endcode
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

    bool empty() const { return src_.extent(0) == 0 && dst_.extent(0) == 0; }

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

    bool empty() const { return out_.extent(0) == 0 && pair_.empty(); }

   private:
    out_view_type out_;
    output_pair pair_;
  };

 public:
  /**
   * @brief You're not allowed to construct this with no vectors.
   */
  rmat_rectangular_gen_output() = delete;

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
   * @brief Whether the vector(s) are all length zero.
   */
  bool empty() const
  {
    if (std::holds_alternative<out_view_type>(data_)) {
      return std::get<out_view_type>(data_).extent(0) == 0;
    } else if (std::holds_alternative<output_pair>(data_)) {
      return std::get<output_pair>(data_).empty();
    } else {  // std::holds_alternative<output_triple>(data_)
      return std::get<output_triple>(data_).empty();
    }
  }

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
    } else {  // if (std::holds_alternative<>(output_pair))
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
    } else {  // if (std::holds_alternative<out_view_type>(data_))
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
    } else {  // if (std::holds_alternative<out_view_type>(data_))
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
    } else {  // if (std::holds_alternative<output_triple>(data_))
      return std::get<output_triple>(data_).number_of_edges();
    }
  }

 private:
  std::variant<out_view_type, output_pair, output_triple> data_;
};

}  // end namespace detail
}  // end namespace random
}  // end namespace raft
