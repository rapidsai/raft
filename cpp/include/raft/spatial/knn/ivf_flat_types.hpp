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

#include "ann_common.hpp"

#include <raft/core/mdarray.hpp>
#include <raft/distance/distance_type.hpp>
#include <raft/integer_utils.h>

#include <optional>

namespace raft::spatial::knn::ivf_flat {

/** Size of the interleaved group (see `index::data` description). */
constexpr static uint32_t kIndexGroupSize = 32;

/**
 * @brief IVF-flat index.
 *
 * This structure is supposed to be immutable: it's only constructed using `ivf_flat::build`,
 * and should never be modified.
 * At the same time, we expose all its members and allow the aggregate construction, so that
 * third-party users can implement custom serialization/deserialization routines or modify
 * the index building process.
 *
 * It would seem logical to make all the type's members constant. However, we can't do that
 * because it would imply copying data when the index is moved. The current solution to this
 * is to make all public factory functions, such as `ivf_flat::build` return `const index`.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename T, typename IdxT>
struct index {
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");
  /**
   * Vectorized load/store size in elements, determines the size of interleaved data chunks.
   *
   * TODO: in theory, we can lift this to the template parameter and keep it at hardware maximum
   * possible value by padding the `dim` of the data https://github.com/rapidsai/raft/issues/711
   */
  const uint32_t veclen;
  /** Distance metric used for clustering. */
  const raft::distance::DistanceType metric;

  /**
   * Inverted list data [size, dim].
   *
   * The data consists of the dataset rows, grouped by their labels (into clusters/lists).
   * Within each list (cluster), the data is grouped into blocks of `kGroupSize` interleaved
   * vectors. Note, the total index length is slightly larger than the source dataset length,
   * because each cluster is padded by `kGroupSize` elements.
   *
   * Interleaving pattern:
   * within groups of `kGroupSize` rows, the data is interleaved with the block size equal to
   * `veclen * sizeof(T)`. That is, a chunk of `veclen` consecutive components of one row is
   * followed by a chunk of the same size of the next row, and so on.
   *
   * __Example__: veclen = 2, dim = 6, kGroupSize = 32, list_size = 31
   * `
   *   x[ 0, 0], x[ 0, 1], x[ 1, 0], x[ 1, 1], ... x[14, 0], x[14, 1], x[15, 0], x[15, 1],
   *   x[16, 0], x[16, 1], x[17, 0], x[17, 1], ... x[30, 0], x[30, 1],    -    ,    -    ,
   *   x[ 0, 2], x[ 0, 3], x[ 1, 2], x[ 1, 3], ... x[14, 2], x[14, 3], x[15, 2], x[15, 3],
   *   x[16, 2], x[16, 3], x[17, 2], x[17, 3], ... x[30, 2], x[30, 3],    -    ,    -    ,
   *   x[ 0, 4], x[ 0, 5], x[ 1, 4], x[ 1, 5], ... x[14, 4], x[14, 5], x[15, 4], x[15, 5],
   *   x[16, 4], x[16, 5], x[17, 4], x[17, 5], ... x[30, 4], x[30, 5],    -    ,    -    ,
   * `
   */
  device_mdarray<T, extent_2d, row_major> data;
  /** Inverted list indices: ids of items in the source data [size] */
  device_mdarray<IdxT, extent_1d, row_major> indices;
  /** Sizes of the lists (clusters) [n_lists] */
  device_mdarray<uint32_t, extent_1d, row_major> list_sizes;
  /**
   * Offsets into the lists [n_lists + 1].
   * The last value contains the total length of the index.
   */
  device_mdarray<IdxT, extent_1d, row_major> list_offsets;
  /** k-means cluster centers corresponding to the lists [n_lists, dim] */
  device_mdarray<float, extent_2d, row_major> centers;
  /** (Optional) Precomputed norms of the `centers` w.r.t. the chosen distance metric [n_lists]  */
  std::optional<device_mdarray<float, extent_1d, row_major>> center_norms;

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&) = delete;
  index(index&&)      = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index& = default;
  ~index()                          = default;

  /** Total length of the index. */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    return static_cast<uint32_t>(data.extent(0));
  }
  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(data.extent(1));
  }
  /** Number of clusters/inverted lists. */
  [[nodiscard]] constexpr inline auto n_lists() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centers.extent(0));
  }

  /** Throw an error if the index content is inconsistent. */
  inline void check_consistency() const
  {
    RAFT_EXPECTS(dim() % veclen == 0, "dimensionality is not a multiple of the veclen");
    RAFT_EXPECTS(data.extent(0) == indices.extent(0), "inconsistent index size");
    RAFT_EXPECTS(data.extent(1) == centers.extent(1), "inconsistent data dimensionality");
    RAFT_EXPECTS(                                             //
      (centers.extent(0) == list_sizes.extent(0)) &&          //
        (centers.extent(0) + 1 == list_offsets.extent(0)) &&  //
        (!center_norms.has_value() || centers.extent(0) == center_norms->extent(0)),
      "inconsistent number of lists (clusters)");
    RAFT_EXPECTS(reinterpret_cast<size_t>(data.data()) % (veclen * sizeof(T)) == 0,
                 "The data storage pointer is not aligned to the vector length");
  }
};

struct index_params : ivf_index_params {
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters = 20;
  /** The fraction of data to use during iterative kmeans building. */
  double kmeans_trainset_fraction = 0.5;
};

struct search_params : ivf_search_params {
};

static_assert(std::is_standard_layout_v<index<float, uint32_t>>);
static_assert(std::is_aggregate_v<index<float, uint32_t>>);

}  // namespace raft::spatial::knn::ivf_flat
