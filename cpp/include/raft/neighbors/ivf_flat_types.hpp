/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "ann_types.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/error.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/util/integer_utils.hpp>

#include <memory>
#include <optional>
#include <thrust/fill.h>
#include <type_traits>

namespace raft::neighbors::ivf_flat {
/**
 * @ingroup ivf_flat
 * @{
 */

/** Size of the interleaved group (see `index::data` description). */
constexpr static uint32_t kIndexGroupSize = 32;

/**
 * Default value filled in the `indices()` array.
 * One may encounter it trying to access a record within a cluster that is outside of the
 * `list_sizes()` bound (due to the record alignment `kIndexGroupSize`).
 */
template <typename IdxT>
constexpr static IdxT kInvalidRecord = std::numeric_limits<IdxT>::max() - 1;

struct index_params : ann::index_params {
  /** The number of inverted lists (clusters) */
  uint32_t n_lists = 1024;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters = 20;
  /** The fraction of data to use during iterative kmeans building. */
  double kmeans_trainset_fraction = 0.5;
  /**
   * By default (adaptive_centers = false), the cluster centers are trained in `ivf_flat::build`,
   * and never modified in `ivf_flat::extend`. As a result, you may need to retrain the index
   * from scratch after invoking (`ivf_flat::extend`) a few times with new data, the distribution of
   * which is no longer representative of the original training set.
   *
   * The alternative behavior (adaptive_centers = true) is to update the cluster centers for new
   * data when it is added. In this case, `index.centers()` are always exactly the centroids of the
   * data in the corresponding clusters. The drawback of this behavior is that the centroids depend
   * on the order of adding new data (through the classification of the added data); that is,
   * `index.centers()` "drift" together with the changing distribution of the newly added data.
   */
  bool adaptive_centers = false;
};

struct search_params : ann::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
};

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

/** The data for a single list (cluster). */
template <typename T, typename IdxT, typename SizeT = uint32_t>
struct list_data {
  /** Cluster data. */
  device_matrix<T, SizeT, row_major> data;
  /** Source indices. */
  device_vector<IdxT, SizeT> indices;
  /** The actual size of the content. */
  std::atomic<SizeT> size;

  list_data(raft::device_resources const& res, SizeT n_rows, uint32_t dim)
    : size{n_rows}
  {
    auto capacity = round_up_safe<SizeT>(n_rows, kIndexGroupSize);
    try {
      data = make_device_matrix<T, SizeT, row_major>(res, capacity, dim);
      indices = make_device_vector<IdxT, SizeT>(res, capacity);
    } catch (std::bad_alloc& e) {
      RAFT_FAIL(
        "ivf-flat: failed to allocate a big enough index list to hold all data "
        "(requested size: %zu records, selected capacity: %zu records). "
        "Allocator exception: %s",
        size_t(n_rows),
        size_t(capacity),
        e.what());
    }
    // Fill the index buffer with a pre-defined marker for easier debugging
    thrust::fill_n(
      res.get_thrust_policy(), indices.data_handle(), indices.size(), kInvalidRecord<IdxT>);
  }
};

/**
 * @brief IVF-flat index.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename T, typename IdxT>
struct index : ann::index {
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  /**
   * Vectorized load/store size in elements, determines the size of interleaved data chunks.
   *
   * TODO: in theory, we can lift this to the template parameter and keep it at hardware maximum
   * possible value by padding the `dim` of the data https://github.com/rapidsai/raft/issues/711
   */
  [[nodiscard]] constexpr inline auto veclen() const noexcept -> uint32_t { return veclen_; }
  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> raft::distance::DistanceType
  {
    return metric_;
  }
  /** Whether `centers()` change upon extending the index (ivf_pq::extend). */
  [[nodiscard]] constexpr inline auto adaptive_centers() const noexcept -> bool
  {
    return adaptive_centers_;
  }
  /**
   * Inverted list data [size, dim].
   *
   * The data consists of the dataset rows, grouped by their labels (into clusters/lists).
   * Within each list (cluster), the data is grouped into blocks of `kIndexGroupSize` interleaved
   * vectors. Note, the total index length is slightly larger than the source dataset length,
   * because each cluster is padded by `kIndexGroupSize` elements.
   *
   * Interleaving pattern:
   * within groups of `kIndexGroupSize` rows, the data is interleaved with the block size equal to
   * `veclen * sizeof(T)`. That is, a chunk of `veclen` consecutive components of one row is
   * followed by a chunk of the same size of the next row, and so on.
   *
   * __Example__: veclen = 2, dim = 6, kIndexGroupSize = 32, list_size = 31
   *
   *     x[ 0, 0], x[ 0, 1], x[ 1, 0], x[ 1, 1], ... x[14, 0], x[14, 1], x[15, 0], x[15, 1],
   *     x[16, 0], x[16, 1], x[17, 0], x[17, 1], ... x[30, 0], x[30, 1],    -    ,    -    ,
   *     x[ 0, 2], x[ 0, 3], x[ 1, 2], x[ 1, 3], ... x[14, 2], x[14, 3], x[15, 2], x[15, 3],
   *     x[16, 2], x[16, 3], x[17, 2], x[17, 3], ... x[30, 2], x[30, 3],    -    ,    -    ,
   *     x[ 0, 4], x[ 0, 5], x[ 1, 4], x[ 1, 5], ... x[14, 4], x[14, 5], x[15, 4], x[15, 5],
   *     x[16, 4], x[16, 5], x[17, 4], x[17, 5], ... x[30, 4], x[30, 5],    -    ,    -    ,
   *
   */
  /** Sizes of the lists (clusters) [n_lists] */
  inline auto list_sizes() noexcept -> device_vector_view<uint32_t, uint32_t>
  {
    return list_sizes_.view();
  }
  [[nodiscard]] inline auto list_sizes() const noexcept
    -> device_vector_view<const uint32_t, uint32_t>
  {
    return list_sizes_.view();
  }

  /** k-means cluster centers corresponding to the lists [n_lists, dim] */
  inline auto centers() noexcept -> device_matrix_view<float, uint32_t, row_major>
  {
    return centers_.view();
  }
  [[nodiscard]] inline auto centers() const noexcept
    -> device_matrix_view<const float, uint32_t, row_major>
  {
    return centers_.view();
  }

  /**
   * (Optional) Precomputed norms of the `centers` w.r.t. the chosen distance metric [n_lists].
   *
   * NB: this may be empty if the index is empty or if the metric does not require the center norms
   * calculation.
   */
  inline auto center_norms() noexcept
    -> std::optional<device_vector_view<float, uint32_t>>
  {
    if (center_norms_.has_value()) {
      return std::make_optional<device_vector_view<float, uint32_t>>(
        center_norms_->view());
    } else {
      return std::nullopt;
    }
  }
  [[nodiscard]] inline auto center_norms() const noexcept
    -> std::optional<device_vector_view<const float, uint32_t>>
  {
    if (center_norms_.has_value()) {
      return std::make_optional<device_vector_view<const float, uint32_t>>(
        center_norms_->view());
    } else {
      return std::nullopt;
    }
  }

  /** Total length of the index. */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    return total_size_;
  }
  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t
  {
    return centers_.extent(1);
  }
  /** Number of clusters/inverted lists. */
  [[nodiscard]] constexpr inline auto n_lists() const noexcept -> uint32_t
  {
    return lists_.extent(0);
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&) = delete;
  index(index&&)      = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index& = default;
  ~index()                          = default;

  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::device_resources const& res,
        raft::distance::DistanceType metric,
        uint32_t n_lists,
        bool adaptive_centers,
        uint32_t dim)
    : ann::index(),
      veclen_(calculate_veclen(dim)),
      metric_(metric),
      adaptive_centers_(adaptive_centers),
      centers_(make_device_matrix<float, uint32_t>(res, n_lists, dim)),
      center_norms_(std::nullopt),
      lists_{make_host_vector<std::shared_ptr<list_data<T, IdxT>>, uint32_t>(n_lists)},
      list_sizes_{make_device_vector<uint32_t, uint32_t>(res, n_lists)},
      data_ptrs_{make_device_vector<T*, uint32_t>(res, n_lists)},
      inds_ptrs_{make_device_vector<IdxT*, uint32_t>(res, n_lists)}
  {
    check_consistency();
    for (uint32_t i = 0; i < n_lists; i++) {
      lists_(i) = std::shared_ptr<list_data<T, IdxT>>();
    }
  }

  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::device_resources const& res, const index_params& params, uint32_t dim)
    : index(res, params.metric, params.n_lists, params.adaptive_centers, dim)
  {
  }

  /** Pointers to the inverted lists (clusters) data  [n_lists]. */
  inline auto data_ptrs() noexcept -> device_vector_view<T*, uint32_t>
  {
    return data_ptrs_.view();
  }
  [[nodiscard]] inline auto data_ptrs() const noexcept
    -> device_vector_view<T* const, uint32_t>
  {
    return data_ptrs_.view();
  }

  /** Pointers to the inverted lists (clusters) indices  [n_lists]. */
  inline auto inds_ptrs() noexcept -> device_vector_view<IdxT*, uint32_t>
  {
    return inds_ptrs_.view();
  }
  [[nodiscard]] inline auto inds_ptrs() const noexcept
    -> device_vector_view<IdxT* const, uint32_t>
  {
    return inds_ptrs_.view();
  }

  /**
   * Update the state of the dependent index members.
   */
  void recompute_internal_state(raft::device_resources const& res)
  {
    auto stream  = res.get_stream();

    // Actualize the list pointers
    auto this_lists     = lists();
    auto this_data_ptrs = data_ptrs();
    auto this_inds_ptrs = inds_ptrs();
    IdxT recompute_total_size = 0;
    for (uint32_t label = 0; label < this_lists.size(); label++) {
      const auto data_ptr = this_lists(label) ? this_lists(label)->data.data_handle() : nullptr;
      const auto inds_ptr = this_lists(label) ? this_lists(label)->indices.data_handle() : nullptr;
      const auto list_size = this_lists(label) ? IdxT(this_lists(label)->size) : 0;
      copy(&this_data_ptrs(label), &data_ptr, 1, stream);
      copy(&this_inds_ptrs(label), &inds_ptr, 1, stream);
      recompute_total_size += list_size;
    }
    total_size_ = recompute_total_size;
  }

  /** Lists' data and indices. */
  inline auto lists() noexcept
    -> host_vector_view<std::shared_ptr<list_data<T, IdxT>>, uint32_t>
  {
    return lists_.view();
  }
  [[nodiscard]] inline auto lists() const noexcept
    -> host_vector_view<const std::shared_ptr<list_data<T, IdxT>>, uint32_t>
  {
    return lists_.view();
  }

 private:
  /**
   * TODO: in theory, we can lift this to the template parameter and keep it at hardware maximum
   * possible value by padding the `dim` of the data https://github.com/rapidsai/raft/issues/711
   */
  uint32_t veclen_;
  raft::distance::DistanceType metric_;
  bool adaptive_centers_;
  host_vector<std::shared_ptr<list_data<T, IdxT>>, uint32_t> lists_;
  device_vector<uint32_t, uint32_t> list_sizes_;
  device_matrix<float, uint32_t, row_major> centers_;
  std::optional<device_vector<float, uint32_t>> center_norms_;

  // Computed members
  device_vector<T*, uint32_t> data_ptrs_;
  device_vector<IdxT*, uint32_t> inds_ptrs_;
  IdxT total_size_;

  /** Throw an error if the index content is inconsistent. */
  void check_consistency()
  {
    auto n_lists = lists_.extent(0);
    RAFT_EXPECTS(dim() % veclen_ == 0, "dimensionality is not a multiple of the veclen");
    RAFT_EXPECTS(list_sizes_.extent(0) == n_lists, "inconsistent list size");
    RAFT_EXPECTS(data_ptrs_.extent(0) == n_lists, "inconsistent list size");
    RAFT_EXPECTS(inds_ptrs_.extent(0) == n_lists, "inconsistent list size");
    RAFT_EXPECTS(                                               //
      (centers_.extent(0) == list_sizes_.extent(0)) &&          //
        (!center_norms_.has_value() || centers_.extent(0) == center_norms_->extent(0)),
      "inconsistent number of lists (clusters)");
  }

  static auto calculate_veclen(uint32_t dim) -> uint32_t
  {
    // TODO: consider padding the dimensions and fixing veclen to its maximum possible value as a
    // template parameter (https://github.com/rapidsai/raft/issues/711)
    uint32_t veclen = 16 / sizeof(T);
    while (dim % veclen != 0) {
      veclen = veclen >> 1;
    }
    return veclen;
  }
};

/** @} */

}  // namespace raft::neighbors::ivf_flat
