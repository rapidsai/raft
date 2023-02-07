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
#include <raft/core/error.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/util/integer_utils.hpp>

#include <optional>
#include <type_traits>

namespace raft::neighbors::ivf_flat {
/**
 * @ingroup ivf_flat
 * @{
 */

/** Size of the interleaved group (see `index::data` description). */
constexpr static uint32_t kIndexGroupSize = 32;

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
template <typename T, typename IdxT = uint32_t>
struct list_data {
  /** Cluster data. */
  device_mdarray<T, extent_2d<IdxT>, row_major> data;
  /** Source indices. */
  device_mdarray<IdxT, extent_1d<IdxT>, row_major> indices;
  /** The actual size of the content. */
  std::atomic<IdxT> size;

  list_data(raft::device_resources const& handle, IdxT n_rows, uint32_t dim)
    : size{n_rows}
  {
    auto capacity = round_up_safe<IdxT>(bound_by_power_of_two<IdxT>(size), kIndexGroupSize);
    try {
      data = make_device_mdarray<T>(handle, make_extents<IdxT>(capacity, dim));
      indices = make_device_mdarray<IdxT>(handle, make_extents<IdxT>(capacity));
    } catch (std::bad_alloc& e) {
      RAFT_FAIL(
        "ivf-flat: failed to allocate a big enough index list to hold all data "
        "(requested size: %zu records, selected capacity: %zu records). "
        "Allocator exception: %s",
        size_t(size),
        size_t(capacity),
        e.what());
    }
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
   * Default value filled in the `indices()` array.
   * One may encounter it trying to access a record within a cluster that is outside of the
   * `list_sizes()` bound (due to the record alignment `kIndexGroupSize`).
   */
  constexpr static IdxT kInvalidRecord = std::numeric_limits<IdxT>::max() - 1;
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
  inline auto data() noexcept -> device_mdspan<T, extent_2d<IdxT>, row_major>
  {
    return data_.view();
  }
  [[nodiscard]] inline auto data() const noexcept
    -> device_mdspan<const T, extent_2d<size_t>, row_major>
  {
    return data_.view();
  }

  /** Inverted list indices: ids of items in the source data [size] */
  inline auto indices() noexcept -> device_mdspan<IdxT, extent_1d<IdxT>, row_major>
  {
    return indices_.view();
  }
  [[nodiscard]] inline auto indices() const noexcept
    -> device_mdspan<const IdxT, extent_1d<IdxT>, row_major>
  {
    return indices_.view();
  }

  /** Sizes of the lists (clusters) [n_lists] */
  inline auto list_sizes() noexcept -> device_mdspan<uint32_t, extent_1d<uint32_t>, row_major>
  {
    return list_sizes_.view();
  }
  [[nodiscard]] inline auto list_sizes() const noexcept
    -> device_mdspan<const uint32_t, extent_1d<uint32_t>, row_major>
  {
    return list_sizes_.view();
  }

  /**
   * Offsets into the lists [n_lists + 1].
   * The last value contains the total length of the index.
   */
  inline auto list_offsets() noexcept -> device_mdspan<IdxT, extent_1d<uint32_t>, row_major>
  {
    return list_offsets_.view();
  }
  [[nodiscard]] inline auto list_offsets() const noexcept
    -> device_mdspan<const IdxT, extent_1d<uint32_t>, row_major>
  {
    return list_offsets_.view();
  }

  /** k-means cluster centers corresponding to the lists [n_lists, dim] */
  inline auto centers() noexcept -> device_mdspan<float, extent_2d<uint32_t>, row_major>
  {
    return centers_.view();
  }
  [[nodiscard]] inline auto centers() const noexcept
    -> device_mdspan<const float, extent_2d<uint32_t>, row_major>
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
    -> std::optional<device_mdspan<float, extent_1d<uint32_t>, row_major>>
  {
    if (center_norms_.has_value()) {
      return std::make_optional<device_mdspan<float, extent_1d<uint32_t>, row_major>>(
        center_norms_->view());
    } else {
      return std::nullopt;
    }
  }
  [[nodiscard]] inline auto center_norms() const noexcept
    -> std::optional<device_mdspan<const float, extent_1d<uint32_t>, row_major>>
  {
    if (center_norms_.has_value()) {
      return std::make_optional<device_mdspan<const float, extent_1d<uint32_t>, row_major>>(
        center_norms_->view());
    } else {
      return std::nullopt;
    }
  }

  /** Total length of the index. */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT { return indices_.extent(0); }
  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t
  {
    return centers_.extent(1);
  }
  /** Number of clusters/inverted lists. */
  [[nodiscard]] constexpr inline auto n_lists() const noexcept -> uint32_t
  {
    return centers_.extent(0);
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&) = delete;
  index(index&&)      = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index& = default;
  ~index()                          = default;

  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::device_resources const& handle,
        raft::distance::DistanceType metric,
        uint32_t n_lists,
        bool adaptive_centers,
        uint32_t dim)
    : ann::index(),
      veclen_(calculate_veclen(dim)),
      metric_(metric),
      adaptive_centers_(adaptive_centers),
      data_(make_device_mdarray<T>(handle, make_extents<IdxT>(0, dim))),
      indices_(make_device_mdarray<IdxT>(handle, make_extents<IdxT>(0))),
      list_sizes_(make_device_mdarray<uint32_t>(handle, make_extents<uint32_t>(n_lists))),
      list_offsets_(make_device_mdarray<IdxT>(handle, make_extents<uint32_t>(n_lists + 1))),
      centers_(make_device_mdarray<float>(handle, make_extents<uint32_t>(n_lists, dim))),
      center_norms_(std::nullopt)
  {
    check_consistency();
  }

  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::device_resources const& handle, const index_params& params, uint32_t dim)
    : index(handle, params.metric, params.n_lists, params.adaptive_centers, dim)
  {
  }

  /**
   * Replace the content of the index with new uninitialized mdarrays to hold the indicated amount
   * of data.
   */
  void resize_list(raft::device_resources const& handle, uint32_t label, uint32_t list_size)
  {
    uint32_t prev_size = 0;
    auto& list         = lists()(label);
    bool skip_resize   = false;
    if (list) {
      copy(&prev_size, &list_sizes()(label), 1, handle.get_stream());
      handle.sync_stream();
      if (list_size <= list->indices.extent(0)) {
        auto shared_list_size = prev_size;
        if (list_size <= prev_size ||
            list->size.compare_exchange_strong(shared_list_size, list_size)) {
          // We don't need to resize the list if:
          //  1. The list exists
          //  2. The new size fits in the list
          //  3. The list doesn't grow or no-one else has grown it yet
          skip_resize = true;
        }
      }
    }
    // Sic! We're writing the min(list_size, prev_size)
    //       to keep the number of _valid_ records after update
    const auto new_list_size = std::min(list_size, prev_size);
    raft::copy(&list_sizes()(label), &new_list_size, 1, handle.get_stream());
    if (skip_resize) { return; }
    auto new_list = new list_data<T, IdxT>(handle, list_size, dim());
    if (prev_size > 0) {
      auto copied_data_extents = make_extents<size_t>(prev_size, dim());
      auto copied_view         = make_mdspan<T, size_t, row_major, false, true>(
        new_list->data.data_handle(), copied_data_extents);
      copy(copied_view.data_handle(),
           list->data.data_handle(),
           copied_view.size(),
           handle.get_stream());
      copy(new_list->indices.data_handle(),
           list->indices.data_handle(),
           prev_size,
           handle.get_stream());
    }
    // swap the shared pointer content with the new list
    list.reset(new_list);
    // fill unused index spaces with placeholder values for easier debugging
    thrust::fill_n(handle.get_thrust_policy(),
                   list->indices.data_handle() + prev_size,
                   list->indices.size() - prev_size,
                   kInvalidRecord);
    // keep the device pointers updated
    const auto new_data_ptr = list->data.data_handle();
    copy(&(data_ptrs()(label)), &new_data_ptr, 1, handle.get_stream());
    const auto new_inds_ptr = list->indices.data_handle();
    copy(&(inds_ptrs()(label)), &new_inds_ptr, 1, handle.get_stream());



    data_    = make_device_mdarray<T>(handle, make_extents<IdxT>(index_size, dim()));
    indices_ = make_device_mdarray<IdxT>(handle, make_extents<IdxT>(index_size));

    switch (metric_) {
      case raft::distance::DistanceType::L2Expanded:
      case raft::distance::DistanceType::L2SqrtExpanded:
      case raft::distance::DistanceType::L2Unexpanded:
      case raft::distance::DistanceType::L2SqrtUnexpanded:
        center_norms_ = make_device_mdarray<float>(handle, make_extents<uint32_t>(n_lists()));
        break;
      default: center_norms_ = std::nullopt;
    }

    check_consistency();
  }

 private:
  /**
   * TODO: in theory, we can lift this to the template parameter and keep it at hardware maximum
   * possible value by padding the `dim` of the data https://github.com/rapidsai/raft/issues/711
   */
  uint32_t veclen_;
  raft::distance::DistanceType metric_;
  bool adaptive_centers_;
  host_mdarray<std::shared_ptr<list_data<T, IdxT>, extent_1d<uint32_t>, row_major> lists_;
  //device_mdarray<T, extent_2d<IdxT>, row_major> data_;
  //device_mdarray<IdxT, extent_1d<IdxT>, row_major> indices_;
  //device_mdarray<uint32_t, extent_1d<uint32_t>, row_major> list_sizes_;
  //device_mdarray<IdxT, extent_1d<uint32_t>, row_major> list_offsets_;
  device_mdarray<float, extent_2d<uint32_t>, row_major> centers_;
  std::optional<device_mdarray<float, extent_1d<uint32_t>, row_major>> center_norms_;

  // Computed members
  device_mdarray<T*, extent_1d<uint32_t>, row_major> data_ptrs_;
  device_mdarray<IdxT*, extent_1d<uint32_t>, row_major> inds_ptrs_;
  host_mdarray<IdxT, extent_1d<uint32_t>, row_major> accum_sorted_sizes_;

  /** Throw an error if the index content is inconsistent. */
  void check_consistency()
  {
    RAFT_EXPECTS(dim() % veclen_ == 0, "dimensionality is not a multiple of the veclen");
    RAFT_EXPECTS(data_.extent(0) == indices_.extent(0), "inconsistent index size");
    RAFT_EXPECTS(data_.extent(1) == IdxT(centers_.extent(1)), "inconsistent data dimensionality");
    RAFT_EXPECTS(                                               //
      (centers_.extent(0) == list_sizes_.extent(0)) &&          //
        (centers_.extent(0) + 1 == list_offsets_.extent(0)) &&  //
        (!center_norms_.has_value() || centers_.extent(0) == center_norms_->extent(0)),
      "inconsistent number of lists (clusters)");
    RAFT_EXPECTS(reinterpret_cast<size_t>(data_.data_handle()) % (veclen_ * sizeof(T)) == 0,
                 "The data storage pointer is not aligned to the vector length");
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
