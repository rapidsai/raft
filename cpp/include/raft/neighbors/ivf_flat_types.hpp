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

#include "ann_types.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/ivf_list_types.hpp>
#include <raft/util/integer_utils.hpp>

#include <algorithm>  // std::max
#include <memory>
#include <optional>
#include <type_traits>

namespace raft::neighbors::ivf_flat {
/**
 * @addtogroup ivf_flat
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
  /**
   * By default, the algorithm allocates more space than necessary for individual clusters
   * (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of
   * data copies during repeated calls to `extend` (extending the database).
   *
   * The alternative is the conservative allocation behavior; when enabled, the algorithm always
   * allocates the minimum amount of memory required to store the given number of records. Set this
   * flag to `true` if you prefer to use as little GPU memory for the database as possible.
   */
  bool conservative_memory_allocation = false;
};

struct search_params : ann::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
};

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

template <typename SizeT, typename ValueT, typename IdxT>
struct list_spec {
  using value_type   = ValueT;
  using list_extents = matrix_extent<SizeT>;
  using index_type   = IdxT;

  SizeT align_max;
  SizeT align_min;
  uint32_t dim;

  constexpr list_spec(uint32_t dim, bool conservative_memory_allocation)
    : dim(dim),
      align_min(kIndexGroupSize),
      align_max(conservative_memory_allocation ? kIndexGroupSize : 1024)
  {
  }

  // Allow casting between different size-types (for safer size and offset calculations)
  template <typename OtherSizeT>
  constexpr explicit list_spec(const list_spec<OtherSizeT, ValueT, IdxT>& other_spec)
    : dim{other_spec.dim}, align_min{other_spec.align_min}, align_max{other_spec.align_max}
  {
  }

  /** Determine the extents of an array enough to hold a given amount of data. */
  constexpr auto make_list_extents(SizeT n_rows) const -> list_extents
  {
    return make_extents<SizeT>(n_rows, dim);
  }
};

template <typename ValueT, typename IdxT, typename SizeT = uint32_t>
using list_data = ivf::list<list_spec, SizeT, ValueT, IdxT>;

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
  /** Sizes of the lists (clusters) [n_lists]
   * NB: This may differ from the actual list size if the shared lists have been extended by another
   * index
   */
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
  inline auto center_norms() noexcept -> std::optional<device_vector_view<float, uint32_t>>
  {
    if (center_norms_.has_value()) {
      return std::make_optional<device_vector_view<float, uint32_t>>(center_norms_->view());
    } else {
      return std::nullopt;
    }
  }
  [[nodiscard]] inline auto center_norms() const noexcept
    -> std::optional<device_vector_view<const float, uint32_t>>
  {
    if (center_norms_.has_value()) {
      return std::make_optional<device_vector_view<const float, uint32_t>>(center_norms_->view());
    } else {
      return std::nullopt;
    }
  }

  /**
   * Accumulated list sizes, sorted in descending order [n_lists + 1].
   * The last value contains the total length of the index.
   * The value at index zero is always zero.
   *
   * That is, the content of this span is as if the `list_sizes` was sorted and then accumulated.
   *
   * This span is used during search to estimate the maximum size of the workspace.
   */
  inline auto accum_sorted_sizes() noexcept -> host_vector_view<IdxT, uint32_t>
  {
    return accum_sorted_sizes_.view();
  }
  [[nodiscard]] inline auto accum_sorted_sizes() const noexcept
    -> host_vector_view<const IdxT, uint32_t>
  {
    return accum_sorted_sizes_.view();
  }

  /** Total length of the index. */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    return accum_sorted_sizes()(n_lists());
  }

  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t
  {
    return centers_.extent(1);
  }
  /** Number of clusters/inverted lists. */
  [[nodiscard]] constexpr inline auto n_lists() const noexcept -> uint32_t { return lists_.size(); }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::resources const& res,
        raft::distance::DistanceType metric,
        uint32_t n_lists,
        bool adaptive_centers,
        bool conservative_memory_allocation,
        uint32_t dim)
    : ann::index(),
      veclen_(calculate_veclen(dim)),
      metric_(metric),
      adaptive_centers_(adaptive_centers),
      conservative_memory_allocation_{conservative_memory_allocation},
      lists_{n_lists},
      list_sizes_{make_device_vector<uint32_t, uint32_t>(res, n_lists)},
      centers_(make_device_matrix<float, uint32_t>(res, n_lists, dim)),
      center_norms_(std::nullopt),
      data_ptrs_{make_device_vector<T*, uint32_t>(res, n_lists)},
      inds_ptrs_{make_device_vector<IdxT*, uint32_t>(res, n_lists)},
      accum_sorted_sizes_{make_host_vector<IdxT, uint32_t>(n_lists + 1)}
  {
    check_consistency();
    accum_sorted_sizes_(n_lists) = 0;
  }

  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::resources const& res, const index_params& params, uint32_t dim)
    : index(res,
            params.metric,
            params.n_lists,
            params.adaptive_centers,
            params.conservative_memory_allocation,
            dim)
  {
  }

  /** Pointers to the inverted lists (clusters) data  [n_lists]. */
  inline auto data_ptrs() noexcept -> device_vector_view<T*, uint32_t> { return data_ptrs_.view(); }
  [[nodiscard]] inline auto data_ptrs() const noexcept -> device_vector_view<T* const, uint32_t>
  {
    return data_ptrs_.view();
  }

  /** Pointers to the inverted lists (clusters) indices  [n_lists]. */
  inline auto inds_ptrs() noexcept -> device_vector_view<IdxT*, uint32_t>
  {
    return inds_ptrs_.view();
  }
  [[nodiscard]] inline auto inds_ptrs() const noexcept -> device_vector_view<IdxT* const, uint32_t>
  {
    return inds_ptrs_.view();
  }
  /**
   * Whether to use convervative memory allocation when extending the list (cluster) data
   * (see index_params.conservative_memory_allocation).
   */
  [[nodiscard]] constexpr inline auto conservative_memory_allocation() const noexcept -> bool
  {
    return conservative_memory_allocation_;
  }

  void allocate_center_norms(raft::resources const& res)
  {
    switch (metric_) {
      case raft::distance::DistanceType::L2Expanded:
      case raft::distance::DistanceType::L2SqrtExpanded:
      case raft::distance::DistanceType::L2Unexpanded:
      case raft::distance::DistanceType::L2SqrtUnexpanded:
        center_norms_ = make_device_vector<float, uint32_t>(res, n_lists());
        break;
      default: center_norms_ = std::nullopt;
    }
  }

  /** Lists' data and indices. */
  inline auto lists() noexcept -> std::vector<std::shared_ptr<list_data<T, IdxT>>>&
  {
    return lists_;
  }
  [[nodiscard]] inline auto lists() const noexcept
    -> const std::vector<std::shared_ptr<list_data<T, IdxT>>>&
  {
    return lists_;
  }

  /** Throw an error if the index content is inconsistent. */
  void check_consistency()
  {
    auto n_lists = lists_.size();
    RAFT_EXPECTS(dim() % veclen_ == 0, "dimensionality is not a multiple of the veclen");
    RAFT_EXPECTS(list_sizes_.extent(0) == n_lists, "inconsistent list size");
    RAFT_EXPECTS(data_ptrs_.extent(0) == n_lists, "inconsistent list size");
    RAFT_EXPECTS(inds_ptrs_.extent(0) == n_lists, "inconsistent list size");
    RAFT_EXPECTS(                                       //
      (centers_.extent(0) == list_sizes_.extent(0)) &&  //
        (!center_norms_.has_value() || centers_.extent(0) == center_norms_->extent(0)),
      "inconsistent number of lists (clusters)");
  }

 private:
  /**
   * TODO: in theory, we can lift this to the template parameter and keep it at hardware maximum
   * possible value by padding the `dim` of the data https://github.com/rapidsai/raft/issues/711
   */
  uint32_t veclen_;
  raft::distance::DistanceType metric_;
  bool adaptive_centers_;
  bool conservative_memory_allocation_;
  std::vector<std::shared_ptr<list_data<T, IdxT>>> lists_;
  device_vector<uint32_t, uint32_t> list_sizes_;
  device_matrix<float, uint32_t, row_major> centers_;
  std::optional<device_vector<float, uint32_t>> center_norms_;

  // Computed members
  device_vector<T*, uint32_t> data_ptrs_;
  device_vector<IdxT*, uint32_t> inds_ptrs_;
  host_vector<IdxT, uint32_t> accum_sorted_sizes_;

  static auto calculate_veclen(uint32_t dim) -> uint32_t
  {
    // TODO: consider padding the dimensions and fixing veclen to its maximum possible value as a
    // template parameter (https://github.com/rapidsai/raft/issues/711)

    // NOTE: keep this consistent with the select_interleaved_scan_kernel logic
    // in detail/ivf_flat_interleaved_scan-inl.cuh.
    uint32_t veclen = std::max<uint32_t>(1, 16 / sizeof(T));
    if (dim % veclen != 0) { veclen = 1; }
    return veclen;
  }
};

/** @} */

}  // namespace raft::neighbors::ivf_flat
