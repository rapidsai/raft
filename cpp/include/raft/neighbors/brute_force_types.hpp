/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <raft/core/resource/cuda_stream.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>

#include <raft/core/logger.hpp>

namespace raft::neighbors::brute_force {
/**
 * @addtogroup brute_force_knn
 * @{
 */

/**
 * @brief Brute Force index.
 *
 * The index stores the dataset and norms for the dataset in device memory.
 *
 * @tparam T data element type
 */
template <typename T>
struct index : ann::index {
 public:
  /** Distance metric used for retrieval */
  [[nodiscard]] constexpr inline raft::distance::DistanceType metric() const noexcept
  {
    return metric_;
  }

  /** Total length of the index (number of vectors). */
  [[nodiscard]] constexpr inline int64_t size() const noexcept { return dataset_view_.extent(0); }

  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline uint32_t dim() const noexcept { return dataset_view_.extent(1); }

  /** Dataset [size, dim] */
  [[nodiscard]] inline auto dataset() const noexcept
    -> device_matrix_view<const T, int64_t, row_major>
  {
    return dataset_view_;
  }

  /** Dataset norms */
  [[nodiscard]] inline auto norms() const -> device_vector_view<const T, int64_t, row_major>
  {
    return norms_view_.value();
  }

  /** Whether or not this index has dataset norms */
  [[nodiscard]] inline bool has_norms() const noexcept { return norms_view_.has_value(); }

  [[nodiscard]] inline T metric_arg() const noexcept { return metric_arg_; }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  /** Construct a brute force index from dataset
   *
   * Constructs a brute force index from a dataset. This lets us precompute norms for
   * the dataset, providing a speed benefit over doing this at query time.

   * If the dataset is already in GPU memory, then this class stores a non-owning reference to
   * the dataset. If the dataset is in host memory, it will be copied to the device and the
   * index will own the device memory.
   */
  template <typename data_accessor>
  index(raft::resources const& res,
        mdspan<const T, matrix_extent<int64_t>, row_major, data_accessor> dataset,
        std::optional<raft::device_vector<T, int64_t>>&& norms,
        raft::distance::DistanceType metric,
        T metric_arg = 0.0)
    : ann::index(),
      metric_(metric),
      dataset_(make_device_matrix<T, int64_t>(res, 0, 0)),
      norms_(std::move(norms)),
      metric_arg_(metric_arg)
  {
    if (norms_) { norms_view_ = make_const_mdspan(norms_.value().view()); }
    update_dataset(res, dataset);
    resource::sync_stream(res);
  }

  /** Construct a brute force index from dataset
   *
   * This class stores a non-owning reference to the dataset and norms here.
   * Having precomputed norms gives us a performance advantage at query time.
   */
  index(raft::resources const& res,
        raft::device_matrix_view<const T, int64_t, row_major> dataset_view,
        std::optional<raft::device_vector_view<const T, int64_t>> norms_view,
        raft::distance::DistanceType metric,
        T metric_arg = 0.0)
    : ann::index(),
      metric_(metric),
      dataset_(make_device_matrix<T, int64_t>(res, 0, 0)),
      dataset_view_(dataset_view),
      norms_view_(norms_view),
      metric_arg_(metric_arg)
  {
  }

 private:
  /**
   * Replace the dataset with a new dataset.
   */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, row_major> dataset)
  {
    dataset_view_ = dataset;
  }

  /**
   * Replace the dataset with a new dataset.
   *
   * We create a copy of the dataset on the device. The index manages the lifetime of this copy.
   */
  void update_dataset(raft::resources const& res,
                      raft::host_matrix_view<const T, int64_t, row_major> dataset)
  {
    dataset_ = make_device_matrix<T, int64_t>(dataset.extents(0), dataset.extents(1));
    raft::copy(dataset_.data_handle(),
               dataset.data_handle(),
               dataset.size(),
               resource::get_cuda_stream(res));
    dataset_view_ = make_const_mdspan(dataset_.view());
  }

  raft::distance::DistanceType metric_;
  raft::device_matrix<T, int64_t, row_major> dataset_;
  std::optional<raft::device_vector<T, int64_t>> norms_;
  std::optional<raft::device_vector_view<const T, int64_t>> norms_view_;
  raft::device_matrix_view<const T, int64_t, row_major> dataset_view_;
  T metric_arg_;
};

/** @} */

}  // namespace raft::neighbors::brute_force
