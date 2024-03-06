/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/neighbors_types.hpp>

namespace raft::neighbors::brute_force {
/**
 * @addtogroup brute_force_knn
 * @{
 */

using ann::index_params;
using ann::search_params;

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
  [[nodiscard]] constexpr inline auto size() const noexcept { return dataset_view_.extent(0); }

  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept { return dataset_view_.extent(1); }

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

  template <typename data_accessor>
  index(raft::resources const& res,
        index_params const& params,
        mdspan<const T, matrix_extent<int64_t>, row_major, data_accessor> dataset,
        std::optional<raft::device_vector<T, int64_t>>&& norms = std::nullopt)
    : ann::index(),
      metric_(params.metric),
      dataset_(make_device_matrix<T, int64_t>(res, 0, 0)),
      norms_(std::move(norms)),
      metric_arg_(params.metric_arg)
  {
    if (norms_) { norms_view_ = make_const_mdspan(norms_.value().view()); }
    update_dataset(res, dataset);
    resource::sync_stream(res);
  }

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
    dataset_ = make_device_matrix<T, int64_t>(res, dataset.extent(0), dataset.extent(1));
    raft::copy(res, dataset_.view(), dataset);
    dataset_view_ = make_const_mdspan(dataset_.view());
  }

 private:
  raft::distance::DistanceType metric_;
  raft::device_matrix<T, int64_t, row_major> dataset_;
  std::optional<raft::device_vector<T, int64_t>> norms_;
  std::optional<raft::device_vector_view<const T, int64_t>> norms_view_;
  raft::device_matrix_view<const T, int64_t, row_major> dataset_view_;
  T metric_arg_;
};

/**
 * @brief Interface for performing queries over values of k
 *
 * This interface lets you iterate over batches of k from a brute_force::index.
 * This lets you do things like retrieve the first 100 neighbors for a query,
 * apply post processing to remove any unwanted items and then if needed get the
 * next 100 closest neighbors for the query.
 *
 * This query interface exposes C++ iterators through the ::begin and ::end, and
 * is compatible with range based for loops.
 *
 * Note that this class is an abstract class without any cuda dependencies, meaning
 * that it doesn't require a cuda compiler to use - but also means it can't be directly
 * instantiated.  See the raft::neighbors::brute_force::make_batch_k_query
 * function for usage examples.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 */
template <typename T, typename IdxT = int64_t>
class batch_k_query {
 public:
  batch_k_query(const raft::resources& res,
                int64_t index_size,
                int64_t query_size,
                int64_t batch_size)
    : res(res), index_size(index_size), query_size(query_size), batch_size(batch_size)
  {
  }
  virtual ~batch_k_query() {}

  using value_type = raft::neighbors::batch<T, IdxT>;

  class iterator {
   public:
    using value_type = raft::neighbors::batch<T, IdxT>;
    using reference  = const value_type&;
    using pointer    = const value_type*;

    iterator(const batch_k_query<T, IdxT>* query, int64_t offset = 0)
      : current(query->res, 0, 0), batches(query->res, 0, 0), query(query), offset(offset)
    {
      query->load_batch(offset, query->batch_size, &batches);
      query->slice_batch(batches, offset, query->batch_size, &current);
    }

    reference operator*() const { return current; }

    pointer operator->() const { return &current; }

    iterator& operator++()
    {
      advance(query->batch_size);
      return *this;
    }

    iterator operator++(int)
    {
      iterator previous(*this);
      operator++();
      return previous;
    }

    /**
     * @brief Advance the iterator, using a custom size for the next batch
     *
     * Using operator++ means that we will load up the same batch_size for each
     * batch. This method allows us to get around this restriction, and load up
     * arbitrary batch sizes on each iteration.
     * See raft::neighbors::brute_force::make_batch_k_query for a usage example.
     *
     * @param[in] next_batch_size: size of the next batch to load up
     */
    void advance(int64_t next_batch_size)
    {
      offset = std::min(offset + current.batch_size(), query->index_size);
      if (offset + next_batch_size > batches.batch_size()) {
        query->load_batch(offset, next_batch_size, &batches);
      }
      query->slice_batch(batches, offset, next_batch_size, &current);
    }

    friend bool operator==(const iterator& lhs, const iterator& rhs)
    {
      return (lhs.query == rhs.query) && (lhs.offset == rhs.offset);
    };
    friend bool operator!=(const iterator& lhs, const iterator& rhs) { return !(lhs == rhs); };

   protected:
    // the current batch of data
    value_type current;

    // the currently loaded group of data (containing multiple batches of data that we can iterate
    // through)
    value_type batches;

    const batch_k_query<T, IdxT>* query;
    int64_t offset, current_batch_size;
  };

  iterator begin() const { return iterator(this); }
  iterator end() const { return iterator(this, index_size); }

 protected:
  // these two methods need cuda code, and are implemented in the subclass
  virtual void load_batch(int64_t offset,
                          int64_t next_batch_size,
                          batch<T, IdxT>* output) const = 0;
  virtual void slice_batch(const value_type& input,
                           int64_t offset,
                           int64_t batch_size,
                           value_type* output) const    = 0;

  const raft::resources& res;
  int64_t index_size, query_size, batch_size;
};
/** @} */

}  // namespace raft::neighbors::brute_force
