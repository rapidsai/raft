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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>

#include <raft/matrix/slice.cuh>
#include <raft/neighbors/brute_force.cuh>

#include <raft/core/logger.hpp>

namespace raft::neighbors::brute_force {
/**
 * @addtogroup brute_force
 * @{
 */

/**
 * @brief Brute force query over batches of k
 *
 * This class lets you query for batches of k. For example, you can get
 * the first 100 neighbors, then the next 100 neighbors etc.
 *
 * Example usage:
 * @code{.cpp}
 * // create a brute_force knn index
 * raft::device_resources res;
 * auto index = raft::neighbors::brute_force::build(res,
 *                                                  raft::make_const_mdspan(dataset.view()));
 *
 * // search the index in batches of 128 nearest neighbors
 * auto search = raft::make_const_mdspan(dataset.view());
 * for (auto & batch: batch_k_query(res, index, query, 128)) {
 *  // batch.indices() and batch.distances() contain the information on the current batch
 * }
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 */
template <typename T, typename IdxT = int64_t>
class batch_k_query {
 public:
  /**
   * @brief Construct a brute force batch k query object
   *
   * Constructs the batch_k_query - which lets you iterate over batches of
   * nearest neighbors.
   *
   * @param[in] res
   * @param[in] index The index to query
   * @param[in] query A device matrix view to query for [n_queries, index->dim()]
   * @param[in] batch_size The size of each batch
   */
  batch_k_query(const raft::resources& res,
                const raft::neighbors::brute_force::index<T>& index,
                raft::device_matrix_view<const T, int64_t, row_major> query,
                int64_t batch_size)
    : res(res), index(index), query(query), batch_size(batch_size)
  {
    auto metric = index.metric();

    // precompute query norms, and re-use across batches
    if (metric == raft::distance::DistanceType::L2Expanded ||
        metric == raft::distance::DistanceType::L2SqrtExpanded ||
        metric == raft::distance::DistanceType::CosineExpanded) {
      query_norms = make_device_vector<T, int64_t>(res, query.extent(0));

      if (metric == raft::distance::DistanceType::CosineExpanded) {
        raft::linalg::norm(res,
                           query,
                           query_norms->view(),
                           raft::linalg::NormType::L2Norm,
                           raft::linalg::Apply::ALONG_ROWS,
                           raft::sqrt_op{});
      } else {
        raft::linalg::norm(res,
                           query,
                           query_norms->view(),
                           raft::linalg::NormType::L2Norm,
                           raft::linalg::Apply::ALONG_ROWS);
      }
    }
  }

  /** a single batch of nearest neighbors in device memory */
  class batch {
   public:
    /** Create a new empty batch of data */
    batch(raft::resources const& res, int64_t rows, int64_t cols)
      : indices_(make_device_matrix<IdxT, int64_t>(res, rows, cols)),
        distances_(make_device_matrix<T, int64_t>(res, rows, cols)),
        start_k(0),
        end_k(cols)
    {
    }

    /** Returns the indices for the batch */
    device_matrix_view<const IdxT, int64_t> indices() const
    {
      return raft::make_const_mdspan(indices_.view());
    }

    /** Returns the distances for the batch */
    device_matrix_view<const T, int64_t> distances() const
    {
      return raft::make_const_mdspan(distances_.view());
    }

    friend class iterator;
    friend class batch_k_query<T, IdxT>;

   protected:
    raft::device_matrix<IdxT, int64_t> indices_;
    raft::device_matrix<T, int64_t> distances_;
    int64_t start_k;
    int64_t end_k;
  };

  class iterator {
   public:
    using value_type = batch;
    using reference  = const value_type&;
    using pointer    = const value_type*;

    iterator(const batch_k_query<T, IdxT>* query, int64_t offset = 0)
      : current(query->res, 0, 0), batches(query->res, 0, 0), query(query), offset(offset)
    {
      load_batches();
      slice_current_batch();
    }

    reference operator*() const { return current; }

    pointer operator->() const { return &current; }

    iterator& operator++()
    {
      offset = std::min(offset + query->batch_size, query->index.size());
      if (offset + query->batch_size > current_batch_size) { load_batches(); }
      slice_current_batch();
      return *this;
    }

    iterator operator++(int)
    {
      iterator previous(*this);
      operator++();
      return previous;
    }

    friend bool operator==(const iterator& lhs, const iterator& rhs)
    {
      return (lhs.query == rhs.query) && (lhs.offset == rhs.offset);
    };
    friend bool operator!=(const iterator& lhs, const iterator& rhs) { return !(lhs == rhs); };

   protected:
    void load_batches()
    {
      if (offset >= query->index.size()) { return; }

      // we're aiming to load multiple batches here - since we don't know the max iteration
      // grow the size we're loading exponentially
      int64_t batch_size =
        std::min(std::max(offset * 2, query->batch_size * 2), query->index.size());
      batches = batch(query->res, query->query.extent(0), batch_size);
      query->load_batch(batches);
      current_batch_size = batch_size;
    }

    void slice_current_batch()
    {
      auto num_queries = batches.indices_.extent(0);
      auto batch_size  = std::min(query->batch_size, query->index.size() - offset);
      current          = batch(query->res, num_queries, batch_size);

      if (!num_queries || !batch_size) { return; }

      matrix::slice_coordinates<int64_t> coords{0, offset, num_queries, offset + batch_size};
      matrix::slice(query->res, batches.indices(), current.indices_.view(), coords);
      matrix::slice(query->res, batches.distances(), current.distances_.view(), coords);
    }

    // the current batch of data
    batch current;

    // the currently loaded group of data (containing multiple batches of data that we can iterate
    // through)
    batch batches;

    const batch_k_query<T, IdxT>* query;
    int64_t offset, current_batch_size;
  };

 protected:
  void load_batch(batch& output) const
  {
    std::optional<raft::device_vector_view<const float, int64_t>> query_norms_view;
    if (query_norms) { query_norms_view = query_norms->view(); }

    brute_force::search<T, IdxT>(
      res, index, query, output.indices_.view(), output.distances_.view(), query_norms_view);
  }

  const raft::resources& res;
  const raft::neighbors::brute_force::index<T>& index;
  raft::device_matrix_view<const T, int64_t, row_major> query;
  int64_t batch_size;
  std::optional<device_vector<T, int64_t>> query_norms;
};

/** @} */
}  // namespace raft::neighbors::brute_force
