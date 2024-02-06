/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <memory>

#ifndef RAFT_EXPLICIT_INSTANTIATE_ONLY
#include "brute_force-inl.cuh"
#endif

#ifdef RAFT_COMPILED
#include "brute_force-ext.cuh"
#endif

#include <raft/neighbors/detail/knn_brute_force_batch_k_query.cuh>

namespace raft::neighbors::brute_force {
/**
 * @brief Make a brute force query over batches of k
 *
 * This lets you query for batches of k. For example, you can get
 * the first 100 neighbors, then the next 100 neighbors etc.
 *
 * Example usage:
 * @code{.cpp}
 * #include <raft/neighbors/brute_force.cuh>
 * #include <raft/core/device_mdarray.hpp>
 * #include <raft/random/make_blobs.cuh>

 * // create a random dataset
 * int n_rows = 10000;
 * int n_cols = 10000;

 * raft::device_resources res;
 * auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, n_cols);
 * auto labels = raft::make_device_vector<int64_t, int64_t>(res, n_rows);

 * raft::random::make_blobs(res, dataset.view(), labels.view());
 *
 * // create a brute_force knn index from the dataset
 * auto index = raft::neighbors::brute_force::build(res,
 *                                                  raft::make_const_mdspan(dataset.view()));
 *
 * // search the index in batches of 128 nearest neighbors
 * auto search = raft::make_const_mdspan(dataset.view());
 * auto query = make_batch_k_query<float, int>(res, index, search, 128);
 * for (auto & batch: *query) {
 *  // batch.indices() and batch.distances() contain the information on the current batch
 * }
 *
 * // we can also support variable sized batches - loaded up a different number
 * // of neighbors at each iteration through the ::advance method
 * int64_t batch_size = 128;
 * query = make_batch_k_query<float, int>(res, index, search, batch_size);
 * for (auto it = query->begin(); it != query->end(); it.advance(batch_size)) {
 *  // batch.indices() and batch.distances() contain the information on the current batch
 *
 *  batch_size += 16; // load up an extra 16 items in the next batch
 * }
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 * @param[in] res
 * @param[in] index The index to query
 * @param[in] query A device matrix view to query for [n_queries, index->dim()]
 * @param[in] batch_size The size of each batch
 */

template <typename T, typename IdxT>
std::shared_ptr<batch_k_query<T, IdxT>> make_batch_k_query(
  const raft::resources& res,
  const raft::neighbors::brute_force::index<T>& index,
  raft::device_matrix_view<const T, int64_t, row_major> query,
  int64_t batch_size)
{
  return std::shared_ptr<batch_k_query<T, IdxT>>(
    new detail::gpu_batch_k_query<T, IdxT>(res, index, query, batch_size));
}
}  // namespace raft::neighbors::brute_force
