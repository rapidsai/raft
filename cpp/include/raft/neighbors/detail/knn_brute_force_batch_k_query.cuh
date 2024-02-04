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

#include <raft/linalg/norm.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/neighbors/brute_force_types.hpp>
#include <raft/neighbors/detail/knn_brute_force.cuh>

namespace raft::neighbors::brute_force::detail {
template <typename T, typename IdxT = int64_t>
class gpu_batch_k_query : public batch_k_query<T, IdxT> {
 public:
  gpu_batch_k_query(const raft::resources& res,
                    const raft::neighbors::brute_force::index<T>& index,
                    raft::device_matrix_view<const T, int64_t, row_major> query,
                    int64_t batch_size)
    : batch_k_query<T, IdxT>(res, index.size(), query.extent(0), batch_size),
      index(index),
      query(query)
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

 protected:
  void load_batch(int64_t offset, int64_t next_batch_size, batch<T, IdxT>* output) const override
  {
    if (offset >= index.size()) { return; }

    // we're aiming to load multiple batches here - since we don't know the max iteration
    // grow the size we're loading exponentially
    int64_t batch_size = std::min(std::max(offset * 2, next_batch_size * 2), this->index_size);
    output->resize(this->res, this->query_size, batch_size);

    std::optional<raft::device_vector_view<const float, int64_t>> query_norms_view;
    if (query_norms) { query_norms_view = query_norms->view(); }

    raft::neighbors::detail::brute_force_search<T, IdxT>(
      this->res, index, query, output->indices(), output->distances(), query_norms_view);
  };

  void slice_batch(const batch<T, IdxT>& input,
                   int64_t offset,
                   int64_t batch_size,
                   batch<T, IdxT>* output) const override
  {
    auto num_queries = input.indices().extent(0);
    batch_size       = std::min(batch_size, index.size() - offset);

    output->resize(this->res, num_queries, batch_size);

    if (!num_queries || !batch_size) { return; }

    matrix::slice_coordinates<int64_t> coords{0, offset, num_queries, offset + batch_size};
    matrix::slice(this->res, input.indices(), output->indices(), coords);
    matrix::slice(this->res, input.distances(), output->distances(), coords);
  }

  const raft::neighbors::brute_force::index<T>& index;
  raft::device_matrix_view<const T, int64_t, row_major> query;
  std::optional<device_vector<T, int64_t>> query_norms;
};
}  // namespace raft::neighbors::brute_force::detail
