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

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/cagra_serialize.cuh>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../common/ann_types.hpp"
#include "../common/thread_pool.hpp"
#include "raft_ann_bench_utils.h"
#include <raft/util/cudart_utils.hpp>

#include <hnswlib.h>

namespace raft::bench::ann {

template <typename T>
struct hnsw_dist_t {
  using type = void;
};

template <>
struct hnsw_dist_t<float> {
  using type = float;
};

template <>
struct hnsw_dist_t<uint8_t> {
  using type = int;
};

template <>
struct hnsw_dist_t<int8_t> {
  using type = int;
};

template <typename T, typename IdxT>
class RaftCagraHnswlib : public ANN<T> {
 public:
  using typename ANN<T>::AnnSearchParam;

  struct SearchParam : public AnnSearchParam {
    int ef;
    int num_threads = 1;
  };

  using BuildParam = raft::neighbors::cagra::index_params;

  RaftCagraHnswlib(Metric metric, int dim, const BuildParam& param, int concurrent_searches = 1)
    : ANN<T>(metric, dim), index_params_(param), dimension_(dim), handle_(cudaStreamPerThread)
  {
    index_params_.metric = parse_metric_type(metric);
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
  }

  ~RaftCagraHnswlib() noexcept {}

  void build(const T* dataset, size_t nrow, cudaStream_t stream) final;

  void set_search_param(const AnnSearchParam& param) override;

  // TODO: if the number of results is less than k, the remaining elements of 'neighbors'
  // will be filled with (size_t)-1
  void search(const T* queries,
              int batch_size,
              int k,
              size_t* neighbors,
              float* distances,
              cudaStream_t stream = 0) const override;

  // to enable dataset access from GPU memory
  AlgoProperty get_preference() const override
  {
    AlgoProperty property;
    property.dataset_memory_type = MemoryType::HostMmap;
    property.query_memory_type   = MemoryType::Host;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;

 private:
  void get_search_knn_results_(const T* query, int k, size_t* indices, float* distances) const;

  raft::device_resources handle_;
  BuildParam index_params_;
  std::optional<raft::neighbors::cagra::index<T, IdxT>> index_;
  int device_;
  int dimension_;

  std::unique_ptr<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>> appr_alg_;
  std::unique_ptr<hnswlib::SpaceInterface<typename hnsw_dist_t<T>::type>> space_;
  int num_threads_;
  std::unique_ptr<FixedThreadPool> thread_pool_;

  Objective metric_objective_;
};

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::build(const T* dataset, size_t nrow, cudaStream_t)
{
  if (raft::get_device_for_address(dataset) == -1) {
    auto dataset_view =
      raft::make_host_matrix_view<const T, int64_t>(dataset, IdxT(nrow), dimension_);
    index_.emplace(raft::neighbors::cagra::build(handle_, index_params_, dataset_view));
    return;
  } else {
    auto dataset_view =
      raft::make_device_matrix_view<const T, int64_t>(dataset, IdxT(nrow), dimension_);
    index_.emplace(raft::neighbors::cagra::build(handle_, index_params_, dataset_view));
    return;
  }
}

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::set_search_param(const AnnSearchParam& param_)
{
  auto param        = dynamic_cast<const SearchParam&>(param_);
  appr_alg_->ef_    = param.ef;
  metric_objective_ = param.metric_objective;

  bool use_pool = (metric_objective_ == Objective::LATENCY && param.num_threads > 1) &&
                  (!thread_pool_ || num_threads_ != param.num_threads);
  if (use_pool) {
    num_threads_ = param.num_threads;
    thread_pool_ = std::make_unique<FixedThreadPool>(num_threads_);
  }
}

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::save(const std::string& file) const
{
  raft::neighbors::cagra::serialize_to_hnswlib<T, IdxT>(handle_, file, *index_);
}

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::load(const std::string& file)
{
  if constexpr (std::is_same_v<T, float>) {
    if (static_cast<Metric>(index_params_.metric) == Metric::kInnerProduct) {
      space_ = std::make_unique<hnswlib::InnerProductSpace>(dimension_);
    } else {
      space_ = std::make_unique<hnswlib::L2Space>(dimension_);
    }
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    space_ = std::make_unique<hnswlib::L2SpaceI>(dimension_);
  }

  std::cout << "about to create index" << std::endl;
  appr_alg_ = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    space_.get(), file);
  std::cout << "about to failed" << std::endl;
  appr_alg_->base_layer_only = true;
}

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::search(
  const T* queries, int batch_size, int k, size_t* neighbors, float* distances, cudaStream_t) const
{
  auto f = [&](int i) {
    // hnsw can only handle a single vector at a time.
    get_search_knn_results_(queries + i * dimension_, k, neighbors + i * k, distances + i * k);
  };
  if (metric_objective_ == Objective::LATENCY) {
    thread_pool_->submit(f, batch_size);
  } else {
    for (int i = 0; i < batch_size; i++) {
      f(i);
    }
  }
}

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::get_search_knn_results_(const T* query,
                                         int k,
                                         size_t* indices,
                                         float* distances) const
{
  auto result = appr_alg_->searchKnn(query, k);
  assert(result.size() >= static_cast<size_t>(k));

  for (int i = k - 1; i >= 0; --i) {
    indices[i]   = result.top().second;
    distances[i] = result.top().first;
    result.pop();
  }
}

}  // namespace raft::bench::ann
