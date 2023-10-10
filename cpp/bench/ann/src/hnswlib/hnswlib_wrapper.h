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

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "../common/ann_types.hpp"
#include "../common/thread_pool.hpp"
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

template <typename T>
class HnswLib : public ANN<T> {
 public:
  // https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
  struct BuildParam {
    int M;
    int ef_construction;
    int num_threads = omp_get_num_procs();
  };

  using typename ANN<T>::AnnSearchParam;
  struct SearchParam : public AnnSearchParam {
    int ef;
    int num_threads = omp_get_num_procs();
  };

  HnswLib(Metric metric, int dim, const BuildParam& param);

  void build(const T* dataset, size_t nrow, cudaStream_t stream = 0) override;

  void set_search_param(const AnnSearchParam& param) override;
  void search(const T* query,
              int batch_size,
              int k,
              size_t* indices,
              float* distances,
              cudaStream_t stream = 0) const override;

  void save(const std::string& path_to_index) const override;
  void load(const std::string& path_to_index) override;

  AlgoProperty get_preference() const override
  {
    AlgoProperty property;
    property.dataset_memory_type = MemoryType::Host;
    property.query_memory_type   = MemoryType::Host;
    return property;
  }

 private:
  void get_search_knn_results_(const T* query, int k, size_t* indices, float* distances) const;

  std::unique_ptr<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>> appr_alg_;
  std::unique_ptr<hnswlib::SpaceInterface<typename hnsw_dist_t<T>::type>> space_;

  using ANN<T>::metric_;
  using ANN<T>::dim_;
  int ef_construction_;
  int m_;
  int num_threads_;
  std::unique_ptr<FixedThreadPool> thread_pool_;
};

template <typename T>
HnswLib<T>::HnswLib(Metric metric, int dim, const BuildParam& param) : ANN<T>(metric, dim)
{
  assert(dim_ > 0);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t>);
  if constexpr (std::is_same_v<T, uint8_t>) {
    if (metric_ != Metric::kEuclidean) {
      throw std::runtime_error("hnswlib<uint8_t> only supports Euclidean distance");
    }
  }

  ef_construction_ = param.ef_construction;
  m_               = param.M;
  num_threads_     = param.num_threads;
}

template <typename T>
void HnswLib<T>::build(const T* dataset, size_t nrow, cudaStream_t)
{
  if constexpr (std::is_same_v<T, float>) {
    if (metric_ == Metric::kInnerProduct) {
      space_ = std::make_unique<hnswlib::InnerProductSpace>(dim_);
    } else {
      space_ = std::make_unique<hnswlib::L2Space>(dim_);
    }
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    space_ = std::make_unique<hnswlib::L2SpaceI>(dim_);
  }

  appr_alg_ = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    space_.get(), nrow, m_, ef_construction_);

  thread_pool_                  = std::make_unique<FixedThreadPool>(num_threads_);
  const size_t items_per_thread = nrow / (num_threads_ + 1);

  thread_pool_->submit(
    [&](size_t i) {
      if (i < items_per_thread && i % 10000 == 0) {
        char buf[20];
        std::time_t now = std::time(nullptr);
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

        printf("%s building %zu / %zu\n", buf, i, items_per_thread);
        fflush(stdout);
      }

      appr_alg_->addPoint(dataset + i * dim_, i);
    },
    nrow);
}

template <typename T>
void HnswLib<T>::set_search_param(const AnnSearchParam& param_)
{
  auto param     = dynamic_cast<const SearchParam&>(param_);
  appr_alg_->ef_ = param.ef;

  if (!thread_pool_ || num_threads_ != param.num_threads) {
    num_threads_ = param.num_threads;
    thread_pool_ = std::make_unique<FixedThreadPool>(num_threads_);
  }
}

template <typename T>
void HnswLib<T>::search(
  const T* query, int batch_size, int k, size_t* indices, float* distances, cudaStream_t) const
{
  thread_pool_->submit(
    [&](int i) {
      // hnsw can only handle a single vector at a time.
      get_search_knn_results_(query + i * dim_, k, indices + i * k, distances + i * k);
    },
    batch_size);
}

template <typename T>
void HnswLib<T>::save(const std::string& path_to_index) const
{
  appr_alg_->saveIndex(std::string(path_to_index));
}

template <typename T>
void HnswLib<T>::load(const std::string& path_to_index)
{
  if constexpr (std::is_same_v<T, float>) {
    if (metric_ == Metric::kInnerProduct) {
      space_ = std::make_unique<hnswlib::InnerProductSpace>(dim_);
    } else {
      space_ = std::make_unique<hnswlib::L2Space>(dim_);
    }
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    space_ = std::make_unique<hnswlib::L2SpaceI>(dim_);
  }

  appr_alg_ = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    space_.get(), path_to_index);
}

template <typename T>
void HnswLib<T>::get_search_knn_results_(const T* query,
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

};  // namespace raft::bench::ann
