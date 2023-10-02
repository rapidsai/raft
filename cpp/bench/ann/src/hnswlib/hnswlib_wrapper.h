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

#include <omp.h>

#include "../common/ann_types.hpp"
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

class FixedThreadPool {
 public:
  FixedThreadPool(int num_threads)
  {
    if (num_threads < 1) {
      throw std::runtime_error("num_threads must >= 1");
    } else if (num_threads == 1) {
      return;
    }

    tasks_ = new Task_[num_threads];

    threads_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      threads_.emplace_back([&, i] {
        auto& task = tasks_[i];
        while (true) {
          std::unique_lock<std::mutex> lock(task.mtx);
          task.cv.wait(lock,
                       [&] { return task.has_task || finished_.load(std::memory_order_relaxed); });
          if (finished_.load(std::memory_order_relaxed)) { break; }

          task.task();
          task.has_task = false;
        }
      });
    }
  }

  ~FixedThreadPool()
  {
    if (threads_.empty()) { return; }

    finished_.store(true, std::memory_order_relaxed);
    for (unsigned i = 0; i < threads_.size(); ++i) {
      auto& task = tasks_[i];
      std::lock_guard<std::mutex>(task.mtx);

      task.cv.notify_one();
      threads_[i].join();
    }

    delete[] tasks_;
  }

  template <typename Func, typename IdxT>
  void submit(Func f, IdxT len)
  {
    if (threads_.empty()) {
      for (IdxT i = 0; i < len; ++i) {
        f(i);
      }
      return;
    }

    const int num_threads = threads_.size();
    // one extra part for competition among threads
    const IdxT items_per_thread = len / (num_threads + 1);
    std::atomic<IdxT> cnt(items_per_thread * num_threads);

    auto wrapped_f = [&](IdxT start, IdxT end) {
      for (IdxT i = start; i < end; ++i) {
        f(i);
      }

      while (true) {
        IdxT i = cnt.fetch_add(1, std::memory_order_relaxed);
        if (i >= len) { break; }
        f(i);
      }
    };

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      IdxT start = i * items_per_thread;
      auto& task = tasks_[i];
      {
        std::lock_guard lock(task.mtx);
        (void)lock;  // stop nvcc warning
        task.task = std::packaged_task<void()>([=] { wrapped_f(start, start + items_per_thread); });
        futures.push_back(task.task.get_future());
        task.has_task = true;
      }
      task.cv.notify_one();
    }

    for (auto& fut : futures) {
      fut.wait();
    }
    return;
  }

 private:
  struct alignas(64) Task_ {
    std::mutex mtx;
    std::condition_variable cv;
    bool has_task = false;
    std::packaged_task<void()> task;
  };

  Task_* tasks_;
  std::vector<std::thread> threads_;
  std::atomic<bool> finished_{false};
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
