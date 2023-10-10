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

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <omp.h>
#include <stdexcept>
#include <thread>
#include <utility>

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
