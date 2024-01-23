/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/error.hpp>

#include <cstddef>
#include <list>
#include <mutex>
#include <optional>
#include <tuple>
#include <utility>

namespace raft::cache {

/** Associative cache with least recently used replacement policy. */
template <typename K,
          typename HashK = std::hash<K>,
          typename EqK   = std::equal_to<K>,
          typename... Values>
class lru {
 public:
  /** Default cache size. */
  static constexpr size_t kDefaultSize = 100;

  explicit lru(size_t size = kDefaultSize) : size_(size), data_(size), order_(size)
  {
    for (size_t i = 0; i < size_; i++) {
      order_[i] = i + 1;
      data_[i]  = std::nullopt;
    }
    RAFT_EXPECTS(size >= 1, "The cache must fit at least one record.");
  }

  void set(const K& key, const Values&... values)
  {
    std::lock_guard<std::mutex> guard(lock_);
    auto pos  = root_;
    auto prev = root_;
    while (true) {
      auto next = order_[pos];
      if (next >= size_ || !data_[pos].has_value()) { break; }
      prev = pos;
      pos  = next;
    }
    update_lru(prev, pos);
    data_[pos].emplace(key, values...);
  }

  auto get(const K& key, Values*... values) -> bool
  {
    std::lock_guard<std::mutex> guard(lock_);
    auto pos  = root_;
    auto prev = root_;
    while (pos < size_ && data_[pos].has_value()) {
      auto& val = data_[pos].value();
      if (std::get<0>(val) == key) {
        update_lru(prev, pos);
        set_values(val, values..., std::index_sequence_for<Values...>());
        return true;
      }
      prev = pos;
      pos  = order_[pos];
    }
    return false;
  }

 private:
  std::size_t size_;
  std::vector<std::optional<std::tuple<K, Values...>>> data_;
  std::vector<std::size_t> order_;
  std::mutex lock_{};
  std::size_t root_{0};

  // Place `pos` at the root of the queue.
  inline void update_lru(std::size_t prev, std::size_t pos)
  {
    if (pos != root_) {
      order_[prev] = order_[pos];
      order_[pos]  = root_;
      root_        = pos;
    }
  }

  template <size_t... Is>
  static void set_values(const std::tuple<K, Values...>& tup,
                         Values*... vals,
                         std::index_sequence<Is...>)
  {
    ((*vals = std::get<Is + 1>(tup)), ...);
  }
};

};  // namespace raft::cache
