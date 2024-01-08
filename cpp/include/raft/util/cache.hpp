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

#include <raft/core/error.hpp>

#include <cstddef>
#include <list>
#include <mutex>
#include <tuple>
#include <unordered_map>
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

  explicit lru(size_t size = kDefaultSize) : size_(size)
  {
    RAFT_EXPECTS(size >= 1, "The cache must fit at least one record.");
  }

  void set(const K& key, const Values&... values)
  {
    std::lock_guard<std::mutex> guard(lock_);
    auto pos = map_.find(key);
    if (pos == map_.end()) {
      if (map_.size() >= size_) {
        map_.erase(queue_.back());
        queue_.pop_back();
      }
    } else {
      queue_.erase(std::get<0>(pos->second));
    }
    queue_.push_front(key);
    map_[key] = std::make_tuple(queue_.begin(), values...);
  }

  auto get(const K& key, Values*... values) -> bool
  {
    std::lock_guard<std::mutex> guard(lock_);
    auto pos = map_.find(key);
    if (pos == map_.end()) { return false; }
    auto& map_val = pos->second;
    queue_.erase(std::get<0>(map_val));
    queue_.push_front(key);
    std::get<0>(map_val) = queue_.begin();
    set_values(map_val, values..., std::index_sequence_for<Values...>());
    return true;
  }

 private:
  using queue_iterator = typename std::list<K>::iterator;
  std::list<K> queue_{};
  std::unordered_map<K, std::tuple<queue_iterator, Values...>, HashK, EqK> map_{};
  std::mutex lock_{};
  size_t size_;

  template <size_t... Is>
  static void set_values(const std::tuple<queue_iterator, Values...>& tup,
                         Values*... vals,
                         std::index_sequence<Is...>)
  {
    ((*vals = std::get<Is + 1>(tup)), ...);
  }
};

};  // namespace raft::cache
