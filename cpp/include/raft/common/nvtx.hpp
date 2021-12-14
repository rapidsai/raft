/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <optional>
#include "detail/nvtx.hpp"

namespace raft::common {

/**
 * @brief Push a named nvtx range
 * @param format range name format (accepts printf-style arguments)
 * @param args the arguments for the printf-style formatting
 */
template <typename... Args>
inline void push_nvtx_range(const char* format, Args... args)
{
  detail::push_range(format, args...);
}

/**
 * @brief Synchronize CUDA stream and push a named nvtx range
 * @param format range name format (accepts printf-style arguments)
 * @param args the arguments for the printf-style formatting
 * @param stream stream to synchronize
 */
template <typename... Args>
inline void push_nvtx_range(rmm::cuda_stream_view stream, const char* format, Args... args)
{
  detail::push_range(stream, format, args...);
}

/** Pop the latest range */
inline void pop_nvtx_range() { detail::pop_range(); }

/**
 * @brief Synchronize CUDA stream and pop the latest nvtx range
 * @param stream stream to synchronize
 */
inline void pop_nvtx_range(rmm::cuda_stream_view stream) { detail::pop_range(stream); }

/** Push a named nvtx range that would be popped at the end of the object lifetime. */
class nvtx_range {
 private:
  std::optional<rmm::cuda_stream_view> stream_maybe_;

 public:
  /**
   * Synchronize CUDA stream and push a named nvtx range
   * At the end of the object lifetime, synchronize again and pop the range.
   *
   * @param stream stream to synchronize
   * @param format range name format (accepts printf-style arguments)
   * @param args the arguments for the printf-style formatting
   */
  template <typename... Args>
  explicit nvtx_range(rmm::cuda_stream_view stream, const char* format, Args... args)
    : stream_maybe_(std::make_optional(stream))
  {
    push_nvtx_range(stream, format, args...);
  }

  /**
   * Push a named nvtx range.
   * At the end of the object lifetime, pop the range back.
   *
   * @param format range name format (accepts printf-style arguments)
   * @param args the arguments for the printf-style formatting
   */
  template <typename... Args>
  explicit nvtx_range(const char* format, Args... args) : stream_maybe_(std::nullopt)
  {
    push_nvtx_range(format, args...);
  }

  ~nvtx_range()
  {
    if (stream_maybe_.has_value()) {
      pop_nvtx_range(*stream_maybe_);
    } else {
      pop_nvtx_range();
    }
  }

  /* This object is not meant to be touched. */
  nvtx_range(const nvtx_range&) = delete;
  nvtx_range(nvtx_range&&)      = delete;
  auto operator=(const nvtx_range&) -> nvtx_range& = delete;
  auto operator=(nvtx_range&&) -> nvtx_range&      = delete;
  static auto operator new(std::size_t) -> void*   = delete;
  static auto operator new[](std::size_t) -> void* = delete;
};

}  // namespace raft::common
