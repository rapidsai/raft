/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include "detail/nvtx.hpp"

namespace raft {
namespace common {

/**
 * @brief Push a named nvtx range
 * @param format range name format (accepts printf-style arguments)
 * @param args the arguments for the printf-style formatting
 */
template <typename... Args>
inline void PUSH_NVTX_RANGE(const char* format, Args... args)
{
  detail::pushRange(format, args...);
}

/**
 * @brief Synchronize CUDA stream and push a named nvtx range
 * @param format range name format (accepts printf-style arguments)
 * @param args the arguments for the printf-style formatting
 * @param stream stream to synchronize
 */
template <typename... Args>
inline void PUSH_NVTX_RANGE(rmm::cuda_stream_view stream, const char* format, Args... args)
{
  detail::pushRange(stream, format, args...);
}

/** Pop the latest range */
inline void POP_NVTX_RANGE() { detail::popRange(); }

/**
 * @brief Synchronize CUDA stream and pop the latest nvtx range
 * @param stream stream to synchronize
 */
inline void POP_NVTX_RANGE(rmm::cuda_stream_view stream) { detail::popRange(stream); }

/** Push a named nvtx range that would be popped at the end of the object lifetime. */
class NvtxRange {
 private:
  std::optional<rmm::cuda_stream_view> streamMaybe;

  /* This object is not meant to be touched. */
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange(NvtxRange&&)      = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
  NvtxRange& operator=(NvtxRange&&)        = delete;
  static void* operator new(std::size_t)   = delete;
  static void* operator new[](std::size_t) = delete;

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
  NvtxRange(rmm::cuda_stream_view stream, const char* format, Args... args)
    : streamMaybe(std::make_optional(stream))
  {
    PUSH_NVTX_RANGE(stream, format, args...);
  }

  /**
   * Push a named nvtx range.
   * At the end of the object lifetime, pop the range back.
   *
   * @param format range name format (accepts printf-style arguments)
   * @param args the arguments for the printf-style formatting
   */
  template <typename... Args>
  NvtxRange(const char* format, Args... args) : streamMaybe(std::nullopt)
  {
    PUSH_NVTX_RANGE(format, args...);
  }

  ~NvtxRange()
  {
    if (streamMaybe.has_value()) {
      POP_NVTX_RANGE(*streamMaybe);
    } else {
      POP_NVTX_RANGE();
    }
  }
};

/*!
  \def RAFT_USING_NVTX_RANGE(...)
  When NVTX is enabled, push a named nvtx range and pop it at the end of the enclosing code block.

  This macro initializes a dummy NvtxRange variable on the stack,
  which pushes the range in its constructor and pops it in the destructor.
*/
#ifdef NVTX_ENABLED
#define RAFT_USING_NVTX_RANGE(...) raft::common::NvtxRange _AUTO_RANGE_##__LINE__(__VA_ARGS__)
#else
#define RAFT_USING_NVTX_RANGE(...) (void)0
#endif

}  // namespace common
}  // namespace raft
