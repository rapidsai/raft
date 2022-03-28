/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <stdarg.h>

#define SPDLOG_HEADER_ONLY
#include <spdlog/sinks/stdout_color_sinks.h>  // NOLINT
#include <spdlog/spdlog.h>                    // NOLINT

#include <algorithm>

#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include <raft/common/detail/callback_sink.hpp>

/**
 * @defgroup logging levels used in raft
 *
 * @note exactly match the corresponding ones (but reverse in terms of value)
 *       in spdlog for wrapping purposes
 *
 * @{
 */
#define RAFT_LEVEL_TRACE    6
#define RAFT_LEVEL_DEBUG    5
#define RAFT_LEVEL_INFO     4
#define RAFT_LEVEL_WARN     3
#define RAFT_LEVEL_ERROR    2
#define RAFT_LEVEL_CRITICAL 1
#define RAFT_LEVEL_OFF      0
/** @} */

#if !defined(RAFT_ACTIVE_LEVEL)
#define RAFT_ACTIVE_LEVEL RAFT_LEVEL_DEBUG
#endif

namespace spdlog {
class logger;
namespace sinks {
template <class Mutex>
class CallbackSink;
using callback_sink_mt = CallbackSink<std::mutex>;
};  // namespace sinks
};  // namespace spdlog

namespace raft::detail {

/**
 * @defgroup CStringFormat Expand a C-style format string
 *
 * @brief Expands C-style formatted string into std::string
 *
 * @param[in] fmt format string
 * @param[in] vl  respective values for each of format modifiers in the string
 *
 * @return the expanded `std::string`
 *
 * @{
 */
std::string format(const char* fmt, va_list& vl)
{
  char buf[4096];
  vsnprintf(buf, sizeof(buf), fmt, vl);
  return std::string(buf);
}

std::string format(const char* fmt, ...)
{
  va_list vl;
  va_start(vl, fmt);
  std::string str = format(fmt, vl);
  va_end(vl);
  return str;
}
/** @} */

int convert_level_to_spdlog(int level)
{
  level = std::max(RAFT_LEVEL_OFF, std::min(RAFT_LEVEL_TRACE, level));
  return RAFT_LEVEL_TRACE - level;
}

};  // namespace raft::detail

/**
 * @defgroup loggerMacros Helper macros for dealing with logging
 * @{
 */
#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_TRACE)
#define RAFT_LOG_TRACE(fmt, ...)                                 \
  do {                                                           \
    std::stringstream ss;                                        \
    ss << raft::detail::format("%s:%d ", __FILE__, __LINE__);    \
    ss << raft::detail::format(fmt, ##__VA_ARGS__);              \
    raft::logger::get().log(RAFT_LEVEL_TRACE, ss.str().c_str()); \
  } while (0)
#else
#define RAFT_LOG_TRACE(fmt, ...) void(0)
#endif

#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
#define RAFT_LOG_DEBUG(fmt, ...)                                 \
  do {                                                           \
    std::stringstream ss;                                        \
    ss << raft::detail::format("%s:%d ", __FILE__, __LINE__);    \
    ss << raft::detail::format(fmt, ##__VA_ARGS__);              \
    raft::logger::get().log(RAFT_LEVEL_DEBUG, ss.str().c_str()); \
  } while (0)
#else
#define RAFT_LOG_DEBUG(fmt, ...) void(0)
#endif

#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_INFO)
#define RAFT_LOG_INFO(fmt, ...) raft::logger::get().log(RAFT_LEVEL_INFO, fmt, ##__VA_ARGS__)
#else
#define RAFT_LOG_INFO(fmt, ...) void(0)
#endif

#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_WARN)
#define RAFT_LOG_WARN(fmt, ...) raft::logger::get().log(RAFT_LEVEL_WARN, fmt, ##__VA_ARGS__)
#else
#define RAFT_LOG_WARN(fmt, ...) void(0)
#endif

#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_ERROR)
#define RAFT_LOG_ERROR(fmt, ...) raft::logger::get().log(RAFT_LEVEL_ERROR, fmt, ##__VA_ARGS__)
#else
#define RAFT_LOG_ERROR(fmt, ...) void(0)
#endif

#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_CRITICAL)
#define RAFT_LOG_CRITICAL(fmt, ...) raft::logger::get().log(RAFT_LEVEL_CRITICAL, fmt, ##__VA_ARGS__)
#else
#define RAFT_LOG_CRITICAL(fmt, ...) void(0)
#endif
/** @} */