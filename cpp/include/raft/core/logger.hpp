/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <raft/core/logger_macros.hpp>

#include <rapids_logger/logger.hpp>

#include <sstream>

namespace raft {

/**
 * @brief Returns the default sink for the global logger.
 *
 * If the environment variable `RAFT_DEBUG_LOG_FILE` is defined, the default sink is a sink to that
 * file. Otherwise, the default is to dump to stderr.
 *
 * @return sink_ptr The sink to use
 */
inline rapids_logger::sink_ptr default_sink()
{
  auto* filename = std::getenv("RAFT_DEBUG_LOG_FILE");
  if (filename != nullptr) {
    return std::make_shared<rapids_logger::basic_file_sink_mt>(filename, true);
  }
  return std::make_shared<rapids_logger::stderr_sink_mt>();
}

/**
 * @brief Returns the default log pattern for the global logger.
 *
 * @return std::string The default log pattern.
 */
inline std::string default_pattern() { return "[%6t][%H:%M:%S:%f][%-6l] %v"; }

/**
 * @brief Get the default logger.
 *
 * @return logger& The default logger
 */
inline rapids_logger::logger& default_logger()
{
  static rapids_logger::logger logger_ = [] {
    rapids_logger::logger logger_{"RAFT", {default_sink()}};
    logger_.set_pattern(default_pattern());
    return logger_;
  }();
  return logger_;
}

}  // namespace raft

#if (RAFT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_TRACE)
#define RAFT_LOG_TRACE_VEC(ptr, len)                                             \
  do {                                                                           \
    std::stringstream ss;                                                        \
    ss << raft::detail::format("%s:%d ", __FILE__, __LINE__);                    \
    print_vector(#ptr, ptr, len, ss);                                            \
    raft::default_logger().log(RAPIDS_LOGGER_LOG_LEVEL_TRACE, ss.str().c_str()); \
  } while (0)
#else
#define RAFT_LOG_TRACE_VEC(ptr, len) void(0)
#endif
