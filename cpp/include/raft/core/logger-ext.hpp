/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/detail/macros.hpp>  // RAFT_INLINE_CONDITIONAL

#include <memory>         // std::unique_ptr
#include <string>         // std::string
#include <unordered_map>  // std::unordered_map

namespace raft {

static const std::string RAFT_NAME = "raft";
static const std::string default_log_pattern("[%L] [%H:%M:%S.%f] %v");

namespace detail {
RAFT_INLINE_CONDITIONAL std::string format(const char* fmt, ...);
}
/**
 * @brief The main Logging class for raft library.
 *
 * This class acts as a thin wrapper over the underlying `spdlog` interface. The
 * design is done in this way in order to avoid us having to also ship `spdlog`
 * header files in our installation.
 *
 * @todo This currently only supports logging to stdout. Need to add support in
 *       future to add custom loggers as well [Issue #2046]
 */
class logger {
 public:
  // @todo setting the logger once per process with
  logger(std::string const& name_ = "");
  /**
   * @brief Singleton method to get the underlying logger object
   *
   * @return the singleton logger object
   */
  static logger& get(std::string const& name = "");

  /**
   * @brief Set the logging level.
   *
   * Only messages with level equal or above this will be printed
   *
   * @param[in] level logging level
   *
   * @note The log level will actually be set only if the input is within the
   *       range [RAFT_LEVEL_TRACE, RAFT_LEVEL_OFF]. If it is not, then it'll
   *       be ignored. See documentation of decisiontree for how this gets used
   */
  void set_level(int level);

  /**
   * @brief Set the logging pattern
   *
   * @param[in] pattern the pattern to be set. Refer this link
   *                    https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
   *                    to know the right syntax of this pattern
   */
  void set_pattern(const std::string& pattern);

  /**
   * @brief Register a callback function to be run in place of usual log call
   *
   * @param[in] callback the function to be run on all logged messages
   */
  void set_callback(void (*callback)(int lvl, const char* msg));

  /**
   * @brief Register a flush function compatible with the registered callback
   *
   * @param[in] flush the function to use when flushing logs
   */
  void set_flush(void (*flush)());

  /**
   * @brief Tells whether messages will be logged for the given log level
   *
   * @param[in] level log level to be checked for
   * @return true if messages will be logged for this level, else false
   */
  bool should_log_for(int level) const;
  /**
   * @brief Query for the current log level
   *
   * @return the current log level
   */
  int get_level() const;

  /**
   * @brief Get the current logging pattern
   * @return the pattern
   */
  std::string get_pattern() const;

  /**
   * @brief Main logging method
   *
   * @param[in] level logging level of this message
   * @param[in] fmt   C-like format string, followed by respective params
   */
  void log(int level, const char* fmt, ...);

  /**
   * @brief Flush logs by calling flush on underlying logger
   */
  void flush();

  ~logger();

 private:
  logger();
  // pimpl pattern:
  // https://learn.microsoft.com/en-us/cpp/cpp/pimpl-for-compile-time-encapsulation-modern-cpp?view=msvc-170
  class impl;
  std::unique_ptr<impl> pimpl;
  static inline std::unordered_map<std::string, std::shared_ptr<raft::logger>> log_map;
};  // class logger

/**
 * @brief An object used for scoped log level setting
 *
 * Instances of `raft::log_level_setter` will set RAFT logging to the level
 * indicated on construction and will revert to the previous set level on
 * destruction.
 */
struct log_level_setter {
  explicit log_level_setter(int level)
  {
    prev_level_ = logger::get(RAFT_NAME).get_level();
    logger::get(RAFT_NAME).set_level(level);
  }
  ~log_level_setter() { logger::get(RAFT_NAME).set_level(prev_level_); }

 private:
  int prev_level_;
};  // class log_level_setter

};  // namespace raft
