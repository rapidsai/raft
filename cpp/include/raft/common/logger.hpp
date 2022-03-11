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

#include <algorithm>

#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include <raft/common/detail/logger.hpp>

namespace raft {

static const std::string default_log_pattern("[%L] [%H:%M:%S.%f] %v");

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
  logger(std::string const& name_ = "")
    : sink{std::make_shared<spdlog::sinks::callback_sink_mt>()},
      spdlogger{std::make_shared<spdlog::logger>(name_, sink)},
      cur_pattern()
  {
    set_pattern(default_log_pattern);
    set_level(RAFT_LEVEL_INFO);
  }
  /**
   * @brief Singleton method to get the underlying logger object
   *
   * @return the singleton logger object
   */
  static logger& get(std::string const& name = "")
  {
    if (log_map.find(name) == log_map.end()) {
      log_map[name] = std::make_shared<raft::logger>(name);
    }
    return *log_map[name];
  }

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
  void set_level(int level)
  {
    level = detail::convert_level_to_spdlog(level);
    spdlogger->set_level(static_cast<spdlog::level::level_enum>(level));
  }

  /**
   * @brief Set the logging pattern
   *
   * @param[in] pattern the pattern to be set. Refer this link
   *                    https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
   *                    to know the right syntax of this pattern
   */
  void set_pattern(const std::string& pattern)
  {
    cur_pattern = pattern;
    spdlogger->set_pattern(pattern);
  }

  /**
   * @brief Register a callback function to be run in place of usual log call
   *
   * @param[in] callback the function to be run on all logged messages
   */
  void set_callback(void (*callback)(int lvl, const char* msg)) { sink->set_callback(callback); }

  /**
   * @brief Register a flush function compatible with the registered callback
   *
   * @param[in] flush the function to use when flushing logs
   */
  void set_flush(void (*flush)()) { sink->set_flush(flush); }

  /**
   * @brief Tells whether messages will be logged for the given log level
   *
   * @param[in] level log level to be checked for
   * @return true if messages will be logged for this level, else false
   */
  bool should_log_for(int level) const
  {
    level        = detail::convert_level_to_spdlog(level);
    auto level_e = static_cast<spdlog::level::level_enum>(level);
    return spdlogger->should_log(level_e);
  }

  /**
   * @brief Query for the current log level
   *
   * @return the current log level
   */
  int get_level() const
  {
    auto level_e = spdlogger->level();
    return RAFT_LEVEL_TRACE - static_cast<int>(level_e);
  }

  /**
   * @brief Get the current logging pattern
   * @return the pattern
   */
  std::string get_pattern() const { return cur_pattern; }

  /**
   * @brief Main logging method
   *
   * @param[in] level logging level of this message
   * @param[in] fmt   C-like format string, followed by respective params
   */
  void log(int level, const char* fmt, ...)
  {
    level        = detail::convert_level_to_spdlog(level);
    auto level_e = static_cast<spdlog::level::level_enum>(level);
    // explicit check to make sure that we only expand messages when required
    if (spdlogger->should_log(level_e)) {
      va_list vl;
      va_start(vl, fmt);
      auto msg = detail::format(fmt, vl);
      va_end(vl);
      spdlogger->log(level_e, msg);
    }
  }

  /**
   * @brief Flush logs by calling flush on underlying logger
   */
  void flush() { spdlogger->flush(); }

  ~logger() {}

 private:
  logger();

  static inline std::unordered_map<std::string, std::shared_ptr<raft::logger>> log_map;
  std::shared_ptr<spdlog::sinks::callback_sink_mt> sink;
  std::shared_ptr<spdlog::logger> spdlogger;
  std::string cur_pattern;
  int cur_level;
};  // class logger

/**
 * @brief RAII based pattern setter for logger class
 *
 * @code{.cpp}
 * {
 *   PatternSetter _("%l -- %v");
 *   RAFT_LOG_INFO("Test message\n");
 * }
 * @endcode
 */
class PatternSetter {
 public:
  /**
   * @brief Set the pattern for the rest of the log messages
   * @param[in] pattern pattern to be set
   */
  PatternSetter(const std::string& pattern = "%v") : prev_pattern()
  {
    prev_pattern = logger::get().get_pattern();
    logger::get().set_pattern(pattern);
  }

  /**
   * @brief This will restore the previous pattern that was active during the
   *        moment this object was created
   */
  ~PatternSetter() { logger::get().set_pattern(prev_pattern); }

 private:
  std::string prev_pattern;
};  // class PatternSetter

};  // namespace raft