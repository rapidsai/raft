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

#include "logger-macros.hpp"

#include <stdarg.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
// The logger-ext.hpp file contains the class declaration of the logger class.
// In this case, it is okay to include the logger-ext.hpp file because it
// contains no RAFT_EXPLICIT template instantiations.
#include "logger-ext.hpp"

#define SPDLOG_HEADER_ONLY
#include <raft/core/detail/callback_sink.hpp>
#include <raft/core/detail/macros.hpp>  // RAFT_INLINE_CONDITIONAL

#include <spdlog/sinks/stdout_color_sinks.h>  // NOLINT
#include <spdlog/spdlog.h>                    // NOLINT

namespace raft {

namespace detail {

inline std::string format(const char* fmt, va_list& vl)
{
  va_list vl_copy;
  va_copy(vl_copy, vl);
  int length = std::vsnprintf(nullptr, 0, fmt, vl_copy);
  assert(length >= 0);
  std::vector<char> buf(length + 1);
  std::vsnprintf(buf.data(), length + 1, fmt, vl);
  return std::string(buf.data());
}

RAFT_INLINE_CONDITIONAL std::string format(const char* fmt, ...)
{
  va_list vl;
  va_start(vl, fmt);
  std::string str = format(fmt, vl);
  va_end(vl);
  return str;
}

inline int convert_level_to_spdlog(int level)
{
  level = std::max(RAFT_LEVEL_OFF, std::min(RAFT_LEVEL_TRACE, level));
  return RAFT_LEVEL_TRACE - level;
}

}  // namespace detail

class logger::impl {  // defined privately here
                      // ... all private data and functions: all of these
                      //     can now change without recompiling callers ...
 public:
  std::shared_ptr<spdlog::sinks::callback_sink_mt> sink;
  std::shared_ptr<spdlog::logger> spdlogger;
  std::string cur_pattern;
  int cur_level;

  impl(std::string const& name_ = "")
    : sink{std::make_shared<spdlog::sinks::callback_sink_mt>()},
      spdlogger{std::make_shared<spdlog::logger>(name_, sink)},
      cur_pattern()
  {
  }
};  // class logger::impl

RAFT_INLINE_CONDITIONAL logger::logger(std::string const& name_) : pimpl(new impl(name_))
{
  set_pattern(default_log_pattern);
  set_level(RAFT_ACTIVE_LEVEL);
}

RAFT_INLINE_CONDITIONAL logger& logger::get(std::string const& name)
{
  if (log_map.find(name) == log_map.end()) { log_map[name] = std::make_shared<raft::logger>(name); }
  return *log_map[name];
}

RAFT_INLINE_CONDITIONAL void logger::set_level(int level)
{
  level = raft::detail::convert_level_to_spdlog(level);
  pimpl->spdlogger->set_level(static_cast<spdlog::level::level_enum>(level));
}

RAFT_INLINE_CONDITIONAL void logger::set_pattern(const std::string& pattern)
{
  pimpl->cur_pattern = pattern;
  pimpl->spdlogger->set_pattern(pattern);
}

RAFT_INLINE_CONDITIONAL void logger::set_callback(void (*callback)(int lvl, const char* msg))
{
  pimpl->sink->set_callback(callback);
}

RAFT_INLINE_CONDITIONAL void logger::set_flush(void (*flush)()) { pimpl->sink->set_flush(flush); }

RAFT_INLINE_CONDITIONAL bool logger::should_log_for(int level) const
{
  level        = raft::detail::convert_level_to_spdlog(level);
  auto level_e = static_cast<spdlog::level::level_enum>(level);
  return pimpl->spdlogger->should_log(level_e);
}

RAFT_INLINE_CONDITIONAL int logger::get_level() const
{
  auto level_e = pimpl->spdlogger->level();
  return RAFT_LEVEL_TRACE - static_cast<int>(level_e);
}

RAFT_INLINE_CONDITIONAL std::string logger::get_pattern() const { return pimpl->cur_pattern; }

RAFT_INLINE_CONDITIONAL void logger::log(int level, const char* fmt, ...)
{
  level        = raft::detail::convert_level_to_spdlog(level);
  auto level_e = static_cast<spdlog::level::level_enum>(level);
  // explicit check to make sure that we only expand messages when required
  if (pimpl->spdlogger->should_log(level_e)) {
    va_list vl;
    va_start(vl, fmt);
    auto msg = raft::detail::format(fmt, vl);
    va_end(vl);
    pimpl->spdlogger->log(level_e, msg);
  }
}

RAFT_INLINE_CONDITIONAL void logger::flush() { pimpl->spdlogger->flush(); }

RAFT_INLINE_CONDITIONAL logger::~logger() {}

};  // namespace raft
