/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#ifndef __RAFT_RT_ERROR
#define __RAFT_RT_ERROR

#pragma once

#include <cstdio>
#include <execinfo.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace raft {

/**
 * @defgroup error_handling Exceptions & Error Handling
 * @{
 */

/** base exception class for the whole of raft */
class exception : public std::exception {
 public:
  /** default ctor */
  explicit exception() noexcept : std::exception(), msg_() {}

  /** copy ctor */
  exception(exception const& src) noexcept : std::exception(), msg_(src.what())
  {
    collect_call_stack();
  }

  /** ctor from an input message */
  explicit exception(std::string const msg) noexcept : std::exception(), msg_(std::move(msg))
  {
    collect_call_stack();
  }

  /** get the message associated with this exception */
  char const* what() const noexcept override { return msg_.c_str(); }

 private:
  /** message associated with this exception */
  std::string msg_;

  /** append call stack info to this exception's message for ease of debug */
  // Courtesy: https://www.gnu.org/software/libc/manual/html_node/Backtraces.html
  void collect_call_stack() noexcept
  {
#ifdef __GNUC__
    constexpr int kMaxStackDepth = 64;
    void* stack[kMaxStackDepth];  // NOLINT
    auto depth = backtrace(stack, kMaxStackDepth);
    std::ostringstream oss;
    oss << std::endl << "Obtained " << depth << " stack frames" << std::endl;
    char** strings = backtrace_symbols(stack, depth);
    if (strings == nullptr) {
      oss << "But no stack trace could be found!" << std::endl;
      msg_ += oss.str();
      return;
    }
    ///@todo: support for demangling of C++ symbol names
    for (int i = 0; i < depth; ++i) {
      oss << "#" << i << " in " << strings[i] << std::endl;
    }
    free(strings);
    msg_ += oss.str();
#endif  // __GNUC__
  }
};

/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * RAFT_EXPECTS and  RAFT_FAIL macros.
 *
 */
struct logic_error : public raft::exception {
  explicit logic_error(char const* const message) : raft::exception(message) {}
  explicit logic_error(std::string const& message) : raft::exception(message) {}
};

/**
 * @brief Exception thrown when attempting to use CUDA features from a non-CUDA
 * build
 *
 */
struct non_cuda_build_error : public raft::exception {
  explicit non_cuda_build_error(char const* const message) : raft::exception(message) {}
  explicit non_cuda_build_error(std::string const& message) : raft::exception(message) {}
};

/**
 * @}
 */

}  // namespace raft

// FIXME: Need to be replaced with RAFT_FAIL
/** macro to throw a runtime error */
#define THROW(fmt, ...)                                                                       \
  do {                                                                                        \
    int size1 =                                                                               \
      std::snprintf(nullptr, 0, "exception occurred! file=%s line=%d: ", __FILE__, __LINE__); \
    int size2 = std::snprintf(nullptr, 0, fmt, ##__VA_ARGS__);                                \
    if (size1 < 0 || size2 < 0)                                                               \
      throw raft::exception("Error in snprintf, cannot handle raft exception.");              \
    auto size = size1 + size2 + 1; /* +1 for final '\0' */                                    \
    auto buf  = std::make_unique<char[]>(size_t(size));                                       \
    std::snprintf(buf.get(),                                                                  \
                  size1 + 1 /* +1 for '\0' */,                                                \
                  "exception occurred! file=%s line=%d: ",                                    \
                  __FILE__,                                                                   \
                  __LINE__);                                                                  \
    std::snprintf(buf.get() + size1, size2 + 1 /* +1 for '\0' */, fmt, ##__VA_ARGS__);        \
    std::string msg(buf.get(), buf.get() + size - 1); /* -1 to remove final '\0' */           \
    throw raft::exception(msg);                                                               \
  } while (0)

// FIXME: Need to be replaced with RAFT_EXPECTS
/** macro to check for a conditional and assert on failure */
#define ASSERT(check, fmt, ...)              \
  do {                                       \
    if (!(check)) THROW(fmt, ##__VA_ARGS__); \
  } while (0)

/**
 * Macro to append error message to first argument.
 * This should only be called in contexts where it is OK to throw exceptions!
 */
#define SET_ERROR_MSG(msg, location_prefix, fmt, ...)                                            \
  do {                                                                                           \
    int size1 = std::snprintf(nullptr, 0, "%s", location_prefix);                                \
    int size2 = std::snprintf(nullptr, 0, "file=%s line=%d: ", __FILE__, __LINE__);              \
    int size3 = std::snprintf(nullptr, 0, fmt, ##__VA_ARGS__);                                   \
    if (size1 < 0 || size2 < 0 || size3 < 0)                                                     \
      throw raft::exception("Error in snprintf, cannot handle raft exception.");                 \
    auto size = size1 + size2 + size3 + 1; /* +1 for final '\0' */                               \
    std::vector<char> buf(size);                                                                 \
    std::snprintf(buf.data(), size1 + 1 /* +1 for '\0' */, "%s", location_prefix);               \
    std::snprintf(                                                                               \
      buf.data() + size1, size2 + 1 /* +1 for '\0' */, "file=%s line=%d: ", __FILE__, __LINE__); \
    std::snprintf(buf.data() + size1 + size2, size3 + 1 /* +1 for '\0' */, fmt, ##__VA_ARGS__);  \
    msg += std::string(buf.data(), buf.data() + size - 1); /* -1 to remove final '\0' */         \
  } while (0)

/**
 * @defgroup assertion Assertion and error macros
 * @{
 */

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when a condition is false
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] fmt String literal description of the reason that cond is expected to be true with
 * optional format tagas
 * @throw raft::logic_error if the condition evaluates to false.
 */
#define RAFT_EXPECTS(cond, fmt, ...)                              \
  do {                                                            \
    if (!(cond)) {                                                \
      std::string msg{};                                          \
      SET_ERROR_MSG(msg, "RAFT failure at ", fmt, ##__VA_ARGS__); \
      throw raft::logic_error(msg);                               \
    }                                                             \
  } while (0)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * @param[in] fmt String literal description of the reason that this code path is erroneous with
 * optional format tagas
 * @throw always throws raft::logic_error
 */
#define RAFT_FAIL(fmt, ...)                                     \
  do {                                                          \
    std::string msg{};                                          \
    SET_ERROR_MSG(msg, "RAFT failure at ", fmt, ##__VA_ARGS__); \
    throw raft::logic_error(msg);                               \
  } while (0)

/**
 * @}
 */

#endif
