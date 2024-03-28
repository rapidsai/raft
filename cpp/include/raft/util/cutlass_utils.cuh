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

#include <cutlass/cutlass.h>

namespace raft {

/**
 * @brief Exception thrown when a CUTLASS error is encountered.
 */
struct cutlass_error : public raft::exception {
  explicit cutlass_error(char const* const message) : raft::exception(message) {}
  explicit cutlass_error(std::string const& message) : raft::exception(message) {}
};

}  // namespace raft

/**
 * @brief Error checking macro for CUTLASS functions.
 *
 * Invokes a CUTLASS function call, if the call does not return cutlass::Status::kSuccess,
 * throws an exception detailing the CUTLASS error that occurred.
 *
 */
#define RAFT_CUTLASS_TRY(call)                        \
  do {                                                \
    cutlass::Status const status = call;              \
    if (status != cutlass::Status::kSuccess) {        \
      std::string msg{};                              \
      SET_ERROR_MSG(msg,                              \
                    "CUTLASS error encountered at: ", \
                    "call='%s', Reason=%s",           \
                    #call,                            \
                    cutlassGetStatusString(status));  \
      throw raft::cutlass_error(msg);                 \
    }                                                 \
  } while (0)
