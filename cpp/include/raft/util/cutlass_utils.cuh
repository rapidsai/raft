/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cutlass/cutlass.h>
#include <raft/core/error.hpp>

namespace raft {

/**
 * @brief Exception thrown when a CUDA error is encountered.
 */
struct cutlass_error : public raft::exception {
  explicit cutlass_error(char const* const message) : raft::exception(message) {}
  explicit cutlass_error(std::string const& message) : raft::exception(message) {}
};

}  // namespace raft

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
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
