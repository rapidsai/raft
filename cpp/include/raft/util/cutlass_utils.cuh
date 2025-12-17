/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
