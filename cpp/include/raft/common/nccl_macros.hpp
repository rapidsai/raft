/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Error checking macro for NCCL runtime API functions.
 *
 * Invokes a NCCL runtime API function call, if the call does not return ncclSuccess, throws an
 * exception detailing the NCCL error that occurred
 */
#define RAFT_NCCL_TRY(call)                        \
  do {                                             \
    ncclResult_t const status = (call);            \
    if (ncclSuccess != status) {                   \
      std::string msg{};                           \
      SET_ERROR_MSG(msg,                           \
                    "NCCL error encountered at: ", \
                    "call='%s', Reason=%d:%s",     \
                    #call,                         \
                    status,                        \
                    ncclGetErrorString(status));   \
      throw raft::logic_error(msg);                \
    }                                              \
  } while (0);

// FIXME: Remove after consumer rename
#ifndef NCCL_TRY
#define NCCL_TRY(call) RAFT_NCCL_TRY(call)
#endif

#define RAFT_NCCL_TRY_NO_THROW(call)                                                   \
  do {                                                                                 \
    ncclResult_t status = call;                                                        \
    if (ncclSuccess != status) {                                                       \
      printf("NCCL call='%s' failed. Reason:%s\n", #call, ncclGetErrorString(status)); \
    }                                                                                  \
  } while (0)

// FIXME: Remove after consumer rename
#ifndef NCCL_TRY_NO_THROW
#define NCCL_TRY_NO_THROW(call) RAFT_NCCL_TRY_NO_THROW(call)
#endif
