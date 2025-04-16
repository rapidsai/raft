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
