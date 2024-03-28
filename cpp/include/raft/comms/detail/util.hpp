/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <raft/core/interruptible.hpp>

#include <nccl.h>

#include <string>

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

namespace raft {
namespace comms {
namespace detail {

constexpr size_t get_datatype_size(const datatype_t datatype)
{
  switch (datatype) {
    case datatype_t::CHAR: return sizeof(char);
    case datatype_t::UINT8: return sizeof(uint8_t);
    case datatype_t::INT32: return sizeof(int);
    case datatype_t::UINT32: return sizeof(unsigned int);
    case datatype_t::INT64: return sizeof(int64_t);
    case datatype_t::UINT64: return sizeof(uint64_t);
    case datatype_t::FLOAT32: return sizeof(float);
    case datatype_t::FLOAT64: return sizeof(double);
    default: throw "Unsupported datatype";
  }
}

constexpr ncclDataType_t get_nccl_datatype(const datatype_t datatype)
{
  switch (datatype) {
    case datatype_t::CHAR: return ncclChar;
    case datatype_t::UINT8: return ncclUint8;
    case datatype_t::INT32: return ncclInt;
    case datatype_t::UINT32: return ncclUint32;
    case datatype_t::INT64: return ncclInt64;
    case datatype_t::UINT64: return ncclUint64;
    case datatype_t::FLOAT32: return ncclFloat;
    case datatype_t::FLOAT64: return ncclDouble;
    default: throw "Unsupported datatype";
  }
}

constexpr ncclRedOp_t get_nccl_op(const op_t op)
{
  switch (op) {
    case op_t::SUM: return ncclSum;
    case op_t::PROD: return ncclProd;
    case op_t::MIN: return ncclMin;
    case op_t::MAX: return ncclMax;
    default: throw "Unsupported datatype";
  }
}

inline status_t nccl_sync_stream(ncclComm_t comm, cudaStream_t stream)
{
  cudaError_t cudaErr;
  ncclResult_t ncclErr, ncclAsyncErr;
  while (1) {
    cudaErr = cudaStreamQuery(stream);
    if (cudaErr == cudaSuccess) return status_t::SUCCESS;

    if (cudaErr != cudaErrorNotReady) {
      // An error occurred querying the status of the stream_
      return status_t::ERROR;
    }

    ncclErr = ncclCommGetAsyncError(comm, &ncclAsyncErr);
    if (ncclErr != ncclSuccess) {
      // An error occurred retrieving the asynchronous error
      return status_t::ERROR;
    }

    if (ncclAsyncErr != ncclSuccess || !interruptible::yield_no_throw()) {
      // An asynchronous error happened. Stop the operation and destroy
      // the communicator
      ncclErr = ncclCommAbort(comm);
      if (ncclErr != ncclSuccess)
        // Caller may abort with an exception or try to re-create a new communicator.
        return status_t::ABORT;
      // TODO: shouldn't we place status_t::ERROR above under the condition, and
      //       status_t::ABORT below here (i.e. after successful ncclCommAbort)?
    }

    // Let other threads (including NCCL threads) use the CPU.
    std::this_thread::yield();
  }
}

};  // namespace detail
};  // namespace comms
};  // namespace raft
