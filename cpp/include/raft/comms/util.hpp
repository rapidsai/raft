/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <nccl.h>
#include <raft/error.hpp>
#include <string>

/**
 * @brief Error checking macro for NCCL runtime API functions.
 *
 * Invokes a NCCL runtime API function call, if the call does not return ncclSuccess, throws an
 * exception detailing the NCCL error that occurred
 */
#define NCCL_TRY(call)                                                        \
  do {                                                                        \
    ncclResult_t const status = (call);                                       \
    if (ncclSuccess != status) {                                              \
      std::string msg{};                                                      \
      SET_ERROR_MSG(msg,                                                      \
                    "NCCL error encountered at: ", "call='%s', Reason=%d:%s", \
                    #call, status, ncclGetErrorString(status));               \
      throw raft::logic_error(msg);                                           \
    }                                                                         \
  } while (0);

#define NCCL_TRY_NO_THROW(call)                           \
  do {                                                    \
    ncclResult_t status = call;                           \
    if (ncclSuccess != status) {                          \
      printf("NCCL call='%s' failed. Reason:%s\n", #call, \
             ncclGetErrorString(status));                 \
    }                                                     \
  } while (0)

namespace raft {
namespace comms {

constexpr size_t get_datatype_size(const DataType datatype) {
  switch (datatype) {
    case DataType::kChar:
      return sizeof(char);
    case DataType::kUint8:
      return sizeof(uint8_t);
    case DataType::kInt32:
      return sizeof(int);
    case DataType::kUint32:
      return sizeof(unsigned int);
    case DataType::kInt64:
      return sizeof(int64_t);
    case DataType::kUint64:
      return sizeof(uint64_t);
    case DataType::kFloat32:
      return sizeof(float);
    case DataType::kFloat64:
      return sizeof(double);
    default:
      throw "Unsupported datatype";
  }
}

constexpr ncclDataType_t get_nccl_datatype(const DataType datatype) {
  switch (datatype) {
    case DataType::kChar:
      return ncclChar;
    case DataType::kUint8:
      return ncclUint8;
    case DataType::kInt32:
      return ncclInt;
    case DataType::kUint32:
      return ncclUint32;
    case DataType::kInt64:
      return ncclInt64;
    case DataType::kUint64:
      return ncclUint64;
    case DataType::kFloat32:
      return ncclFloat;
    case DataType::kFloat64:
      return ncclDouble;
    default:
      throw "Unsupported datatype";
  }
}

constexpr ncclRedOp_t get_nccl_op(const Op op) {
  switch (op) {
    case Op::kSum:
      return ncclSum;
    case Op::kProd:
      return ncclProd;
    case Op::kMin:
      return ncclMin;
    case Op::kMax:
      return ncclMax;
    default:
      throw "Unsupported datatype";
  }
}
};  // namespace comms
};  // namespace raft
