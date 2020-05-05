/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>
#include <execinfo.h>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
///@todo: enable once logging has been enabled in raft
//#include "logger.hpp"

namespace raft {

/** base exception class for the whole of raft */
class Exception : public std::exception {
 public:
  /** default ctor */
  Exception() throw() : std::exception(), msg() {}

  /** copy ctor */
  Exception(const Exception& src) throw() : std::exception(), msg(src.what()) {
    collectCallStack();
  }

  /** ctor from an input message */
  Exception(const std::string& _msg) throw() : std::exception(), msg(_msg) {
    collectCallStack();
  }

  /** dtor */
  virtual ~Exception() throw() {}

  /** get the message associated with this exception */
  virtual const char* what() const throw() { return msg.c_str(); }

 private:
  /** message associated with this exception */
  std::string msg;

  /** append call stack info to this exception's message for ease of debug */
  // Courtesy: https://www.gnu.org/software/libc/manual/html_node/Backtraces.html
  void collectCallStack() throw() {
#ifdef __GNUC__
    const int MaxStackDepth = 64;
    void* stack[MaxStackDepth];
    auto depth = backtrace(stack, MaxStackDepth);
    std::ostringstream oss;
    oss << std::endl << "Obtained " << depth << " stack frames" << std::endl;
    char** strings = backtrace_symbols(stack, depth);
    if (strings == nullptr) {
      oss << "But no stack trace could be found!" << std::endl;
      msg += oss.str();
      return;
    }
    ///@todo: support for demangling of C++ symbol names
    for (int i = 0; i < depth; ++i) {
      oss << "#" << i << " in " << strings[i] << std::endl;
    }
    free(strings);
    msg += oss.str();
#endif  // __GNUC__
  }
};

/** macro to throw a runtime error */
#define THROW(fmt, ...)                                                        \
  do {                                                                         \
    std::string msg;                                                           \
    char errMsg[2048];                                                         \
    std::snprintf(errMsg, sizeof(errMsg),                                      \
                  "Exception occured! file=%s line=%d: ", __FILE__, __LINE__); \
    msg += errMsg;                                                             \
    std::snprintf(errMsg, sizeof(errMsg), fmt, ##__VA_ARGS__);                 \
    msg += errMsg;                                                             \
    throw raft::Exception(msg);                                                \
  } while (0)

/** macro to check for a conditional and assert on failure */
#define ASSERT(check, fmt, ...)              \
  do {                                       \
    if (!(check)) THROW(fmt, ##__VA_ARGS__); \
  } while (0)

/** check for cuda runtime API errors and assert accordingly */
#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t status = call;                                         \
    ASSERT(status == cudaSuccess, "FAIL: call='%s'. Reason:%s", #call, \
           cudaGetErrorString(status));                                \
  } while (0)

///@todo: enable this only after we have added logging support in raft
// /**
//  * @brief check for cuda runtime API errors but log error instead of raising
//  *        exception.
//  */
// #define CUDA_CHECK_NO_THROW(call)                                            \
//   do {                                                                       \
//     cudaError_t status = call;                                               \
//     if (status != cudaSuccess) {                                             \
//       RAFT_LOG_ERROR("CUDA call='%s' at file=%s line=%d failed with %s ",    \
//                      #call, __FILE__, __LINE__, cudaGetErrorString(status)); \
//     }                                                                        \
//   } while (0)

/** helper method to get max usable shared mem per block parameter */
inline int getSharedMemPerBlock() {
  int devId;
  CUDA_CHECK(cudaGetDevice(&devId));
  int smemPerBlk;
  CUDA_CHECK(cudaDeviceGetAttribute(&smemPerBlk,
                                    cudaDevAttrMaxSharedMemoryPerBlock, devId));
  return smemPerBlk;
}
/** helper method to get multi-processor count parameter */
inline int getMultiProcessorCount() {
  int devId;
  CUDA_CHECK(cudaGetDevice(&devId));
  int mpCount;
  CUDA_CHECK(
    cudaDeviceGetAttribute(&mpCount, cudaDevAttrMultiProcessorCount, devId));
  return mpCount;
}

/**
 * @brief Generic copy method for all kinds of transfers
 * @tparam Type data type
 * @param dst destination pointer
 * @param src source pointer
 * @param len lenth of the src/dst buffers in terms of number of elements
 * @param stream cuda stream
 */
template <typename Type>
void copy(Type* dst, const Type* src, size_t len, cudaStream_t stream) {
  CUDA_CHECK(
    cudaMemcpyAsync(dst, src, len * sizeof(Type), cudaMemcpyDefault, stream));
}

/**
 * @defgroup Copy Copy methods
 * These are here along with the generic 'copy' method in order to improve
 * code readability using explicitly specified function names
 * @{
 */
/** performs a host to device copy */
template <typename Type>
void updateDevice(Type* dPtr, const Type* hPtr, size_t len,
                  cudaStream_t stream) {
  copy(dPtr, hPtr, len, stream);
}

/** performs a device to host copy */
template <typename Type>
void updateHost(Type* hPtr, const Type* dPtr, size_t len, cudaStream_t stream) {
  copy(hPtr, dPtr, len, stream);
}

template <typename Type>
void copyAsync(Type* dPtr1, const Type* dPtr2, size_t len,
               cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(dPtr1, dPtr2, len * sizeof(Type),
                             cudaMemcpyDeviceToDevice, stream));
}
/** @} */

/**
 * @defgroup Debug Utils for debugging device buffers
 * @{
 */
template <class T, class OutStream>
void printHostVector(const char* variableName, const T* hostMem,
                     size_t componentsCount, OutStream& out) {
  out << variableName << "=[";
  for (size_t i = 0; i < componentsCount; ++i) {
    if (i != 0) out << ",";
    out << hostMem[i];
  }
  out << "];\n";
}

template <class T>
void printHostVector(const char* variableName, const T* hostMem,
                     size_t componentsCount) {
  printHostVector(variableName, hostMem, componentsCount, std::cout);
  std::cout.flush();
}

template <class T, class OutStream>
void printDevVector(const char* variableName, const T* devMem,
                    size_t componentsCount, OutStream& out) {
  T* hostMem = new T[componentsCount];
  CUDA_CHECK(cudaMemcpy(hostMem, devMem, componentsCount * sizeof(T),
                        cudaMemcpyDeviceToHost));
  printHostVector(variableName, hostMem, componentsCount, out);
  delete[] hostMem;
}

template <class T>
void printDevVector(const char* variableName, const T* devMem,
                    size_t componentsCount) {
  printDevVector(variableName, devMem, componentsCount, std::cout);
  std::cout.flush();
}
/** @} */

};  // namespace raft
