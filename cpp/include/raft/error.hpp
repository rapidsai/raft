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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <cusparse_v2.h>
#include <nccl.h>

#include <stdexcept>
#include <string>


namespace raft {

/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * RAFT_EXPECTS, RAFT_FAIL, CUML_EXPECTS, CUML_FAIL, CUGRAPH_EXPECTS, CUGRAPH_FAIL macros.
 *
 */
struct logic_error : public std::logic_error {
  explicit logic_error(char const* const message) : std::logic_error(message) {}
  explicit logic_error(std::string const& message) : std::logic_error(message) {}
};

/**
 * @brief Exception thrown when a CUDA error is encountered.
 */
struct cuda_error : public std::runtime_error {
  explicit cuda_error(char const* const message) : std::runtime_error(message) {}
  explicit cuda_error(std::string const& message) : std::runtime_error(message) {}
};

/**
 * @brief Exception thrown when a cuRAND error is encountered.
 */
struct curand_error : public std::runtime_error {
  explicit curand_error(char const* const message) : std::runtime_error(message) {}
  explicit curand_error(std::string const& message) : std::runtime_error(message) {}
};

/**
 * @brief Exception thrown when a cuSparse error is encountered.
 */
struct cusparse_error : public std::runtime_error {
  explicit cusparse_error(char const* const message) : std::runtime_error(message) {}
  explicit cusparse_error(std::string const& message) : std::runtime_error(message) {}
};

/**
 * @brief Exception thrown when a NCCL error is encountered.
 */
struct nccl_error : public std::runtime_error {
  explicit nccl_error(char const* const message) : std::runtime_error(message) {}
  explicit nccl_error(std::string const& message) : std::runtime_error(message) {}
};

}  // namespace raft

#define STRINGIFY_DETAIL(x) #x
#define RAFT_STRINGIFY(x) STRINGIFY_DETAIL(x)

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when
 * a condition is violated.
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] reason String literal description of the reason that cond is
 * expected to be true
 * @throw raft::logic_error if the condition evaluates to false.
 */
#define RAFT_EXPECTS(cond, reason)                                  \
  (!!(cond)) ? static_cast<void>(0)                                 \
             : throw raft::logic_error("RAFT failure at: " __FILE__ \
                                       ":" RAFT_STRINGIFY(__LINE__) ": " reason)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * In host code, throws a `raft::logic_error`.
 *
 * @param[in] reason String literal description of the reason
 */
#define RAFT_FAIL(reason) \
  throw raft::logic_error("RAFT failure at: " __FILE__ ":" RAFT_STRINGIFY(__LINE__) ": " reason)

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when
 * a condition is violated.
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] reason String literal description of the reason that cond is
 * expected to be true
 * @throw raft::logic_error if the condition evaluates to false.
 */
#define CUML_EXPECTS(cond, reason)                                  \
  (!!(cond)) ? static_cast<void>(0)                                 \
             : throw raft::logic_error("cuML failure at: " __FILE__ \
                                       ":" RAFT_STRINGIFY(__LINE__) ": " reason)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * In host code, throws a `raft::logic_error`.
 *
 * @param[in] reason String literal description of the reason
 */
#define CUML_FAIL(reason) \
  throw raft::logic_error("cuML failure at: " __FILE__ ":" RAFT_STRINGIFY(__LINE__) ": " reason)

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when
 * a condition is violated.
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] reason String literal description of the reason that cond is
 * expected to be true
 * @throw raft::logic_error if the condition evaluates to false.
 */
#define CUGRAPH_EXPECTS(cond, reason)                                  \
  (!!(cond)) ? static_cast<void>(0)                                    \
             : throw raft::logic_error("cuGRAPH failure at: " __FILE__ \
                                       ":" RAFT_STRINGIFY(__LINE__) ": " reason)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * In host code, throws a `raft::logic_error`.
 *
 * @param[in] reason String literal description of the reason
 */
#define CUGRAPH_FAIL(reason) \
  throw raft::logic_error("cuGRAPH failure at: " __FILE__ ":" RAFT_STRINGIFY(__LINE__) ": " reason)

namespace raft {
namespace detail {

inline void throw_cuda_error(cudaError_t error, const char* file, unsigned int line)
{
  throw raft::cuda_error(std::string{"CUDA error encountered at: " + std::string{file} + ":" +
                                     std::to_string(line) + ": " + std::to_string(error) + " " +
                                     cudaGetErrorName(error) + " " + cudaGetErrorString(error)});
}

inline void throw_nccl_error(ncclResult_t error, const char* file, unsigned int line) {
  throw raft::nccl_error(
    std::string{"NCCL error encountered at: " + std::string{file} + ":" +
                std::to_string(line) + ": " + std::to_string(error) + " " +
                ncclGetErrorString(error)});
}

#define _CURAND_ERR_TO_STR(err) \
  case err:                       \
    return #err;
inline const char* curand_error_to_string(curandStatus_t err) {
  switch(err) {
    _CURAND_ERR_TO_STR(CURAND_STATUS_SUCCESS);
    _CURAND_ERR_TO_STR(CURAND_STATUS_VERSION_MISMATCH);
    _CURAND_ERR_TO_STR(CURAND_STATUS_NOT_INITIALIZED);
    _CURAND_ERR_TO_STR(CURAND_STATUS_ALLOCATION_FAILED);
    _CURAND_ERR_TO_STR(CURAND_STATUS_TYPE_ERROR);
    _CURAND_ERR_TO_STR(CURAND_STATUS_OUT_OF_RANGE);
    _CURAND_ERR_TO_STR(CURAND_STATUS_LENGTH_NOT_MULTIPLE);
    _CURAND_ERR_TO_STR(CURAND_STATUS_DOUBLE_PRECISION_REQUIRED);
    _CURAND_ERR_TO_STR(CURAND_STATUS_LAUNCH_FAILURE);
    _CURAND_ERR_TO_STR(CURAND_STATUS_PREEXISTING_FAILURE);
    _CURAND_ERR_TO_STR(CURAND_STATUS_INITIALIZATION_FAILED);
    _CURAND_ERR_TO_STR(CURAND_STATUS_ARCH_MISMATCH);
    _CURAND_ERR_TO_STR(CURAND_STATUS_INTERNAL_ERROR);
    default:
      return "CURAND_STATUS_UNKNOWN";
  };
}
#undef _CURAND_ERR_TO_STR

inline void throw_curand_error(curandStatus_t error, const char* file, unsigned int line) {
  throw raft::curand_error(
    std::string{"cuRAND error encountered at: " + std::string{file} + ":" +
                std::to_string(line) + ": " + std::to_string(error) + " " +
                curand_error_to_string(error)});
}

// FIXME: unnecessary once CUDA 10.1+ becomes the minimum supported version
#define _CUSPARSE_ERR_TO_STR(err) \
  case err:                       \
    return #err;
inline const char* cusparse_error_to_string(cusparseStatus_t err) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10100
  return cusparseGetErrorString(status);
#else   // CUDART_VERSION
  switch (err) {
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_SUCCESS);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_NOT_INITIALIZED);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_ALLOC_FAILED);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_INVALID_VALUE);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_ARCH_MISMATCH);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_EXECUTION_FAILED);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_INTERNAL_ERROR);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    default:
      return "CUSPARSE_STATUS_UNKNOWN";
  };
#endif  // CUDART_VERSION
}
#undef _CUSPARSE_ERR_TO_STR

inline void throw_cusparse_error(cusparseStatus_t error, const char* file, unsigned int line) {
  throw raft::cusparse_error(
    std::string{"cuSparse error encountered at: " + std::string{file} + ":" +
                std::to_string(line) + ": " + std::to_string(error) + " " +
                cusparse_error_to_string(error)});
}

}  // namespace detail
}  // namespace raft

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 */
#define CUDA_TRY(call)                                            \
  do {                                                            \
    cudaError_t const status = (call);                            \
    if (cudaSuccess != status) {                                  \
      cudaGetLastError();                                         \
      raft::detail::throw_cuda_error(status, __FILE__, __LINE__); \
    }                                                             \
  } while (0);

/**
 * @brief Debug macro to check for CUDA errors
 *
 * In a non-release build, this macro will synchronize the specified stream
 * before error checking. In both release and non-release builds, this macro
 * checks for any pending CUDA errors from previous calls. If an error is
 * reported, an exception is thrown detailing the CUDA error that occurred.
 *
 * The intent of this macro is to provide a mechanism for synchronous and
 * deterministic execution for debugging asynchronous CUDA execution. It should
 * be used after any asynchronous CUDA call, e.g., cudaMemcpyAsync, or an
 * asynchronous kernel launch.
 *
 */
#ifndef NDEBUG
#define CHECK_CUDA(stream) CUDA_TRY(cudaStreamSynchronize(stream));
#else
#define CHECK_CUDA(stream) CUDA_TRY(cudaPeekAtLastError());
#endif

/**
 * @brief Error checking macro for cuRAND runtime API functions.
 *
 * Invokes a cuRAND runtime API function call, if the call does not return
 * CURAND_STATUS_SUCCESS, throws an exception detailing the cuRAND error that occurred
 */
#define CURAND_TRY(call)                                            \
  do {                                                              \
    curandStatus_t const status = (call);                           \
    if (CURAND_STATUS_SUCCESS != status) {                          \
      raft::detail::throw_curand_error(status, __FILE__, __LINE__); \
    }                                                               \
  } while (0);

/**
 * @brief Error checking macro for cuSparse runtime API functions.
 *
 * Invokes a cuSparse runtime API function call, if the call does not return
 * CUSPARSE_STATUS_SUCCESS, throws an exception detailing the cuSparse error that occurred
 */
#define CUSPARSE_TRY(call)                                            \
  do {                                                                \
    cusparseStatus_t const status = (call);                           \
    if (CUSPARSE_STATUS_SUCCESS != status) {                          \
      raft::detail::throw_cusparse_error(status, __FILE__, __LINE__); \
    }                                                                 \
  } while (0);

/**
 * @brief Error checking macro for NCCL runtime API functions.
 *
 * Invokes a NCCL runtime API function call, if the call does not return ncclSuccess, throws an
 * exception detailing the NCCL error that occurred
 */
#define NCCL_TRY(call)                                            \
  do {                                                            \
    ncclResult_t const status = (call);                           \
    if (ncclSuccess != status) {                                  \
      raft::detail::throw_nccl_error(status, __FILE__, __LINE__); \
    }                                                             \
  } while (0);
