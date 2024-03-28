/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#ifndef __RAFT_RT_CUBLAS_MACROS_H
#define __RAFT_RT_CUBLAS_MACROS_H

#pragma once

#include <raft/core/error.hpp>

#include <cublas_v2.h>

///@todo: enable this once we have logger enabled
// #include <cuml/common/logger.hpp>

#include <cstdint>

#define _CUBLAS_ERR_TO_STR(err) \
  case err: return #err

namespace raft {

/**
 * @addtogroup error_handling
 * @{
 */

/**
 * @brief Exception thrown when a cuBLAS error is encountered.
 */
struct cublas_error : public raft::exception {
  explicit cublas_error(char const* const message) : raft::exception(message) {}
  explicit cublas_error(std::string const& message) : raft::exception(message) {}
};

/**
 * @}
 */

namespace linalg {
namespace detail {

inline const char* cublas_error_to_string(cublasStatus_t err)
{
  switch (err) {
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_SUCCESS);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_NOT_INITIALIZED);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_ALLOC_FAILED);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_INVALID_VALUE);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_ARCH_MISMATCH);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_MAPPING_ERROR);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_EXECUTION_FAILED);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_INTERNAL_ERROR);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_NOT_SUPPORTED);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_LICENSE_ERROR);
    default: return "CUBLAS_STATUS_UNKNOWN";
  };
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft

#undef _CUBLAS_ERR_TO_STR

/**
 * @addtogroup assertion
 * @{
 */

/**
 * @brief Error checking macro for cuBLAS runtime API functions.
 *
 * Invokes a cuBLAS runtime API function call, if the call does not return
 * CUBLAS_STATUS_SUCCESS, throws an exception detailing the cuBLAS error that occurred
 */
#define RAFT_CUBLAS_TRY(call)                                              \
  do {                                                                     \
    cublasStatus_t const status = (call);                                  \
    if (CUBLAS_STATUS_SUCCESS != status) {                                 \
      std::string msg{};                                                   \
      SET_ERROR_MSG(msg,                                                   \
                    "cuBLAS error encountered at: ",                       \
                    "call='%s', Reason=%d:%s",                             \
                    #call,                                                 \
                    status,                                                \
                    raft::linalg::detail::cublas_error_to_string(status)); \
      throw raft::cublas_error(msg);                                       \
    }                                                                      \
  } while (0)

// FIXME: Remove after consumers rename
#ifndef CUBLAS_TRY
#define CUBLAS_TRY(call) RAFT_CUBLAS_TRY(call)
#endif

// /**
//  * @brief check for cuda runtime API errors but log error instead of raising
//  *        exception.
//  */
#define RAFT_CUBLAS_TRY_NO_THROW(call)                               \
  do {                                                               \
    cublasStatus_t const status = call;                              \
    if (CUBLAS_STATUS_SUCCESS != status) {                           \
      printf("CUBLAS call='%s' at file=%s line=%d failed with %s\n", \
             #call,                                                  \
             __FILE__,                                               \
             __LINE__,                                               \
             raft::linalg::detail::cublas_error_to_string(status));  \
    }                                                                \
  } while (0)

/**
 * @}
 */
/** FIXME: remove after cuml rename */
#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call) CUBLAS_TRY(call)
#endif

/** FIXME: remove after cuml rename */
#ifndef CUBLAS_CHECK_NO_THROW
#define CUBLAS_CHECK_NO_THROW(call) RAFT_CUBLAS_TRY_NO_THROW(call)
#endif

#endif
