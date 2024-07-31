/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cusparse.h>
///@todo: enable this once logging is enabled
// #include <cuml/common/logger.hpp>

#define _CUSPARSE_ERR_TO_STR(err) \
  case err: return #err;

// Notes:
//(1.) CUDA_VER_10_1_UP aggregates all the CUDA version selection logic;
//(2.) to enforce a lower version,
//
//`#define CUDA_ENFORCE_LOWER
// #include <raft/sparse/detail/cusparse_wrappers.h>`
//
// (i.e., before including this header)
//
#define CUDA_VER_10_1_UP (CUDART_VERSION >= 10010)
#define CUDA_VER_12_4_UP (CUDART_VERSION >= 12040)

namespace raft {

/**
 * @ingroup error_handling
 * @{
 */

/**
 * @brief Exception thrown when a cuSparse error is encountered.
 */
struct cusparse_error : public raft::exception {
  explicit cusparse_error(char const* const message) : raft::exception(message) {}
  explicit cusparse_error(std::string const& message) : raft::exception(message) {}
};

/**
 * @}
 */
namespace sparse {
namespace detail {

inline const char* cusparse_error_to_string(cusparseStatus_t err)
{
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10010
  return cusparseGetErrorString(err);
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
    default: return "CUSPARSE_STATUS_UNKNOWN";
  };
#endif  // CUDART_VERSION
}

}  // namespace detail
}  // namespace sparse
}  // namespace raft

#undef _CUSPARSE_ERR_TO_STR

/**
 * @ingroup assertion
 * @{
 */

/**
 * @brief Error checking macro for cuSparse runtime API functions.
 *
 * Invokes a cuSparse runtime API function call, if the call does not return
 * CUSPARSE_STATUS_SUCCESS, throws an exception detailing the cuSparse error that occurred
 */
#define RAFT_CUSPARSE_TRY(call)                                              \
  do {                                                                       \
    cusparseStatus_t const status = (call);                                  \
    if (CUSPARSE_STATUS_SUCCESS != status) {                                 \
      std::string msg{};                                                     \
      SET_ERROR_MSG(msg,                                                     \
                    "cuSparse error encountered at: ",                       \
                    "call='%s', Reason=%d:%s",                               \
                    #call,                                                   \
                    status,                                                  \
                    raft::sparse::detail::cusparse_error_to_string(status)); \
      throw raft::cusparse_error(msg);                                       \
    }                                                                        \
  } while (0)

/**
 * @}
 */

// FIXME: Remove after consumer rename
#ifndef CUSPARSE_TRY
#define CUSPARSE_TRY(call) RAFT_CUSPARSE_TRY(call)
#endif

// FIXME: Remove after consumer rename
#ifndef CUSPARSE_CHECK
#define CUSPARSE_CHECK(call) CUSPARSE_TRY(call)
#endif

/**
 * @ingroup assertion
 * @{
 */
//@todo: use logger here once logging is enabled
/** check for cusparse runtime API errors but do not assert */
#define RAFT_CUSPARSE_TRY_NO_THROW(call)                           \
  do {                                                             \
    cusparseStatus_t err = call;                                   \
    if (err != CUSPARSE_STATUS_SUCCESS) {                          \
      printf("CUSPARSE call='%s' got errorcode=%d err=%s",         \
             #call,                                                \
             err,                                                  \
             raft::sparse::detail::cusparse_error_to_string(err)); \
    }                                                              \
  } while (0)

/**
 * @}
 */

// FIXME: Remove after consumer rename
#ifndef CUSPARSE_CHECK_NO_THROW
#define CUSPARSE_CHECK_NO_THROW(call) RAFT_CUSPARSE_TRY_NO_THROW(call)
#endif
