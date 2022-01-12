/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <raft/error.hpp>

#include <cusparse_v2.h>
///@todo: enable this once logging is enabled
//#include <cuml/common/logger.hpp>

#define _CUSPARSE_ERR_TO_STR(err) \
  case err: return #err;

// Notes:
//(1.) CUDA_VER_10_1_UP aggregates all the CUDA version selection logic;
//(2.) to enforce a lower version,
//
//`#define CUDA_ENFORCE_LOWER
// #include <raft/sparse/cusparse_wrappers.h>`
//
// (i.e., before including this header)
//
#define CUDA_VER_10_1_UP (CUDART_VERSION >= 10100)

namespace raft {

/**
 * @brief Exception thrown when a cuSparse error is encountered.
 */
struct cusparse_error : public raft::exception {
  explicit cusparse_error(char const* const message) : raft::exception(message) {}
  explicit cusparse_error(std::string const& message) : raft::exception(message) {}
};

namespace sparse {
namespace detail {

inline const char* cusparse_error_to_string(cusparseStatus_t err)
{
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10100
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

// FIXME: Remove after consumer rename
#ifndef CUSPARSE_TRY
#define CUSPARSE_TRY(call) RAFT_CUSPARSE_TRY(call)
#endif

// FIXME: Remove after consumer rename
#ifndef CUSPARSE_CHECK
#define CUSPARSE_CHECK(call) CUSPARSE_TRY(call)
#endif

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

// FIXME: Remove after consumer rename
#ifndef CUSPARSE_CHECK_NO_THROW
#define CUSPARSE_CHECK_NO_THROW(call) RAFT_CUSPARSE_TRY_NO_THROW(call)
#endif

namespace raft {
namespace sparse {

/**
 * @defgroup gthr cusparse gather methods
 * @{
 */
template <typename T>
cusparseStatus_t cusparsegthr(
  cusparseHandle_t handle, int nnz, const T* vals, T* vals_sorted, int* d_P, cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsegthr(cusparseHandle_t handle,
                                     int nnz,
                                     const double* vals,
                                     double* vals_sorted,
                                     int* d_P,
                                     cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return cusparseDgthr(handle, nnz, vals, vals_sorted, d_P, CUSPARSE_INDEX_BASE_ZERO);
#pragma GCC diagnostic pop
}
template <>
inline cusparseStatus_t cusparsegthr(cusparseHandle_t handle,
                                     int nnz,
                                     const float* vals,
                                     float* vals_sorted,
                                     int* d_P,
                                     cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return cusparseSgthr(handle, nnz, vals, vals_sorted, d_P, CUSPARSE_INDEX_BASE_ZERO);
#pragma GCC diagnostic pop
}
/** @} */

/**
 * @defgroup coo2csr cusparse COO to CSR converter methods
 * @{
 */
template <typename T>
void cusparsecoo2csr(
  cusparseHandle_t handle, const T* cooRowInd, int nnz, int m, T* csrRowPtr, cudaStream_t stream);
template <>
inline void cusparsecoo2csr(cusparseHandle_t handle,
                            const int* cooRowInd,
                            int nnz,
                            int m,
                            int* csrRowPtr,
                            cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(cusparseXcoo2csr(handle, cooRowInd, nnz, m, csrRowPtr, CUSPARSE_INDEX_BASE_ZERO));
}
/** @} */

/**
 * @defgroup coosort cusparse coo sort methods
 * @{
 */
template <typename T>
size_t cusparsecoosort_bufferSizeExt(  // NOLINT
  cusparseHandle_t handle,
  int m,
  int n,
  int nnz,
  const T* cooRows,
  const T* cooCols,
  cudaStream_t stream);
template <>
inline size_t cusparsecoosort_bufferSizeExt(  // NOLINT
  cusparseHandle_t handle,
  int m,
  int n,
  int nnz,
  const int* cooRows,
  const int* cooCols,
  cudaStream_t stream)
{
  size_t val;
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, cooRows, cooCols, &val));
  return val;
}

template <typename T>
void cusparsecoosortByRow(  // NOLINT
  cusparseHandle_t handle,
  int m,
  int n,
  int nnz,
  T* cooRows,
  T* cooCols,
  T* P,
  void* pBuffer,
  cudaStream_t stream);
template <>
inline void cusparsecoosortByRow(  // NOLINT
  cusparseHandle_t handle,
  int m,
  int n,
  int nnz,
  int* cooRows,
  int* cooCols,
  int* P,
  void* pBuffer,
  cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(cusparseXcoosortByRow(handle, m, n, nnz, cooRows, cooCols, P, pBuffer));
}
/** @} */

#if not defined CUDA_ENFORCE_LOWER and CUDA_VER_10_1_UP
/**
 * @defgroup cusparse Create CSR operations
 * @{
 */
template <typename IndexT, typename ValueT>
cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                   int64_t rows,
                                   int64_t cols,
                                   int64_t nnz,
                                   IndexT* csrRowOffsets,
                                   IndexT* csrColInd,
                                   ValueT* csrValues);
template <>
inline cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                          int64_t rows,
                                          int64_t cols,
                                          int64_t nnz,
                                          int* csrRowOffsets,
                                          int* csrColInd,
                                          float* csrValues)
{
  return cusparseCreateCsr(spMatDescr,
                           rows,
                           cols,
                           nnz,
                           csrRowOffsets,
                           csrColInd,
                           csrValues,
                           CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_32F);
}
template <>
inline cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                          int64_t rows,
                                          int64_t cols,
                                          int64_t nnz,
                                          int* csrRowOffsets,
                                          int* csrColInd,
                                          double* csrValues)
{
  return cusparseCreateCsr(spMatDescr,
                           rows,
                           cols,
                           nnz,
                           csrRowOffsets,
                           csrColInd,
                           csrValues,
                           CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_64F);
}
template <>
inline cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                          int64_t rows,
                                          int64_t cols,
                                          int64_t nnz,
                                          int64_t* csrRowOffsets,
                                          int64_t* csrColInd,
                                          float* csrValues)
{
  return cusparseCreateCsr(spMatDescr,
                           rows,
                           cols,
                           nnz,
                           csrRowOffsets,
                           csrColInd,
                           csrValues,
                           CUSPARSE_INDEX_64I,
                           CUSPARSE_INDEX_64I,
                           CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_32F);
}
template <>
inline cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                          int64_t rows,
                                          int64_t cols,
                                          int64_t nnz,
                                          int64_t* csrRowOffsets,
                                          int64_t* csrColInd,
                                          double* csrValues)
{
  return cusparseCreateCsr(spMatDescr,
                           rows,
                           cols,
                           nnz,
                           csrRowOffsets,
                           csrColInd,
                           csrValues,
                           CUSPARSE_INDEX_64I,
                           CUSPARSE_INDEX_64I,
                           CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_64F);
}
/** @} */
/**
 * @defgroup cusparse CreateDnVec operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsecreatednvec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, T* values);
template <>
inline cusparseStatus_t cusparsecreatednvec(cusparseDnVecDescr_t* dnVecDescr,
                                            int64_t size,
                                            float* values)
{
  return cusparseCreateDnVec(dnVecDescr, size, values, CUDA_R_32F);
}
template <>
inline cusparseStatus_t cusparsecreatednvec(cusparseDnVecDescr_t* dnVecDescr,
                                            int64_t size,
                                            double* values)
{
  return cusparseCreateDnVec(dnVecDescr, size, values, CUDA_R_64F);
}
/** @} */

/**
 * @defgroup cusparse CreateDnMat operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsecreatednmat(cusparseDnMatDescr_t* dnMatDescr,
                                     int64_t rows,
                                     int64_t cols,
                                     int64_t ld,
                                     T* values,
                                     cusparseOrder_t order);
template <>
inline cusparseStatus_t cusparsecreatednmat(cusparseDnMatDescr_t* dnMatDescr,
                                            int64_t rows,
                                            int64_t cols,
                                            int64_t ld,
                                            float* values,
                                            cusparseOrder_t order)
{
  return cusparseCreateDnMat(dnMatDescr, rows, cols, ld, values, CUDA_R_32F, order);
}
template <>
inline cusparseStatus_t cusparsecreatednmat(cusparseDnMatDescr_t* dnMatDescr,
                                            int64_t rows,
                                            int64_t cols,
                                            int64_t ld,
                                            double* values,
                                            cusparseOrder_t order)
{
  return cusparseCreateDnMat(dnMatDescr, rows, cols, ld, values, CUDA_R_64F, order);
}
/** @} */

/**
 * @defgroup Csrmv cusparse SpMV operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsespmv_buffersize(cusparseHandle_t handle,
                                         cusparseOperation_t opA,
                                         const T* alpha,
                                         const cusparseSpMatDescr_t matA,
                                         const cusparseDnVecDescr_t vecX,
                                         const T* beta,
                                         const cusparseDnVecDescr_t vecY,
                                         cusparseSpMVAlg_t alg,
                                         size_t* bufferSize,
                                         cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsespmv_buffersize(cusparseHandle_t handle,
                                                cusparseOperation_t opA,
                                                const float* alpha,
                                                const cusparseSpMatDescr_t matA,
                                                const cusparseDnVecDescr_t vecX,
                                                const float* beta,
                                                const cusparseDnVecDescr_t vecY,
                                                cusparseSpMVAlg_t alg,
                                                size_t* bufferSize,
                                                cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMV_bufferSize(
    handle, opA, alpha, matA, vecX, beta, vecY, CUDA_R_32F, alg, bufferSize);
}
template <>
inline cusparseStatus_t cusparsespmv_buffersize(cusparseHandle_t handle,
                                                cusparseOperation_t opA,
                                                const double* alpha,
                                                const cusparseSpMatDescr_t matA,
                                                const cusparseDnVecDescr_t vecX,
                                                const double* beta,
                                                const cusparseDnVecDescr_t vecY,
                                                cusparseSpMVAlg_t alg,
                                                size_t* bufferSize,
                                                cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMV_bufferSize(
    handle, opA, alpha, matA, vecX, beta, vecY, CUDA_R_64F, alg, bufferSize);
}

template <typename T>
cusparseStatus_t cusparsespmv(cusparseHandle_t handle,
                              cusparseOperation_t opA,
                              const T* alpha,
                              const cusparseSpMatDescr_t matA,
                              const cusparseDnVecDescr_t vecX,
                              const T* beta,
                              const cusparseDnVecDescr_t vecY,
                              cusparseSpMVAlg_t alg,
                              T* externalBuffer,
                              cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsespmv(cusparseHandle_t handle,
                                     cusparseOperation_t opA,
                                     const float* alpha,
                                     const cusparseSpMatDescr_t matA,
                                     const cusparseDnVecDescr_t vecX,
                                     const float* beta,
                                     const cusparseDnVecDescr_t vecY,
                                     cusparseSpMVAlg_t alg,
                                     float* externalBuffer,
                                     cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, CUDA_R_32F, alg, externalBuffer);
}
template <>
inline cusparseStatus_t cusparsespmv(cusparseHandle_t handle,
                                     cusparseOperation_t opA,
                                     const double* alpha,
                                     const cusparseSpMatDescr_t matA,
                                     const cusparseDnVecDescr_t vecX,
                                     const double* beta,
                                     const cusparseDnVecDescr_t vecY,
                                     cusparseSpMVAlg_t alg,
                                     double* externalBuffer,
                                     cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, CUDA_R_64F, alg, externalBuffer);
}
/** @} */
#else
/**
 * @defgroup Csrmv cusparse csrmv operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsecsrmv(  // NOLINT
  cusparseHandle_t handle,
  cusparseOperation_t trans,
  int m,
  int n,
  int nnz,
  const T* alpha,
  const cusparseMatDescr_t descr,
  const T* csrVal,
  const int* csrRowPtr,
  const int* csrColInd,
  const T* x,
  const T* beta,
  T* y,
  cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsecsrmv(cusparseHandle_t handle,
                                      cusparseOperation_t trans,
                                      int m,
                                      int n,
                                      int nnz,
                                      const float* alpha,
                                      const cusparseMatDescr_t descr,
                                      const float* csrVal,
                                      const int* csrRowPtr,
                                      const int* csrColInd,
                                      const float* x,
                                      const float* beta,
                                      float* y,
                                      cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseScsrmv(
    handle, trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, beta, y);
}
template <>
inline cusparseStatus_t cusparsecsrmv(cusparseHandle_t handle,
                                      cusparseOperation_t trans,
                                      int m,
                                      int n,
                                      int nnz,
                                      const double* alpha,
                                      const cusparseMatDescr_t descr,
                                      const double* csrVal,
                                      const int* csrRowPtr,
                                      const int* csrColInd,
                                      const double* x,
                                      const double* beta,
                                      double* y,
                                      cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseDcsrmv(
    handle, trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, beta, y);
}
/** @} */
#endif

/**
 * @defgroup setpointermode cusparse set pointer mode method
 * @{
 */
// no T dependency...
// template <typename T>
// cusparseStatus_t cusparsesetpointermode(  // NOLINT
//                                         cusparseHandle_t handle,
//                                         cusparsePointerMode_t mode,
//                                         cudaStream_t stream);

// template<>
inline cusparseStatus_t cusparsesetpointermode(cusparseHandle_t handle,
                                               cusparsePointerMode_t mode,
                                               cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSetPointerMode(handle, mode);
}
/** @} */

/**
 * @defgroup Csr2cscEx2 cusparse csr->csc conversion
 * @{
 */

template <typename T>
cusparseStatus_t cusparsecsr2csc_bufferSize(cusparseHandle_t handle,
                                            int m,
                                            int n,
                                            int nnz,
                                            const T* csrVal,
                                            const int* csrRowPtr,
                                            const int* csrColInd,
                                            void* cscVal,
                                            int* cscColPtr,
                                            int* cscRowInd,
                                            cusparseAction_t copyValues,
                                            cusparseIndexBase_t idxBase,
                                            cusparseCsr2CscAlg_t alg,
                                            size_t* bufferSize,
                                            cudaStream_t stream);

template <>
inline cusparseStatus_t cusparsecsr2csc_bufferSize(cusparseHandle_t handle,
                                                   int m,
                                                   int n,
                                                   int nnz,
                                                   const float* csrVal,
                                                   const int* csrRowPtr,
                                                   const int* csrColInd,
                                                   void* cscVal,
                                                   int* cscColPtr,
                                                   int* cscRowInd,
                                                   cusparseAction_t copyValues,
                                                   cusparseIndexBase_t idxBase,
                                                   cusparseCsr2CscAlg_t alg,
                                                   size_t* bufferSize,
                                                   cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));

  return cusparseCsr2cscEx2_bufferSize(handle,
                                       m,
                                       n,
                                       nnz,
                                       csrVal,
                                       csrRowPtr,
                                       csrColInd,
                                       cscVal,
                                       cscColPtr,
                                       cscRowInd,
                                       CUDA_R_32F,
                                       copyValues,
                                       idxBase,
                                       alg,
                                       bufferSize);
}
template <>
inline cusparseStatus_t cusparsecsr2csc_bufferSize(cusparseHandle_t handle,
                                                   int m,
                                                   int n,
                                                   int nnz,
                                                   const double* csrVal,
                                                   const int* csrRowPtr,
                                                   const int* csrColInd,
                                                   void* cscVal,
                                                   int* cscColPtr,
                                                   int* cscRowInd,
                                                   cusparseAction_t copyValues,
                                                   cusparseIndexBase_t idxBase,
                                                   cusparseCsr2CscAlg_t alg,
                                                   size_t* bufferSize,
                                                   cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));

  return cusparseCsr2cscEx2_bufferSize(handle,
                                       m,
                                       n,
                                       nnz,
                                       csrVal,
                                       csrRowPtr,
                                       csrColInd,
                                       cscVal,
                                       cscColPtr,
                                       cscRowInd,
                                       CUDA_R_64F,
                                       copyValues,
                                       idxBase,
                                       alg,
                                       bufferSize);
}

template <typename T>
cusparseStatus_t cusparsecsr2csc(cusparseHandle_t handle,
                                 int m,
                                 int n,
                                 int nnz,
                                 const T* csrVal,
                                 const int* csrRowPtr,
                                 const int* csrColInd,
                                 void* cscVal,
                                 int* cscColPtr,
                                 int* cscRowInd,
                                 cusparseAction_t copyValues,
                                 cusparseIndexBase_t idxBase,
                                 cusparseCsr2CscAlg_t alg,
                                 void* buffer,
                                 cudaStream_t stream);

template <>
inline cusparseStatus_t cusparsecsr2csc(cusparseHandle_t handle,
                                        int m,
                                        int n,
                                        int nnz,
                                        const float* csrVal,
                                        const int* csrRowPtr,
                                        const int* csrColInd,
                                        void* cscVal,
                                        int* cscColPtr,
                                        int* cscRowInd,
                                        cusparseAction_t copyValues,
                                        cusparseIndexBase_t idxBase,
                                        cusparseCsr2CscAlg_t alg,
                                        void* buffer,
                                        cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));

  return cusparseCsr2cscEx2(handle,
                            m,
                            n,
                            nnz,
                            csrVal,
                            csrRowPtr,
                            csrColInd,
                            cscVal,
                            cscColPtr,
                            cscRowInd,
                            CUDA_R_32F,
                            copyValues,
                            idxBase,
                            alg,
                            buffer);
}

template <>
inline cusparseStatus_t cusparsecsr2csc(cusparseHandle_t handle,
                                        int m,
                                        int n,
                                        int nnz,
                                        const double* csrVal,
                                        const int* csrRowPtr,
                                        const int* csrColInd,
                                        void* cscVal,
                                        int* cscColPtr,
                                        int* cscRowInd,
                                        cusparseAction_t copyValues,
                                        cusparseIndexBase_t idxBase,
                                        cusparseCsr2CscAlg_t alg,
                                        void* buffer,
                                        cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));

  return cusparseCsr2cscEx2(handle,
                            m,
                            n,
                            nnz,
                            csrVal,
                            csrRowPtr,
                            csrColInd,
                            cscVal,
                            cscColPtr,
                            cscRowInd,
                            CUDA_R_64F,
                            copyValues,
                            idxBase,
                            alg,
                            buffer);
}

/** @} */

/**
 * @defgroup csrgemm2 cusparse sparse gemm operations
 * @{
 */

template <typename T>
cusparseStatus_t cusparsecsr2dense(cusparseHandle_t handle,
                                   int m,
                                   int n,
                                   const cusparseMatDescr_t descrA,
                                   const T* csrValA,
                                   const int* csrRowPtrA,
                                   const int* csrColIndA,
                                   T* A,
                                   int lda,
                                   cudaStream_t stream);

template <>
inline cusparseStatus_t cusparsecsr2dense(cusparseHandle_t handle,
                                          int m,
                                          int n,
                                          const cusparseMatDescr_t descrA,
                                          const float* csrValA,
                                          const int* csrRowPtrA,
                                          const int* csrColIndA,
                                          float* A,
                                          int lda,
                                          cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
}
template <>
inline cusparseStatus_t cusparsecsr2dense(cusparseHandle_t handle,
                                          int m,
                                          int n,
                                          const cusparseMatDescr_t descrA,
                                          const double* csrValA,
                                          const int* csrRowPtrA,
                                          const int* csrColIndA,
                                          double* A,
                                          int lda,
                                          cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
}

/** @} */

}  // namespace sparse
}  // namespace raft
