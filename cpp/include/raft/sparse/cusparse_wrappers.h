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

#include <raft/error.hpp>

#include <cusparse_v2.h>
///@todo: enable this once logging is enabled
//#include <cuml/common/logger.hpp>

#define _CUSPARSE_ERR_TO_STR(err) \
  case err:                       \
    return #err;

namespace raft {

/**
 * @brief Exception thrown when a cuSparse error is encountered.
 */
struct cusparse_error : public raft::exception {
  explicit cusparse_error(char const* const message)
    : raft::exception(message) {}
  explicit cusparse_error(std::string const& message)
    : raft::exception(message) {}
};

namespace sparse {
namespace detail {

inline const char* cusparse_error_to_string(cusparseStatus_t err) {
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
    default:
      return "CUSPARSE_STATUS_UNKNOWN";
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
#define CUSPARSE_TRY(call)                                                   \
  do {                                                                       \
    cusparseStatus_t const status = (call);                                  \
    if (CUSPARSE_STATUS_SUCCESS != status) {                                 \
      std::string msg{};                                                     \
      SET_ERROR_MSG(msg, "cuSparse error encountered at: ",                  \
                    "call='%s', Reason=%d:%s", #call, status,                \
                    raft::sparse::detail::cusparse_error_to_string(status)); \
      throw raft::cusparse_error(msg);                                       \
    }                                                                        \
  } while (0)

/** FIXME: temporary alias for cuML compatibility */
#define CUSPARSE_CHECK(call) CUSPARSE_TRY(call)

//@todo: enable this once logging is enabled
#if 0
/** check for cusparse runtime API errors but do not assert */
#define CUSPARSE_CHECK_NO_THROW(call)                                          \
  do {                                                                         \
    cusparseStatus_t err = call;                                               \
    if (err != CUSPARSE_STATUS_SUCCESS) {                                      \
      CUML_LOG_ERROR("CUSPARSE call='%s' got errorcode=%d err=%s", #call, err, \
                     raft::sparse::detail::cusparse_error_to_string(err));     \
    }                                                                          \
  } while (0)
#endif

namespace raft {
namespace sparse {

/**
 * @defgroup gthr cusparse gather methods
 * @{
 */
template <typename T>
cusparseStatus_t cusparsegthr(cusparseHandle_t handle, int nnz, const T* vals,
                              T* vals_sorted, int* d_P, cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsegthr(cusparseHandle_t handle, int nnz,
                                     const double* vals, double* vals_sorted,
                                     int* d_P, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseDgthr(handle, nnz, vals, vals_sorted, d_P,
                       CUSPARSE_INDEX_BASE_ZERO);
}
template <>
inline cusparseStatus_t cusparsegthr(cusparseHandle_t handle, int nnz,
                                     const float* vals, float* vals_sorted,
                                     int* d_P, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSgthr(handle, nnz, vals, vals_sorted, d_P,
                       CUSPARSE_INDEX_BASE_ZERO);
}
/** @} */

/**
 * @defgroup coo2csr cusparse COO to CSR converter methods
 * @{
 */
template <typename T>
void cusparsecoo2csr(cusparseHandle_t handle, const T* cooRowInd, int nnz,
                     int m, T* csrRowPtr, cudaStream_t stream);
template <>
inline void cusparsecoo2csr(cusparseHandle_t handle, const int* cooRowInd,
                            int nnz, int m, int* csrRowPtr,
                            cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(cusparseXcoo2csr(handle, cooRowInd, nnz, m, csrRowPtr,
                                  CUSPARSE_INDEX_BASE_ZERO));
}
/** @} */

/**
 * @defgroup coosort cusparse coo sort methods
 * @{
 */
template <typename T>
size_t cusparsecoosort_bufferSizeExt(  // NOLINT
  cusparseHandle_t handle, int m, int n, int nnz, const T* cooRows,
  const T* cooCols, cudaStream_t stream);
template <>
inline size_t cusparsecoosort_bufferSizeExt(  // NOLINT
  cusparseHandle_t handle, int m, int n, int nnz, const int* cooRows,
  const int* cooCols, cudaStream_t stream) {
  size_t val;
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(
    cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, cooRows, cooCols, &val));
  return val;
}

template <typename T>
void cusparsecoosortByRow(  // NOLINT
  cusparseHandle_t handle, int m, int n, int nnz, T* cooRows, T* cooCols, T* P,
  void* pBuffer, cudaStream_t stream);
template <>
inline void cusparsecoosortByRow(  // NOLINT
  cusparseHandle_t handle, int m, int n, int nnz, int* cooRows, int* cooCols,
  int* P, void* pBuffer, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(
    cusparseXcoosortByRow(handle, m, n, nnz, cooRows, cooCols, P, pBuffer));
}
/** @} */

/**
 * @defgroup Gemmi cusparse gemmi operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsegemmi(  // NOLINT
  cusparseHandle_t handle, int m, int n, int k, int nnz, const T* alpha,
  const T* A, int lda, const T* cscValB, const int* cscColPtrB,
  const int* cscRowIndB, const T* beta, T* C, int ldc, cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsegemmi(cusparseHandle_t handle, int m, int n,
                                      int k, int nnz, const float* alpha,
                                      const float* A, int lda,
                                      const float* cscValB,
                                      const int* cscColPtrB,
                                      const int* cscRowIndB, const float* beta,
                                      float* C, int ldc, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB,
                        cscColPtrB, cscRowIndB, beta, C, ldc);
}
template <>
inline cusparseStatus_t cusparsegemmi(cusparseHandle_t handle, int m, int n,
                                      int k, int nnz, const double* alpha,
                                      const double* A, int lda,
                                      const double* cscValB,
                                      const int* cscColPtrB,
                                      const int* cscRowIndB, const double* beta,
                                      double* C, int ldc, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseDgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB,
                        cscColPtrB, cscRowIndB, beta, C, ldc);
}
/** @} */

#if __CUDACC_VER_MAJOR__ > 10
/**
 * @defgroup cusparse Create CSR operations
 * @{
 */
template <typename IndexT, typename ValueT>
cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                   int64_t rows, int64_t cols, int64_t nnz,
                                   IndexT* csrRowOffsets, IndexT* csrColInd,
                                   ValueT* csrValues);
template <>
inline cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                          int64_t rows, int64_t cols,
                                          int64_t nnz, int32_t* csrRowOffsets,
                                          int32_t* csrColInd,
                                          float* csrValues) {
  return cusparseCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets,
                           csrColInd, csrValues, CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_32F);
}
template <>
inline cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                          int64_t rows, int64_t cols,
                                          int64_t nnz, int32_t* csrRowOffsets,
                                          int32_t* csrColInd,
                                          double* csrValues) {
  return cusparseCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets,
                           csrColInd, csrValues, CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_64F);
}
template <>
inline cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                          int64_t rows, int64_t cols,
                                          int64_t nnz, int64_t* csrRowOffsets,
                                          int64_t* csrColInd,
                                          float* csrValues) {
  return cusparseCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets,
                           csrColInd, csrValues, CUSPARSE_INDEX_64I,
                           CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_32F);
}
template <>
inline cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                          int64_t rows, int64_t cols,
                                          int64_t nnz, int64_t* csrRowOffsets,
                                          int64_t* csrColInd,
                                          double* csrValues) {
  return cusparseCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets,
                           csrColInd, csrValues, CUSPARSE_INDEX_64I,
                           CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_64F);
}
/** @} */
/**
 * @defgroup cusparse CreateDnVec operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsecreatednvec(cusparseDnVecDescr_t* dnVecDescr,
                                     int64_t size, T* values);
template <>
inline cusparseStatus_t cusparsecreatednvec(cusparseDnVecDescr_t* dnVecDescr,
                                            int64_t size, float* values) {
  return cusparseCreateDnVec(dnVecDescr, size, values, CUDA_R_32F);
}
template <>
inline cusparseStatus_t cusparsecreatednvec(cusparseDnVecDescr_t* dnVecDescr,
                                            int64_t size, double* values) {
  return cusparseCreateDnVec(dnVecDescr, size, values, CUDA_R_64F);
}
/** @} */

/**
 * @defgroup Csrmv cusparse SpMV operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsespmv_buffersize(
  cusparseHandle_t handle, cusparseOperation_t opA, const T* alpha,
  const cusparseSpMatDescr_t matA, const cusparseDnVecDescr_t vecX,
  const T* beta, const cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg,
  size_t* bufferSize, cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsespmv_buffersize(
  cusparseHandle_t handle, cusparseOperation_t opA, const float* alpha,
  const cusparseSpMatDescr_t matA, const cusparseDnVecDescr_t vecX,
  const float* beta, const cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg,
  size_t* bufferSize, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMV_bufferSize(handle, opA, alpha, matA, vecX, beta, vecY,
                                 CUDA_R_32F, alg, bufferSize);
}
template <>
inline cusparseStatus_t cusparsespmv_buffersize(
  cusparseHandle_t handle, cusparseOperation_t opA, const double* alpha,
  const cusparseSpMatDescr_t matA, const cusparseDnVecDescr_t vecX,
  const double* beta, const cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg,
  size_t* bufferSize, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMV_bufferSize(handle, opA, alpha, matA, vecX, beta, vecY,
                                 CUDA_R_64F, alg, bufferSize);
}

template <typename T>
cusparseStatus_t cusparsespmv(cusparseHandle_t handle, cusparseOperation_t opA,
                              const T* alpha, const cusparseSpMatDescr_t matA,
                              const cusparseDnVecDescr_t vecX, const T* beta,
                              const cusparseDnVecDescr_t vecY,
                              cusparseSpMVAlg_t alg, T* externalBuffer,
                              cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsespmv(
  cusparseHandle_t handle, cusparseOperation_t opA, const float* alpha,
  const cusparseSpMatDescr_t matA, const cusparseDnVecDescr_t vecX,
  const float* beta, const cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg,
  float* externalBuffer, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, CUDA_R_32F,
                      alg, externalBuffer);
}
template <>
inline cusparseStatus_t cusparsespmv(
  cusparseHandle_t handle, cusparseOperation_t opA, const double* alpha,
  const cusparseSpMatDescr_t matA, const cusparseDnVecDescr_t vecX,
  const double* beta, const cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg,
  double* externalBuffer, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, CUDA_R_64F,
                      alg, externalBuffer);
}
/** @} */
#else
/**
 * @defgroup Csrmv cusparse csrmv operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsecsrmv(  // NOLINT
  cusparseHandle_t handle, cusparseOperation_t trans, int m, int n, int nnz,
  const T* alpha, const cusparseMatDescr_t descr, const T* csrVal,
  const int* csrRowPtr, const int* csrColInd, const T* x, const T* beta, T* y,
  cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsecsrmv(
  cusparseHandle_t handle, cusparseOperation_t trans, int m, int n, int nnz,
  const float* alpha, const cusparseMatDescr_t descr, const float* csrVal,
  const int* csrRowPtr, const int* csrColInd, const float* x, const float* beta,
  float* y, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseScsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal,
                        csrRowPtr, csrColInd, x, beta, y);
}
template <>
inline cusparseStatus_t cusparsecsrmv(
  cusparseHandle_t handle, cusparseOperation_t trans, int m, int n, int nnz,
  const double* alpha, const cusparseMatDescr_t descr, const double* csrVal,
  const int* csrRowPtr, const int* csrColInd, const double* x,
  const double* beta, double* y, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseDcsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal,
                        csrRowPtr, csrColInd, x, beta, y);
}
/** @} */
#endif

#if __CUDACC_VER_MAJOR__ > 10
/**
 * @defgroup Csrmm cusparse csrmm operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsespmm_bufferSize(
  cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
  const T* alpha, const cusparseSpMatDescr_t matA,
  const cusparseDnMatDescr_t matB, const T* beta, cusparseDnMatDescr_t matC,
  cusparseSpMMAlg_t alg, size_t* bufferSize, cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsespmm_bufferSize(
  cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
  const float* alpha, const cusparseSpMatDescr_t matA,
  const cusparseDnMatDescr_t matB, const float* beta, cusparseDnMatDescr_t matC,
  cusparseSpMMAlg_t alg, size_t* bufferSize, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMM_bufferSize(handle, opA, opB, alpha, matA, matB, beta,
                                 matC, CUDA_R_32F, alg, bufferSize);
}
template <>
inline cusparseStatus_t cusparsespmm_bufferSize(
  cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
  const double* alpha, const cusparseSpMatDescr_t matA,
  const cusparseDnMatDescr_t matB, const double* beta,
  cusparseDnMatDescr_t matC, cusparseSpMMAlg_t alg, size_t* bufferSize,
  cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMM_bufferSize(handle, opA, opB, alpha, matA, matB, beta,
                                 matC, CUDA_R_64F, alg, bufferSize);
}
template <typename T>
inline cusparseStatus_t cusparsespmm(
  cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
  const T* alpha, const cusparseSpMatDescr_t matA,
  const cusparseDnMatDescr_t matB, const T* beta, cusparseDnMatDescr_t matC,
  cusparseSpMMAlg_t alg, T* externalBuffer, cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsespmm(
  cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
  const float* alpha, const cusparseSpMatDescr_t matA,
  const cusparseDnMatDescr_t matB, const float* beta, cusparseDnMatDescr_t matC,
  cusparseSpMMAlg_t alg, float* externalBuffer, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMM(handle, opA, opB, alpha, matA, matB, beta, matC,
                      CUDA_R_32F, alg, externalBuffer);
}
template <>
inline cusparseStatus_t cusparsespmm(
  cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
  const double* alpha, const cusparseSpMatDescr_t matA,
  const cusparseDnMatDescr_t matB, const double* beta,
  cusparseDnMatDescr_t matC, cusparseSpMMAlg_t alg, double* externalBuffer,
  cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMM(handle, opA, opB, alpha, matA, matB, beta, matC,
                      CUDA_R_64F, alg, externalBuffer);
}
/** @} */
#else
/**
 * @defgroup Csrmm cusparse csrmm operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsecsrmm(  // NOLINT
  cusparseHandle_t handle, cusparseOperation_t trans, int m, int n, int k,
  int nnz, const T* alpha, const cusparseMatDescr_t descr, const T* csrVal,
  const int* csrRowPtr, const int* csrColInd, const T* x, const int ldx,
  const T* beta, T* y, const int ldy, cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsecsrmm(
  cusparseHandle_t handle, cusparseOperation_t trans, int m, int n, int k,
  int nnz, const float* alpha, const cusparseMatDescr_t descr,
  const float* csrVal, const int* csrRowPtr, const int* csrColInd,
  const float* x, const int ldx, const float* beta, float* y, const int ldy,
  cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseScsrmm(handle, trans, m, n, k, nnz, alpha, descr, csrVal,
                        csrRowPtr, csrColInd, x, ldx, beta, y, ldy);
}
template <>
inline cusparseStatus_t cusparsecsrmm(
  cusparseHandle_t handle, cusparseOperation_t trans, int m, int n, int k,
  int nnz, const double* alpha, const cusparseMatDescr_t descr,
  const double* csrVal, const int* csrRowPtr, const int* csrColInd,
  const double* x, const int ldx, const double* beta, double* y, const int ldy,
  cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseDcsrmm(handle, trans, m, n, k, nnz, alpha, descr, csrVal,
                        csrRowPtr, csrColInd, x, ldx, beta, y, ldy);
}
/** @} */
#endif

/**
 * @defgroup csr2coo cusparse CSR to COO converter methods
 * @{
 */
template <typename T>
void cusparsecsr2coo(  // NOLINT
  cusparseHandle_t handle, const int n, const int nnz, const T* csrRowPtr,
  T* cooRowInd, cudaStream_t stream);
template <>
inline void cusparsecsr2coo(cusparseHandle_t handle, const int n, const int nnz,
                            const int* csrRowPtr, int* cooRowInd,
                            cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(cusparseXcsr2coo(handle, csrRowPtr, nnz, n, cooRowInd,
                                  CUSPARSE_INDEX_BASE_ZERO));
}
/** @} */

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
                                               cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSetPointerMode(handle, mode);
}
/** @} */

/**
 * @defgroup CsrmvEx cusparse csrmvex operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsecsrmvex_bufferSize(
  cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA,
  int m, int n, int nnz, const T* alpha, const cusparseMatDescr_t descrA,
  const T* csrValA, const int* csrRowPtrA, const int* csrColIndA, const T* x,
  const T* beta, T* y, size_t* bufferSizeInBytes, cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsecsrmvex_bufferSize(
  cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA,
  int m, int n, int nnz, const float* alpha, const cusparseMatDescr_t descrA,
  const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
  const float* x, const float* beta, float* y, size_t* bufferSizeInBytes,
  cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseCsrmvEx_bufferSize(
    handle, alg, transA, m, n, nnz, alpha, CUDA_R_32F, descrA, csrValA,
    CUDA_R_32F, csrRowPtrA, csrColIndA, x, CUDA_R_32F, beta, CUDA_R_32F, y,
    CUDA_R_32F, CUDA_R_32F, bufferSizeInBytes);
}
template <>
inline cusparseStatus_t cusparsecsrmvex_bufferSize(
  cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA,
  int m, int n, int nnz, const double* alpha, const cusparseMatDescr_t descrA,
  const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
  const double* x, const double* beta, double* y, size_t* bufferSizeInBytes,
  cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseCsrmvEx_bufferSize(
    handle, alg, transA, m, n, nnz, alpha, CUDA_R_64F, descrA, csrValA,
    CUDA_R_64F, csrRowPtrA, csrColIndA, x, CUDA_R_64F, beta, CUDA_R_64F, y,
    CUDA_R_64F, CUDA_R_64F, bufferSizeInBytes);
}

template <typename T>
cusparseStatus_t cusparsecsrmvex(
  cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA,
  int m, int n, int nnz, const T* alpha, const cusparseMatDescr_t descrA,
  const T* csrValA, const int* csrRowPtrA, const int* csrColIndA, const T* x,
  const T* beta, T* y, T* buffer, cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsecsrmvex(
  cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA,
  int m, int n, int nnz, const float* alpha, const cusparseMatDescr_t descrA,
  const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
  const float* x, const float* beta, float* y, float* buffer,
  cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseCsrmvEx(handle, alg, transA, m, n, nnz, alpha, CUDA_R_32F,
                         descrA, csrValA, CUDA_R_32F, csrRowPtrA, csrColIndA, x,
                         CUDA_R_32F, beta, CUDA_R_32F, y, CUDA_R_32F,
                         CUDA_R_32F, buffer);
}
template <>
inline cusparseStatus_t cusparsecsrmvex(
  cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA,
  int m, int n, int nnz, const double* alpha, const cusparseMatDescr_t descrA,
  const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
  const double* x, const double* beta, double* y, double* buffer,
  cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseCsrmvEx(handle, alg, transA, m, n, nnz, alpha, CUDA_R_64F,
                         descrA, csrValA, CUDA_R_64F, csrRowPtrA, csrColIndA, x,
                         CUDA_R_64F, beta, CUDA_R_64F, y, CUDA_R_64F,
                         CUDA_R_64F, buffer);
}
/** @} */

}  // namespace sparse
}  // namespace raft
