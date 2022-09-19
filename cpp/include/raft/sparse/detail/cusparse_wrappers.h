/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cusparse.h>
#include <raft/core/error.hpp>
#include <raft/sparse/detail/cusparse_macros.h>

namespace raft {
namespace sparse {
namespace detail {

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

/**
 * @defgroup Gemmi cusparse gemmi operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsegemmi(  // NOLINT
  cusparseHandle_t handle,
  int m,
  int n,
  int k,
  int nnz,
  const T* alpha,
  const T* A,
  int lda,
  const T* cscValB,
  const int* cscColPtrB,
  const int* cscRowIndB,
  const T* beta,
  T* C,
  int ldc,
  cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsegemmi(cusparseHandle_t handle,
                                      int m,
                                      int n,
                                      int k,
                                      int nnz,
                                      const float* alpha,
                                      const float* A,
                                      int lda,
                                      const float* cscValB,
                                      const int* cscColPtrB,
                                      const int* cscRowIndB,
                                      const float* beta,
                                      float* C,
                                      int ldc,
                                      cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return cusparseSgemmi(
    handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
#pragma GCC diagnostic pop
}
template <>
inline cusparseStatus_t cusparsegemmi(cusparseHandle_t handle,
                                      int m,
                                      int n,
                                      int k,
                                      int nnz,
                                      const double* alpha,
                                      const double* A,
                                      int lda,
                                      const double* cscValB,
                                      const int* cscColPtrB,
                                      const int* cscRowIndB,
                                      const double* beta,
                                      double* C,
                                      int ldc,
                                      cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return cusparseDgemmi(
    handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
#pragma GCC diagnostic pop
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

#if not defined CUDA_ENFORCE_LOWER and CUDA_VER_10_1_UP
/**
 * @defgroup Csrmm cusparse csrmm operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsespmm_bufferSize(cusparseHandle_t handle,
                                         cusparseOperation_t opA,
                                         cusparseOperation_t opB,
                                         const T* alpha,
                                         const cusparseSpMatDescr_t matA,
                                         const cusparseDnMatDescr_t matB,
                                         const T* beta,
                                         cusparseDnMatDescr_t matC,
                                         cusparseSpMMAlg_t alg,
                                         size_t* bufferSize,
                                         cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsespmm_bufferSize(cusparseHandle_t handle,
                                                cusparseOperation_t opA,
                                                cusparseOperation_t opB,
                                                const float* alpha,
                                                const cusparseSpMatDescr_t matA,
                                                const cusparseDnMatDescr_t matB,
                                                const float* beta,
                                                cusparseDnMatDescr_t matC,
                                                cusparseSpMMAlg_t alg,
                                                size_t* bufferSize,
                                                cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMM_bufferSize(
    handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_32F, alg, bufferSize);
}
template <>
inline cusparseStatus_t cusparsespmm_bufferSize(cusparseHandle_t handle,
                                                cusparseOperation_t opA,
                                                cusparseOperation_t opB,
                                                const double* alpha,
                                                const cusparseSpMatDescr_t matA,
                                                const cusparseDnMatDescr_t matB,
                                                const double* beta,
                                                cusparseDnMatDescr_t matC,
                                                cusparseSpMMAlg_t alg,
                                                size_t* bufferSize,
                                                cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMM_bufferSize(
    handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_64F, alg, bufferSize);
}
template <typename T>
inline cusparseStatus_t cusparsespmm(cusparseHandle_t handle,
                                     cusparseOperation_t opA,
                                     cusparseOperation_t opB,
                                     const T* alpha,
                                     const cusparseSpMatDescr_t matA,
                                     const cusparseDnMatDescr_t matB,
                                     const T* beta,
                                     cusparseDnMatDescr_t matC,
                                     cusparseSpMMAlg_t alg,
                                     T* externalBuffer,
                                     cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsespmm(cusparseHandle_t handle,
                                     cusparseOperation_t opA,
                                     cusparseOperation_t opB,
                                     const float* alpha,
                                     const cusparseSpMatDescr_t matA,
                                     const cusparseDnMatDescr_t matB,
                                     const float* beta,
                                     cusparseDnMatDescr_t matC,
                                     cusparseSpMMAlg_t alg,
                                     float* externalBuffer,
                                     cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMM(
    handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_32F, alg, externalBuffer);
}
template <>
inline cusparseStatus_t cusparsespmm(cusparseHandle_t handle,
                                     cusparseOperation_t opA,
                                     cusparseOperation_t opB,
                                     const double* alpha,
                                     const cusparseSpMatDescr_t matA,
                                     const cusparseDnMatDescr_t matB,
                                     const double* beta,
                                     cusparseDnMatDescr_t matC,
                                     cusparseSpMMAlg_t alg,
                                     double* externalBuffer,
                                     cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSpMM(
    handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_64F, alg, externalBuffer);
}
/** @} */
#else
/**
 * @defgroup Csrmm cusparse csrmm operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsecsrmm(  // NOLINT
  cusparseHandle_t handle,
  cusparseOperation_t trans,
  int m,
  int n,
  int k,
  int nnz,
  const T* alpha,
  const cusparseMatDescr_t descr,
  const T* csrVal,
  const int* csrRowPtr,
  const int* csrColInd,
  const T* x,
  const int ldx,
  const T* beta,
  T* y,
  const int ldy,
  cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsecsrmm(cusparseHandle_t handle,
                                      cusparseOperation_t trans,
                                      int m,
                                      int n,
                                      int k,
                                      int nnz,
                                      const float* alpha,
                                      const cusparseMatDescr_t descr,
                                      const float* csrVal,
                                      const int* csrRowPtr,
                                      const int* csrColInd,
                                      const float* x,
                                      const int ldx,
                                      const float* beta,
                                      float* y,
                                      const int ldy,
                                      cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseScsrmm(
    handle, trans, m, n, k, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, ldx, beta, y, ldy);
}
template <>
inline cusparseStatus_t cusparsecsrmm(cusparseHandle_t handle,
                                      cusparseOperation_t trans,
                                      int m,
                                      int n,
                                      int k,
                                      int nnz,
                                      const double* alpha,
                                      const cusparseMatDescr_t descr,
                                      const double* csrVal,
                                      const int* csrRowPtr,
                                      const int* csrColInd,
                                      const double* x,
                                      const int ldx,
                                      const double* beta,
                                      double* y,
                                      const int ldy,
                                      cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseDcsrmm(
    handle, trans, m, n, k, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, ldx, beta, y, ldy);
}
/** @} */
#endif

/**
 * @defgroup csr2coo cusparse CSR to COO converter methods
 * @{
 */
template <typename T>
void cusparsecsr2coo(  // NOLINT
  cusparseHandle_t handle,
  const int n,
  const int nnz,
  const T* csrRowPtr,
  T* cooRowInd,
  cudaStream_t stream);
template <>
inline void cusparsecsr2coo(cusparseHandle_t handle,
                            const int n,
                            const int nnz,
                            const int* csrRowPtr,
                            int* cooRowInd,
                            cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(cusparseXcsr2coo(handle, csrRowPtr, nnz, n, cooRowInd, CUSPARSE_INDEX_BASE_ZERO));
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
                                               cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSetPointerMode(handle, mode);
}
/** @} */

/**
 * @defgroup CsrmvEx cusparse csrmvex operations
 * @{
 */
template <typename T>
cusparseStatus_t cusparsecsrmvex_bufferSize(cusparseHandle_t handle,
                                            cusparseAlgMode_t alg,
                                            cusparseOperation_t transA,
                                            int m,
                                            int n,
                                            int nnz,
                                            const T* alpha,
                                            const cusparseMatDescr_t descrA,
                                            const T* csrValA,
                                            const int* csrRowPtrA,
                                            const int* csrColIndA,
                                            const T* x,
                                            const T* beta,
                                            T* y,
                                            size_t* bufferSizeInBytes,
                                            cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsecsrmvex_bufferSize(cusparseHandle_t handle,
                                                   cusparseAlgMode_t alg,
                                                   cusparseOperation_t transA,
                                                   int m,
                                                   int n,
                                                   int nnz,
                                                   const float* alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const float* csrValA,
                                                   const int* csrRowPtrA,
                                                   const int* csrColIndA,
                                                   const float* x,
                                                   const float* beta,
                                                   float* y,
                                                   size_t* bufferSizeInBytes,
                                                   cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));

#if CUDART_VERSION >= 11020
  cusparseSpMatDescr_t matA;
  cusparsecreatecsr(&matA,
                    m,
                    n,
                    nnz,
                    const_cast<int*>(csrRowPtrA),
                    const_cast<int*>(csrColIndA),
                    const_cast<float*>(csrValA));

  cusparseDnVecDescr_t vecX;
  cusparsecreatednvec(&vecX, static_cast<int64_t>(n), const_cast<float*>(x));

  cusparseDnVecDescr_t vecY;
  cusparsecreatednvec(&vecY, static_cast<int64_t>(n), y);

  cusparseStatus_t result = cusparseSpMV_bufferSize(handle,
                                                    transA,
                                                    alpha,
                                                    matA,
                                                    vecX,
                                                    beta,
                                                    vecY,
                                                    CUDA_R_32F,
                                                    CUSPARSE_SPMV_ALG_DEFAULT,
                                                    bufferSizeInBytes);

  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(matA));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnVec(vecX));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnVec(vecY));
  return result;

#else

  return cusparseCsrmvEx_bufferSize(handle,
                                    alg,
                                    transA,
                                    m,
                                    n,
                                    nnz,
                                    alpha,
                                    CUDA_R_32F,
                                    descrA,
                                    csrValA,
                                    CUDA_R_32F,
                                    csrRowPtrA,
                                    csrColIndA,
                                    x,
                                    CUDA_R_32F,
                                    beta,
                                    CUDA_R_32F,
                                    y,
                                    CUDA_R_32F,
                                    CUDA_R_32F,
                                    bufferSizeInBytes);
#endif
}
template <>
inline cusparseStatus_t cusparsecsrmvex_bufferSize(cusparseHandle_t handle,
                                                   cusparseAlgMode_t alg,
                                                   cusparseOperation_t transA,
                                                   int m,
                                                   int n,
                                                   int nnz,
                                                   const double* alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const double* csrValA,
                                                   const int* csrRowPtrA,
                                                   const int* csrColIndA,
                                                   const double* x,
                                                   const double* beta,
                                                   double* y,
                                                   size_t* bufferSizeInBytes,
                                                   cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));

#if CUDART_VERSION >= 11020
  cusparseSpMatDescr_t matA;
  cusparsecreatecsr(&matA,
                    m,
                    n,
                    nnz,
                    const_cast<int*>(csrRowPtrA),
                    const_cast<int*>(csrColIndA),
                    const_cast<double*>(csrValA));

  cusparseDnVecDescr_t vecX;
  cusparsecreatednvec(&vecX, static_cast<int64_t>(n), const_cast<double*>(x));

  cusparseDnVecDescr_t vecY;
  cusparsecreatednvec(&vecY, static_cast<int64_t>(n), y);

  cusparseStatus_t result = cusparseSpMV_bufferSize(handle,
                                                    transA,
                                                    alpha,
                                                    matA,
                                                    vecX,
                                                    beta,
                                                    vecY,
                                                    CUDA_R_64F,
                                                    CUSPARSE_SPMV_ALG_DEFAULT,
                                                    bufferSizeInBytes);

  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(matA));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnVec(vecX));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnVec(vecY));
  return result;
#else
  return cusparseCsrmvEx_bufferSize(handle,
                                    alg,
                                    transA,
                                    m,
                                    n,
                                    nnz,
                                    alpha,
                                    CUDA_R_64F,
                                    descrA,
                                    csrValA,
                                    CUDA_R_64F,
                                    csrRowPtrA,
                                    csrColIndA,
                                    x,
                                    CUDA_R_64F,
                                    beta,
                                    CUDA_R_64F,
                                    y,
                                    CUDA_R_64F,
                                    CUDA_R_64F,
                                    bufferSizeInBytes);
#endif
}

template <typename T>
cusparseStatus_t cusparsecsrmvex(cusparseHandle_t handle,
                                 cusparseAlgMode_t alg,
                                 cusparseOperation_t transA,
                                 int m,
                                 int n,
                                 int nnz,
                                 const T* alpha,
                                 const cusparseMatDescr_t descrA,
                                 const T* csrValA,
                                 const int* csrRowPtrA,
                                 const int* csrColIndA,
                                 const T* x,
                                 const T* beta,
                                 T* y,
                                 T* buffer,
                                 cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsecsrmvex(cusparseHandle_t handle,
                                        cusparseAlgMode_t alg,
                                        cusparseOperation_t transA,
                                        int m,
                                        int n,
                                        int nnz,
                                        const float* alpha,
                                        const cusparseMatDescr_t descrA,
                                        const float* csrValA,
                                        const int* csrRowPtrA,
                                        const int* csrColIndA,
                                        const float* x,
                                        const float* beta,
                                        float* y,
                                        float* buffer,
                                        cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));

#if CUDART_VERSION >= 11020
  cusparseSpMatDescr_t matA;
  cusparsecreatecsr(&matA,
                    m,
                    n,
                    nnz,
                    const_cast<int*>(csrRowPtrA),
                    const_cast<int*>(csrColIndA),
                    const_cast<float*>(csrValA));

  cusparseDnVecDescr_t vecX;
  cusparsecreatednvec(&vecX, static_cast<int64_t>(n), const_cast<float*>(x));

  cusparseDnVecDescr_t vecY;
  cusparsecreatednvec(&vecY, static_cast<int64_t>(n), y);

  cusparseStatus_t result = cusparseSpMV(
    handle, transA, alpha, matA, vecX, beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);

  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(matA));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnVec(vecX));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnVec(vecY));
  return result;
#else
  return cusparseCsrmvEx(handle,
                         alg,
                         transA,
                         m,
                         n,
                         nnz,
                         alpha,
                         CUDA_R_32F,
                         descrA,
                         csrValA,
                         CUDA_R_32F,
                         csrRowPtrA,
                         csrColIndA,
                         x,
                         CUDA_R_32F,
                         beta,
                         CUDA_R_32F,
                         y,
                         CUDA_R_32F,
                         CUDA_R_32F,
                         buffer);
#endif
}
template <>
inline cusparseStatus_t cusparsecsrmvex(cusparseHandle_t handle,
                                        cusparseAlgMode_t alg,
                                        cusparseOperation_t transA,
                                        int m,
                                        int n,
                                        int nnz,
                                        const double* alpha,
                                        const cusparseMatDescr_t descrA,
                                        const double* csrValA,
                                        const int* csrRowPtrA,
                                        const int* csrColIndA,
                                        const double* x,
                                        const double* beta,
                                        double* y,
                                        double* buffer,
                                        cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));

#if CUDART_VERSION >= 11020
  cusparseSpMatDescr_t matA;
  cusparsecreatecsr(&matA,
                    m,
                    n,
                    nnz,
                    const_cast<int*>(csrRowPtrA),
                    const_cast<int*>(csrColIndA),
                    const_cast<double*>(csrValA));

  cusparseDnVecDescr_t vecX;
  cusparsecreatednvec(&vecX, static_cast<int64_t>(n), const_cast<double*>(x));

  cusparseDnVecDescr_t vecY;
  cusparsecreatednvec(&vecY, static_cast<int64_t>(n), y);

  cusparseStatus_t result = cusparseSpMV(
    handle, transA, alpha, matA, vecX, beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);

  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(matA));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnVec(vecX));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnVec(vecY));
  return result;

#else

  return cusparseCsrmvEx(handle,
                         alg,
                         transA,
                         m,
                         n,
                         nnz,
                         alpha,
                         CUDA_R_64F,
                         descrA,
                         csrValA,
                         CUDA_R_64F,
                         csrRowPtrA,
                         csrColIndA,
                         x,
                         CUDA_R_64F,
                         beta,
                         CUDA_R_64F,
                         y,
                         CUDA_R_64F,
                         CUDA_R_64F,
                         buffer);
#endif
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
cusparseStatus_t cusparsecsrgemm2_buffersizeext(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                int k,
                                                const T* alpha,
                                                const T* beta,
                                                const cusparseMatDescr_t matA,
                                                int nnzA,
                                                const int* rowindA,
                                                const int* indicesA,
                                                const cusparseMatDescr_t matB,
                                                int nnzB,
                                                const int* rowindB,
                                                const int* indicesB,
                                                const cusparseMatDescr_t matD,
                                                int nnzD,
                                                const int* rowindD,
                                                const int* indicesD,
                                                csrgemm2Info_t info,
                                                size_t* pBufferSizeInBytes,
                                                cudaStream_t stream);

template <>
inline cusparseStatus_t cusparsecsrgemm2_buffersizeext(cusparseHandle_t handle,
                                                       int m,
                                                       int n,
                                                       int k,
                                                       const float* alpha,
                                                       const float* beta,
                                                       const cusparseMatDescr_t matA,
                                                       int nnzA,
                                                       const int* rowindA,
                                                       const int* indicesA,
                                                       const cusparseMatDescr_t matB,
                                                       int nnzB,
                                                       const int* rowindB,
                                                       const int* indicesB,
                                                       const cusparseMatDescr_t matD,
                                                       int nnzD,
                                                       const int* rowindD,
                                                       const int* indicesD,
                                                       csrgemm2Info_t info,
                                                       size_t* pBufferSizeInBytes,
                                                       cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return cusparseScsrgemm2_bufferSizeExt(handle,
                                         m,
                                         n,
                                         k,
                                         alpha,
                                         matA,
                                         nnzA,
                                         rowindA,
                                         indicesA,
                                         matB,
                                         nnzB,
                                         rowindB,
                                         indicesB,
                                         beta,
                                         matD,
                                         nnzD,
                                         rowindD,
                                         indicesD,
                                         info,
                                         pBufferSizeInBytes);
#pragma GCC diagnostic pop
}

template <>
inline cusparseStatus_t cusparsecsrgemm2_buffersizeext(cusparseHandle_t handle,
                                                       int m,
                                                       int n,
                                                       int k,
                                                       const double* alpha,
                                                       const double* beta,
                                                       const cusparseMatDescr_t matA,
                                                       int nnzA,
                                                       const int* rowindA,
                                                       const int* indicesA,
                                                       const cusparseMatDescr_t matB,
                                                       int nnzB,
                                                       const int* rowindB,
                                                       const int* indicesB,
                                                       const cusparseMatDescr_t matD,
                                                       int nnzD,
                                                       const int* rowindD,
                                                       const int* indicesD,
                                                       csrgemm2Info_t info,
                                                       size_t* pBufferSizeInBytes,
                                                       cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return cusparseDcsrgemm2_bufferSizeExt(handle,
                                         m,
                                         n,
                                         k,
                                         alpha,
                                         matA,
                                         nnzA,
                                         rowindA,
                                         indicesA,
                                         matB,
                                         nnzB,
                                         rowindB,
                                         indicesB,
                                         beta,
                                         matD,
                                         nnzD,
                                         rowindD,
                                         indicesD,
                                         info,
                                         pBufferSizeInBytes);
#pragma GCC diagnostic pop
}

inline cusparseStatus_t cusparsecsrgemm2nnz(cusparseHandle_t handle,
                                            int m,
                                            int n,
                                            int k,
                                            const cusparseMatDescr_t matA,
                                            int nnzA,
                                            const int* rowindA,
                                            const int* indicesA,
                                            const cusparseMatDescr_t matB,
                                            int nnzB,
                                            const int* rowindB,
                                            const int* indicesB,
                                            const cusparseMatDescr_t matD,
                                            int nnzD,
                                            const int* rowindD,
                                            const int* indicesD,
                                            const cusparseMatDescr_t matC,
                                            int* rowindC,
                                            int* nnzC,
                                            const csrgemm2Info_t info,
                                            void* pBuffer,
                                            cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return cusparseXcsrgemm2Nnz(handle,
                              m,
                              n,
                              k,
                              matA,
                              nnzA,
                              rowindA,
                              indicesA,
                              matB,
                              nnzB,
                              rowindB,
                              indicesB,
                              matD,
                              nnzD,
                              rowindD,
                              indicesD,
                              matC,
                              rowindC,
                              nnzC,
                              info,
                              pBuffer);
#pragma GCC diagnostic pop
}

template <typename T>
cusparseStatus_t cusparsecsrgemm2(cusparseHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  const T* alpha,
                                  const cusparseMatDescr_t descrA,
                                  int nnzA,
                                  const T* csrValA,
                                  const int* csrRowPtrA,
                                  const int* csrColIndA,
                                  const cusparseMatDescr_t descrB,
                                  int nnzB,
                                  const T* csrValB,
                                  const int* csrRowPtrB,
                                  const int* csrColIndB,
                                  const T* beta,
                                  const cusparseMatDescr_t descrD,
                                  int nnzD,
                                  const T* csrValD,
                                  const int* csrRowPtrD,
                                  const int* csrColIndD,
                                  const cusparseMatDescr_t descrC,
                                  T* csrValC,
                                  const int* csrRowPtrC,
                                  int* csrColIndC,
                                  const csrgemm2Info_t info,
                                  void* pBuffer,
                                  cudaStream_t stream);

template <>
inline cusparseStatus_t cusparsecsrgemm2(cusparseHandle_t handle,
                                         int m,
                                         int n,
                                         int k,
                                         const float* alpha,
                                         const cusparseMatDescr_t descrA,
                                         int nnzA,
                                         const float* csrValA,
                                         const int* csrRowPtrA,
                                         const int* csrColIndA,
                                         const cusparseMatDescr_t descrB,
                                         int nnzB,
                                         const float* csrValB,
                                         const int* csrRowPtrB,
                                         const int* csrColIndB,
                                         const float* beta,
                                         const cusparseMatDescr_t descrD,
                                         int nnzD,
                                         const float* csrValD,
                                         const int* csrRowPtrD,
                                         const int* csrColIndD,
                                         const cusparseMatDescr_t descrC,
                                         float* csrValC,
                                         const int* csrRowPtrC,
                                         int* csrColIndC,
                                         const csrgemm2Info_t info,
                                         void* pBuffer,
                                         cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return cusparseScsrgemm2(handle,
                           m,
                           n,
                           k,
                           alpha,
                           descrA,
                           nnzA,
                           csrValA,
                           csrRowPtrA,
                           csrColIndA,
                           descrB,
                           nnzB,
                           csrValB,
                           csrRowPtrB,
                           csrColIndB,
                           beta,
                           descrD,
                           nnzD,
                           csrValD,
                           csrRowPtrD,
                           csrColIndD,
                           descrC,
                           csrValC,
                           csrRowPtrC,
                           csrColIndC,
                           info,
                           pBuffer);
#pragma GCC diagnostic pop
}

template <>
inline cusparseStatus_t cusparsecsrgemm2(cusparseHandle_t handle,
                                         int m,
                                         int n,
                                         int k,
                                         const double* alpha,
                                         const cusparseMatDescr_t descrA,
                                         int nnzA,
                                         const double* csrValA,
                                         const int* csrRowPtrA,
                                         const int* csrColIndA,
                                         const cusparseMatDescr_t descrB,
                                         int nnzB,
                                         const double* csrValB,
                                         const int* csrRowPtrB,
                                         const int* csrColIndB,
                                         const double* beta,
                                         const cusparseMatDescr_t descrD,
                                         int nnzD,
                                         const double* csrValD,
                                         const int* csrRowPtrD,
                                         const int* csrColIndD,
                                         const cusparseMatDescr_t descrC,
                                         double* csrValC,
                                         const int* csrRowPtrC,
                                         int* csrColIndC,
                                         const csrgemm2Info_t info,
                                         void* pBuffer,
                                         cudaStream_t stream)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return cusparseDcsrgemm2(handle,
                           m,
                           n,
                           k,
                           alpha,
                           descrA,
                           nnzA,
                           csrValA,
                           csrRowPtrA,
                           csrColIndA,
                           descrB,
                           nnzB,
                           csrValB,
                           csrRowPtrB,
                           csrColIndB,
                           beta,
                           descrD,
                           nnzD,
                           csrValD,
                           csrRowPtrD,
                           csrColIndD,
                           descrC,
                           csrValC,
                           csrRowPtrC,
                           csrColIndC,
                           info,
                           pBuffer);
#pragma GCC diagnostic pop
}

/** @} */

/**
 * @defgroup csrgemm2 cusparse sparse gemm operations
 * @{
 */

template <typename T>
cusparseStatus_t cusparsecsr2dense_buffersize(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              int nnz,
                                              const cusparseMatDescr_t descrA,
                                              const T* csrValA,
                                              const int* csrRowPtrA,
                                              const int* csrColIndA,
                                              T* A,
                                              int lda,
                                              size_t* buffer_size,
                                              cudaStream_t stream,
                                              bool row_major = false);

template <>
inline cusparseStatus_t cusparsecsr2dense_buffersize(cusparseHandle_t handle,
                                                     int m,
                                                     int n,
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA,
                                                     const float* csrValA,
                                                     const int* csrRowPtrA,
                                                     const int* csrColIndA,
                                                     float* A,
                                                     int lda,
                                                     size_t* buffer_size,
                                                     cudaStream_t stream,
                                                     bool row_major)
{
#if CUDART_VERSION >= 11020
  cusparseOrder_t order = row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;

  cusparseSpMatDescr_t matA;
  cusparsecreatecsr(&matA,
                    m,
                    n,
                    nnz,
                    const_cast<int*>(csrRowPtrA),
                    const_cast<int*>(csrColIndA),
                    const_cast<float*>(csrValA));

  cusparseDnMatDescr_t matB;
  cusparsecreatednmat(&matB,
                      static_cast<int64_t>(m),
                      static_cast<int64_t>(n),
                      static_cast<int64_t>(lda),
                      const_cast<float*>(A),
                      order);

  cusparseStatus_t result = cusparseSparseToDense_bufferSize(
    handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer_size);

  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(matA));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(matB));

#else

  cusparseStatus_t result = CUSPARSE_STATUS_SUCCESS;
  buffer_size[0]          = 0;

#endif
  return result;
}

template <>
inline cusparseStatus_t cusparsecsr2dense_buffersize(cusparseHandle_t handle,
                                                     int m,
                                                     int n,
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA,
                                                     const double* csrValA,
                                                     const int* csrRowPtrA,
                                                     const int* csrColIndA,
                                                     double* A,
                                                     int lda,
                                                     size_t* buffer_size,
                                                     cudaStream_t stream,
                                                     bool row_major)
{
#if CUDART_VERSION >= 11020
  cusparseOrder_t order = row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
  cusparseSpMatDescr_t matA;
  cusparsecreatecsr(&matA,
                    m,
                    n,
                    nnz,
                    const_cast<int*>(csrRowPtrA),
                    const_cast<int*>(csrColIndA),
                    const_cast<double*>(csrValA));

  cusparseDnMatDescr_t matB;
  cusparsecreatednmat(&matB,
                      static_cast<int64_t>(m),
                      static_cast<int64_t>(n),
                      static_cast<int64_t>(lda),
                      const_cast<double*>(A),
                      order);

  cusparseStatus_t result = cusparseSparseToDense_bufferSize(
    handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer_size);

  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(matA));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(matB));

#else
  cusparseStatus_t result = CUSPARSE_STATUS_SUCCESS;
  buffer_size[0]          = 0;

#endif

  return result;
}

template <typename T>
cusparseStatus_t cusparsecsr2dense(cusparseHandle_t handle,
                                   int m,
                                   int n,
                                   int nnz,
                                   const cusparseMatDescr_t descrA,
                                   const T* csrValA,
                                   const int* csrRowPtrA,
                                   const int* csrColIndA,
                                   T* A,
                                   int lda,
                                   void* buffer,
                                   cudaStream_t stream,
                                   bool row_major = false);

template <>
inline cusparseStatus_t cusparsecsr2dense(cusparseHandle_t handle,
                                          int m,
                                          int n,
                                          int nnz,
                                          const cusparseMatDescr_t descrA,
                                          const float* csrValA,
                                          const int* csrRowPtrA,
                                          const int* csrColIndA,
                                          float* A,
                                          int lda,
                                          void* buffer,
                                          cudaStream_t stream,
                                          bool row_major)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));

#if CUDART_VERSION >= 11020
  cusparseOrder_t order = row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
  cusparseSpMatDescr_t matA;
  cusparsecreatecsr(&matA,
                    m,
                    n,
                    nnz,
                    const_cast<int*>(csrRowPtrA),
                    const_cast<int*>(csrColIndA),
                    const_cast<float*>(csrValA));

  cusparseDnMatDescr_t matB;
  cusparsecreatednmat(&matB,
                      static_cast<int64_t>(m),
                      static_cast<int64_t>(n),
                      static_cast<int64_t>(lda),
                      const_cast<float*>(A),
                      order);

  cusparseStatus_t result =
    cusparseSparseToDense(handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer);

  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(matA));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(matB));

  return result;
#else
  return cusparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
#endif
}
template <>
inline cusparseStatus_t cusparsecsr2dense(cusparseHandle_t handle,
                                          int m,
                                          int n,
                                          int nnz,
                                          const cusparseMatDescr_t descrA,
                                          const double* csrValA,
                                          const int* csrRowPtrA,
                                          const int* csrColIndA,
                                          double* A,
                                          int lda,
                                          void* buffer,
                                          cudaStream_t stream,
                                          bool row_major)
{
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));

#if CUDART_VERSION >= 11020
  cusparseOrder_t order = row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
  cusparseSpMatDescr_t matA;
  cusparsecreatecsr(&matA,
                    m,
                    n,
                    nnz,
                    const_cast<int*>(csrRowPtrA),
                    const_cast<int*>(csrColIndA),
                    const_cast<double*>(csrValA));

  cusparseDnMatDescr_t matB;
  cusparsecreatednmat(&matB,
                      static_cast<int64_t>(m),
                      static_cast<int64_t>(n),
                      static_cast<int64_t>(lda),
                      const_cast<double*>(A),
                      order);

  cusparseStatus_t result =
    cusparseSparseToDense(handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer);

  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(matA));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(matB));

  return result;
#else

  return cusparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
#endif
}

/** @} */

}  // namespace detail
}  // namespace sparse
}  // namespace raft