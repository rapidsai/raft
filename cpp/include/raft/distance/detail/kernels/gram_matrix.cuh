/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/distance/distance.cuh>

#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/gemm.cuh>

namespace raft::distance::kernels::detail {

/**
 * Base class for general Gram matrices
 * A Gram matrix is the Hermitian matrix of inner probucts G_ik = <x_i, x_k>
 * Here, the  inner product is evaluated for all elements from vectors sets X1,
 * and X2.
 *
 * To be more precise, on exit the output buffer will store:
 * - if is_row_major == true: out[j+k*n1] = <x1_j, x2_k>,
 * - if is_row_major == false: out[j*n2 + k] = <x1_j, x2_k>,
 * where x1_j is the j-th vector from the x1 set and x2_k is the k-th vector
 * from the x2 set.
 */
template <typename math_t>
class GramMatrixBase {
  cublasHandle_t cublas_handle;

 public:
  GramMatrixBase(cublasHandle_t cublas_handle) : cublas_handle(cublas_handle){};

  virtual ~GramMatrixBase(){};

  /** Convenience function to evaluate the Gram matrix for two vector sets.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1
   * @param ld2 leading dimension of x2
   * @param ld_out leading dimension of out
   */
  virtual void operator()(const math_t* x1,
                          int n1,
                          int n_cols,
                          const math_t* x2,
                          int n2,
                          math_t* out,
                          bool is_row_major,
                          cudaStream_t stream,
                          int ld1    = 0,
                          int ld2    = 0,
                          int ld_out = 0)
  {
    if (ld1 <= 0) { ld1 = is_row_major ? n_cols : n1; }
    if (ld2 <= 0) { ld2 = is_row_major ? n_cols : n2; }
    if (ld_out <= 0) { ld_out = is_row_major ? n2 : n1; }
    evaluate(x1, n1, n_cols, x2, n2, out, is_row_major, stream, ld1, ld2, ld_out);
  }

  virtual void operator()(const raft::handle_t& handle,
                          const int* x1_indptr,
                          const int* x1_indices,
                          const math_t* x1_data,
                          int x1_nnz,
                          int n1,
                          int n_cols,
                          const math_t* x2_data,
                          int n2,
                          math_t* out,
                          bool is_row_major,
                          cudaStream_t stream,
                          int ld2       = 0,
                          int ld_out    = 0,
                          math_t* norm  = nullptr,
                          int offset_x1 = 0,
                          int* idx_x2   = 0)

  {
    if (ld2 <= 0) { ld2 = is_row_major ? n_cols : n2; }
    if (ld_out <= 0) { ld_out = is_row_major ? n2 : n1; }
    evaluateSparseX1(handle,
                     x1_indptr,
                     x1_indices,
                     x1_data,
                     x1_nnz,
                     n1,
                     n_cols,
                     x2_data,
                     n2,
                     out,
                     is_row_major,
                     stream,
                     ld2,
                     ld_out,
                     norm,
                     offset_x1,
                     idx_x2);
  }

  virtual void operator()(const raft::handle_t& handle,
                          const int* x1_indptr,
                          const int* x1_indices,
                          const math_t* x1_data,
                          int x1_nnz,
                          int n1,
                          int n_cols,
                          const int* x2_indptr,
                          const int* x2_indices,
                          const math_t* x2_data,
                          int x2_nnz,
                          int n2,
                          math_t* out,
                          bool is_row_major,
                          cudaStream_t stream,
                          int ld_out = 0)
  {
    if (ld_out <= 0) { ld_out = is_row_major ? n2 : n1; }
    evaluateSparse(handle,
                   x1_indptr,
                   x1_indices,
                   x1_data,
                   x1_nnz,
                   n1,
                   n_cols,
                   x2_indptr,
                   x2_indices,
                   x2_data,
                   x2_nnz,
                   n2,
                   out,
                   is_row_major,
                   stream,
                   ld_out);
  }

  /** Evaluate the Gram matrix for two vector sets using simple dot product.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1 (usually it is n1)
   * @param ld2 leading dimension of x2 (usually it is n2)
   * @param ld_out leading dimension of out (usually it is n1)
   */
  virtual void evaluate(const math_t* x1,
                        int n1,
                        int n_cols,
                        const math_t* x2,
                        int n2,
                        math_t* out,
                        bool is_row_major,
                        cudaStream_t stream,
                        int ld1,
                        int ld2,
                        int ld_out)
  {
    linear(x1, n1, n_cols, x2, n2, out, is_row_major, stream, ld1, ld2, ld_out);
  }

  virtual void evaluateSparseX1(const raft::handle_t& handle,
                                const int* x1_indptr,
                                const int* x1_indices,
                                const math_t* x1_data,
                                int x1_nnz,
                                int n1,
                                int n_cols,
                                const math_t* x2_data,
                                int n2,
                                math_t* out,
                                bool is_row_major,
                                cudaStream_t stream,
                                int ld2,
                                int ld_out,
                                math_t* norm,
                                int offset_x1,
                                int* idx_x2)
  {
    linearSparseX1(handle,
                   x1_indptr,
                   x1_indices,
                   x1_data,
                   x1_nnz,
                   n1,
                   n_cols,
                   x2_data,
                   n2,
                   out,
                   is_row_major,
                   stream,
                   ld2,
                   ld_out);
  }

  virtual void evaluateSparse(const raft::handle_t& handle,
                              const int* x1_indptr,
                              const int* x1_indices,
                              const math_t* x1_data,
                              int x1_nnz,
                              int n1,
                              int n_cols,
                              const int* x2_indptr,
                              const int* x2_indices,
                              const math_t* x2_data,
                              int x2_nnz,
                              int n2,
                              math_t* out,
                              bool is_row_major,
                              cudaStream_t stream,
                              int ld_out)
  {
    linearSparse(handle,
                 x1_indptr,
                 x1_indices,
                 x1_data,
                 x1_nnz,
                 n1,
                 n_cols,
                 x2_indptr,
                 x2_indices,
                 x2_data,
                 x2_nnz,
                 n2,
                 out,
                 is_row_major,
                 stream,
                 ld_out);
  }

  // private:
  // The following methods should be private, they are kept public to avoid:
  // "error: The enclosing parent function ("distance") for an extended
  // __device__ lambda cannot have private or protected access within its class"

  /** Calculates the Gram matrix using simple dot product between vector sets.
   *
   * out = x1 * x2
   *
   * Can be used as a building block for more complex kernel functions.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1
   * @param ld2 leading dimension of x2
   * @param ld_out leading dimension of out
   */
  void linear(const math_t* x1,
              int n1,
              int n_cols,
              const math_t* x2,
              int n2,
              math_t* out,
              bool is_row_major,
              cudaStream_t stream,
              int ld1,
              int ld2,
              int ld_out)
  {
    math_t alpha = 1.0;
    math_t beta  = 0.0;
    if (is_row_major) {
      // #TODO: Call from public API when ready
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemm(cublas_handle,
                                                       CUBLAS_OP_T,
                                                       CUBLAS_OP_N,
                                                       n2,
                                                       n1,
                                                       n_cols,
                                                       &alpha,
                                                       x2,
                                                       ld2,
                                                       x1,
                                                       ld1,
                                                       &beta,
                                                       out,
                                                       ld_out,
                                                       stream));
    } else {
      // #TODO: Call from public API when ready
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemm(cublas_handle,
                                                       CUBLAS_OP_N,
                                                       CUBLAS_OP_T,
                                                       n1,
                                                       n2,
                                                       n_cols,
                                                       &alpha,
                                                       x1,
                                                       ld1,
                                                       x2,
                                                       ld2,
                                                       &beta,
                                                       out,
                                                       ld_out,
                                                       stream));
    }
  }

  void linearSparseX1(const raft::handle_t& handle,
                      const int* x1_indptr,
                      const int* x1_indices,
                      const math_t* x1_data,
                      int x1_nnz,
                      int n1,
                      int n_cols,
                      const math_t* x2_data,
                      int n2,
                      math_t* out,
                      bool is_row_major,
                      cudaStream_t stream,
                      int ld2,
                      int ld_out)
  {
    math_t alpha = 1.0;
    math_t beta  = 0.0;

    cusparseSpMatDescr_t descrX1;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(&descrX1,
                                                              n1,
                                                              n_cols,
                                                              x1_nnz,
                                                              const_cast<int*>(x1_indptr),
                                                              const_cast<int*>(x1_indices),
                                                              const_cast<math_t*>(x1_data)));

    auto order = is_row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;

    cusparseDnMatDescr_t descrX2;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
      &descrX2, n2, n_cols, ld2, const_cast<math_t*>(x2_data), order));

    cusparseDnMatDescr_t descrOut;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
      &descrOut, n1, n2, ld_out, const_cast<math_t*>(out), order));

    auto alg = order == CUSPARSE_ORDER_COL ? CUSPARSE_SPMM_CSR_ALG1 : CUSPARSE_SPMM_CSR_ALG2;

    // compute X1*X2^T
    auto opX1 = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opX2 = CUSPARSE_OPERATION_TRANSPOSE;

    size_t bufferSize;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm_bufferSize(handle.get_cusparse_handle(),
                                                                    opX1,
                                                                    opX2,
                                                                    &alpha,
                                                                    descrX1,
                                                                    descrX2,
                                                                    &beta,
                                                                    descrOut,
                                                                    alg,
                                                                    &bufferSize,
                                                                    stream));

    raft::interruptible::synchronize(stream);

    rmm::device_uvector<math_t> tmp(bufferSize, stream);

    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(handle.get_cusparse_handle(),
                                                         opX1,
                                                         opX2,
                                                         &alpha,
                                                         descrX1,
                                                         descrX2,
                                                         &beta,
                                                         descrOut,
                                                         alg,
                                                         tmp.data(),
                                                         stream));

    RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(descrX1));
    RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(descrX2));
    RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(descrOut));
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  void linearSparse(const raft::handle_t& handle,
                    const int* x1_indptr,
                    const int* x1_indices,
                    const math_t* x1_data,
                    int x1_nnz,
                    int n1,
                    int n_cols,
                    const int* x2_indptr,
                    const int* x2_indices,
                    const math_t* x2_data,
                    int x2_nnz,
                    int n2,
                    math_t* out,
                    bool is_row_major,
                    cudaStream_t stream,
                    int ld_out)
  {
    int minor_out = is_row_major ? n2 : n1;
    ASSERT(ld_out == minor_out, "Sparse linear Kernel distance does not support ld_out parameter");
    distanceSparse(handle,
                   x1_indptr,
                   x1_indices,
                   x1_data,
                   x1_nnz,
                   n1,
                   n_cols,
                   x2_indptr,
                   x2_indices,
                   x2_data,
                   x2_nnz,
                   n2,
                   out,
                   is_row_major,
                   stream,
                   raft::distance::DistanceType::InnerProduct);
  }

  void distanceSparse(const raft::handle_t& handle,
                      const int* x1_indptr,
                      const int* x1_indices,
                      const math_t* x1_data,
                      int x1_nnz,
                      int n1,
                      int n_cols,
                      const int* x2_indptr,
                      const int* x2_indices,
                      const math_t* x2_data,
                      int x2_nnz,
                      int n2,
                      math_t* out,
                      bool is_row_major,
                      cudaStream_t stream,
                      raft::distance::DistanceType metric,
                      float metricArg = 0.0)
  {
    raft::sparse::distance::distances_config_t<int, math_t> dist_config(handle);

    // switch a,b based on is_row_major
    if (!is_row_major) {
      dist_config.a_nrows   = n2;
      dist_config.a_ncols   = n_cols;
      dist_config.a_nnz     = x2_nnz;
      dist_config.a_indptr  = const_cast<int*>(x2_indptr);
      dist_config.a_indices = const_cast<int*>(x2_indices);
      dist_config.a_data    = const_cast<math_t*>(x2_data);
      dist_config.b_nrows   = n1;
      dist_config.b_ncols   = n_cols;
      dist_config.b_nnz     = x1_nnz;
      dist_config.b_indptr  = const_cast<int*>(x1_indptr);
      dist_config.b_indices = const_cast<int*>(x1_indices);
      dist_config.b_data    = const_cast<math_t*>(x1_data);
    } else {
      dist_config.a_nrows   = n1;
      dist_config.a_ncols   = n_cols;
      dist_config.a_nnz     = x1_nnz;
      dist_config.a_indptr  = const_cast<int*>(x1_indptr);
      dist_config.a_indices = const_cast<int*>(x1_indices);
      dist_config.a_data    = const_cast<math_t*>(x1_data);
      dist_config.b_nrows   = n2;
      dist_config.b_ncols   = n_cols;
      dist_config.b_nnz     = x2_nnz;
      dist_config.b_indptr  = const_cast<int*>(x2_indptr);
      dist_config.b_indices = const_cast<int*>(x2_indices);
      dist_config.b_data    = const_cast<math_t*>(x2_data);
    }

    if (raft::sparse::distance::supportedDistance.find(metric) ==
        raft::sparse::distance::supportedDistance.end())
      THROW("DistanceType not supported: %d", metric);

    raft::sparse::distance::pairwiseDistance(out, dist_config, metric, metricArg);
  }

  /** Calculates the Gram matrix using Euclidean distance.
   *
   * Can be used as a building block for more complex kernel functions.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1
   * @param ld2 leading dimension of x2
   * @param ld_out leading dimension of out
   */
  virtual void distance(const math_t* x1,
                        int n1,
                        int n_cols,
                        const math_t* x2,
                        int n2,
                        math_t* out,
                        bool is_row_major,
                        cudaStream_t stream,
                        int ld1,
                        int ld2,
                        int ld_out)
  {
    raft::distance::distance<raft::distance::DistanceType::L2Unexpanded, math_t, math_t, math_t>(
      x1, x2, out, n1, n2, n_cols, stream, is_row_major);
  }
};
};  // end namespace raft::distance::kernels::detail