/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <raft/core/device_resources.hpp>
#include <raft/distance/detail/matrix/matrix.hpp>
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
 protected:
  cublasHandle_t cublas_handle;
  bool legacy_interface;

 public:
  GramMatrixBase() : legacy_interface(false){};
  [[deprecated]] GramMatrixBase(cublasHandle_t cublas_handle)
    : cublas_handle(cublas_handle), legacy_interface(true){};

  virtual ~GramMatrixBase(){};
  /** Convenience function to evaluate the Gram matrix for two vector sets.
   *  Vector sets are provided in Matrix format
   *
   * @param [in] x1 device matrix, size [n1*n_cols]
   * @param [in] x2 device matrix, size [n2*n_cols]
   * @param [out] out (dense) device matrix to store the Gram matrix, size [n1*n2]
   * @param [in] handle raft handle
   * @param norm_x1 optional L2-norm of x1's rows for computation within RBF.
   * @param norm_x2 optional L2-norm of x2's rows for computation within RBF.
   */
  virtual void operator()(const raft::distance::matrix::detail::Matrix<math_t>& x1,
                          const raft::distance::matrix::detail::Matrix<math_t>& x2,
                          raft::distance::matrix::detail::DenseMatrix<math_t>& out,
                          const raft::device_resources& handle,
                          math_t* norm_x1 = nullptr,
                          math_t* norm_x2 = nullptr)
  {
    ASSERT(x1.n_rows == out.n_rows,
           "GramMatrix input matrix dimensions for x1 and out do not match");
    ASSERT(x2.n_rows == out.n_cols,
           "GramMatrix input matrix dimensions for x2 and out do not match");
    ASSERT(x1.n_cols == x2.n_cols, "GramMatrix input matrix dimensions for x1 and x2 do not match");
    evaluate(x1, x2, out, handle, norm_x1, norm_x2);
  }

  /** Evaluate the Gram matrix for two vector sets using simple dot product.
   *
   * @param [in] x1 device matrix, size [n1*n_cols]
   * @param [in] x2 device matrix, size [n2*n_cols]
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] handle raft handle
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  virtual void evaluate(const raft::distance::matrix::detail::Matrix<math_t>& x1,
                        const raft::distance::matrix::detail::Matrix<math_t>& x2,
                        raft::distance::matrix::detail::DenseMatrix<math_t>& out,
                        const raft::device_resources& handle,
                        math_t* norm_x1,
                        math_t* norm_x2)
  {
    linear(x1, x2, out, handle);
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
   * @param [in] x1 device matrix, size [n1*n_cols]
   * @param [in] x2 device matrix, size [n2*n_cols]
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] stream cuda stream
   + @param [in] handle raft handle
   */
  void linear(const raft::distance::matrix::detail::DenseMatrix<math_t>& x1,
              const raft::distance::matrix::detail::DenseMatrix<math_t>& x2,
              raft::distance::matrix::detail::DenseMatrix<math_t>& out,
              cudaStream_t stream,
              cublasHandle_t cublas_handle)
  {
    ASSERT(x1.is_row_major == x2.is_row_major,
           "GramMatrix leading dimensions for x1 and x2 do not match");
    ASSERT(x2.is_row_major == out.is_row_major,
           "GramMatrix leading dimensions for x2 and out do not match");

    math_t alpha = 1.0;
    math_t beta  = 0.0;
    if (out.is_row_major) {
      // #TODO: Call from public API when ready
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemm(cublas_handle,
                                                       CUBLAS_OP_T,
                                                       CUBLAS_OP_N,
                                                       out.n_cols,
                                                       out.n_rows,
                                                       x1.n_cols,
                                                       &alpha,
                                                       x2.data,
                                                       x2.ld,
                                                       x1.data,
                                                       x1.ld,
                                                       &beta,
                                                       out.data,
                                                       out.ld,
                                                       stream));
    } else {
      // #TODO: Call from public API when ready
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemm(cublas_handle,
                                                       CUBLAS_OP_N,
                                                       CUBLAS_OP_T,
                                                       out.n_rows,
                                                       out.n_cols,
                                                       x1.n_cols,
                                                       &alpha,
                                                       x1.data,
                                                       x1.ld,
                                                       x2.data,
                                                       x2.ld,
                                                       &beta,
                                                       out.data,
                                                       out.ld,
                                                       stream));
    }
  }

  void linear(const raft::distance::matrix::detail::CsrMatrix<math_t>& x1,
              const raft::distance::matrix::detail::DenseMatrix<math_t>& x2,
              raft::distance::matrix::detail::DenseMatrix<math_t>& out,
              cudaStream_t stream,
              const cusparseHandle_t& cusparse_handle)
  {
    math_t alpha = 1.0;
    math_t beta  = 0.0;

    ASSERT(x2.is_row_major == out.is_row_major,
           "GramMatrix leading dimensions for x2 and out do not match");

    cusparseSpMatDescr_t descrX1;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(&descrX1,
                                                              x1.n_rows,
                                                              x1.n_cols,
                                                              x1.nnz,
                                                              const_cast<int*>(x1.indptr),
                                                              const_cast<int*>(x1.indices),
                                                              const_cast<math_t*>(x1.data)));

    auto order = out.is_row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;

    cusparseDnMatDescr_t descrX2;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
      &descrX2, x2.n_rows, x2.n_cols, x2.ld, const_cast<math_t*>(x2.data), order));

    cusparseDnMatDescr_t descrOut;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
      &descrOut, out.n_rows, out.n_cols, out.ld, const_cast<math_t*>(out.data), order));

    auto alg = order == CUSPARSE_ORDER_COL ? CUSPARSE_SPMM_CSR_ALG1 : CUSPARSE_SPMM_CSR_ALG2;

    // compute X1*X2^T
    auto opX1 = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opX2 = CUSPARSE_OPERATION_TRANSPOSE;

    size_t bufferSize;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm_bufferSize(cusparse_handle,
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

    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(cusparse_handle,
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

  void linear(const raft::distance::matrix::detail::CsrMatrix<math_t>& x1,
              const raft::distance::matrix::detail::CsrMatrix<math_t>& x2,
              raft::distance::matrix::detail::DenseMatrix<math_t>& out,
              const raft::device_resources& handle)
  {
    int minor_out = out.is_row_major ? out.n_cols : out.n_rows;
    ASSERT(out.ld == minor_out, "Sparse linear Kernel distance does not support ld_out parameter");
    raft::sparse::distance::distances_config_t<int, math_t> dist_config(handle);

    // switch a,b based on is_row_major
    if (!out.is_row_major) {
      dist_config.a_nrows   = x2.n_rows;
      dist_config.a_ncols   = x2.n_cols;
      dist_config.a_nnz     = x2.nnz;
      dist_config.a_indptr  = const_cast<int*>(x2.indptr);
      dist_config.a_indices = const_cast<int*>(x2.indices);
      dist_config.a_data    = const_cast<math_t*>(x2.data);
      dist_config.b_nrows   = x1.n_rows;
      dist_config.b_ncols   = x1.n_cols;
      dist_config.b_nnz     = x1.nnz;
      dist_config.b_indptr  = const_cast<int*>(x1.indptr);
      dist_config.b_indices = const_cast<int*>(x1.indices);
      dist_config.b_data    = const_cast<math_t*>(x1.data);
    } else {
      dist_config.a_nrows   = x1.n_rows;
      dist_config.a_ncols   = x1.n_cols;
      dist_config.a_nnz     = x1.nnz;
      dist_config.a_indptr  = const_cast<int*>(x1.indptr);
      dist_config.a_indices = const_cast<int*>(x1.indices);
      dist_config.a_data    = const_cast<math_t*>(x1.data);
      dist_config.b_nrows   = x2.n_rows;
      dist_config.b_ncols   = x2.n_cols;
      dist_config.b_nnz     = x2.nnz;
      dist_config.b_indptr  = const_cast<int*>(x2.indptr);
      dist_config.b_indices = const_cast<int*>(x2.indices);
      dist_config.b_data    = const_cast<math_t*>(x2.data);
    }

    raft::sparse::distance::pairwiseDistance(
      out.data, dist_config, raft::distance::DistanceType::InnerProduct, 0.0);
  }

  /** Calculates the Gram matrix using simple dot product between vector sets.
   *
   * out = x1 * x2
   *
   * Can be used as a building block for more complex kernel functions.
   *
   * @param [in] x1 device matrix, size [n1*n_cols]
   * @param [in] x2 device matrix, size [n2*n_cols]
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] handle raft handle
   */
  void linear(const raft::distance::matrix::detail::Matrix<math_t>& x1,
              const raft::distance::matrix::detail::Matrix<math_t>& x2,
              raft::distance::matrix::detail::DenseMatrix<math_t>& out,
              const raft::device_resources& handle)
  {
    // dispatch
    if (x1.isDense()) {
      ASSERT(x2.isDense(), "GramMatrix input matrix does not allow Dense*Csr");
      auto x1_dense = x1.asDense();
      auto x2_dense = x2.asDense();
      linear(*x1_dense, *x2_dense, out, handle.get_stream(), handle.get_cublas_handle());
    } else {
      auto x1_csr = x1.asCsr();
      if (x2.isDense()) {
        auto x2_dense = x2.asDense();
        linear(*x1_csr, *x2_dense, out, handle.get_stream(), handle.get_cusparse_handle());
      } else {
        auto x2_csr = x2.asCsr();
        linear(*x1_csr, *x2_csr, out, handle);
      }
    }
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
  [[deprecated]] virtual void evaluate(const math_t* x1,
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
    ASSERT(legacy_interface, "Legacy interface can only be used with legacy ctor.");
    raft::distance::matrix::detail::DenseMatrix dense1(
      const_cast<math_t*>(x1), n1, n_cols, is_row_major, ld1);
    raft::distance::matrix::detail::DenseMatrix dense2(
      const_cast<math_t*>(x2), n2, n_cols, is_row_major, ld2);
    raft::distance::matrix::detail::DenseMatrix dense_out(out, n1, n2, is_row_major, ld_out);
    linear(dense1, dense2, dense_out, stream, cublas_handle);
  }

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
  [[deprecated]] void operator()(const math_t* x1,
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
    ASSERT(legacy_interface, "Legacy interface can only be used with legacy ctor.");
    if (ld1 <= 0) { ld1 = is_row_major ? n_cols : n1; }
    if (ld2 <= 0) { ld2 = is_row_major ? n_cols : n2; }
    if (ld_out <= 0) { ld_out = is_row_major ? n2 : n1; }
    evaluate(x1, n1, n_cols, x2, n2, out, is_row_major, stream, ld1, ld2, ld_out);
  }
};

};  // end namespace raft::distance::kernels::detail
