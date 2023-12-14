/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusparse_handle.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cuda/functional>

#include <algorithm>

// =========================================================
// Useful macros
// =========================================================

// Get index of matrix entry
#define IDX(i, j, lda) ((i) + (j) * (lda))

namespace raft {
namespace spectral {
namespace matrix {
namespace detail {

using size_type = int;  // for now; TODO: move it in appropriate header

// Apply diagonal matrix to vector:
//
template <typename IndexType_, typename ValueType_>
RAFT_KERNEL diagmv(IndexType_ n,
                   ValueType_ alpha,
                   const ValueType_* __restrict__ D,
                   const ValueType_* __restrict__ x,
                   ValueType_* __restrict__ y)
{
  IndexType_ i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] += alpha * D[i] * x[i];
    i += blockDim.x * gridDim.x;
  }
}

// specifies type of algorithm used
// for SpMv:
//
enum struct sparse_mv_alg_t : int {
  SPARSE_MV_UNDEFINED = -1,
  SPARSE_MV_ALG_DEFAULT,  // generic, for any sparse matrix
  SPARSE_MV_ALG1,         // typical for CSR
  SPARSE_MV_ALG2          // may provide better performance for irregular sparse matrices
};

// Vector "view"-like aggregate for linear algebra purposes
//
template <typename value_type>
struct vector_view_t {
  value_type* buffer_;
  size_type size_;

  vector_view_t(value_type* buffer, size_type sz) : buffer_(buffer), size_(sz) {}

  vector_view_t(vector_view_t&& other) : buffer_(other.raw()), size_(other.size()) {}

  vector_view_t& operator=(vector_view_t&& other)
  {
    buffer_ = other.raw();
    size_   = other.size();
  }
};

template <typename value_type>
class vector_t {
 public:
  vector_t(resources const& raft_handle, size_type sz)
    : buffer_(sz, resource::get_cuda_stream(raft_handle)),
      thrust_policy(resource::get_thrust_policy(raft_handle))
  {
  }

  size_type size(void) const { return buffer_.size(); }

  value_type* raw(void) { return buffer_.data(); }

  value_type const* raw(void) const { return buffer_.data(); }

  value_type nrm1() const
  {
    return thrust::reduce(
      thrust_policy,
      buffer_.data(),
      buffer_.data() + buffer_.size(),
      value_type{0},
      cuda::proclaim_return_type<value_type>([] __device__(auto left, auto right) {
        auto abs_left  = left > 0 ? left : -left;
        auto abs_right = right > 0 ? right : -right;
        return abs_left + abs_right;
      }));
  }

  void fill(value_type value)
  {
    thrust::fill_n(thrust_policy, buffer_.data(), buffer_.size(), value);
  }

 private:
  using thrust_exec_policy_t =
    thrust::detail::execute_with_allocator<rmm::mr::thrust_allocator<char>,
                                           thrust::cuda_cub::execute_on_stream_base>;
  rmm::device_uvector<value_type> buffer_;
  const thrust_exec_policy_t thrust_policy;
};

template <typename index_type, typename value_type>
struct sparse_matrix_t {
  sparse_matrix_t(resources const& raft_handle,
                  index_type const* row_offsets,
                  index_type const* col_indices,
                  value_type const* values,
                  index_type const nrows,
                  index_type const ncols,
                  index_type const nnz)
    : handle_(raft_handle),
      row_offsets_(row_offsets),
      col_indices_(col_indices),
      values_(values),
      nrows_(nrows),
      ncols_(ncols),
      nnz_(nnz)
  {
  }

  sparse_matrix_t(resources const& raft_handle,
                  index_type const* row_offsets,
                  index_type const* col_indices,
                  value_type const* values,
                  index_type const nrows,
                  index_type const nnz)
    : handle_(raft_handle),
      row_offsets_(row_offsets),
      col_indices_(col_indices),
      values_(values),
      nrows_(nrows),
      ncols_(nrows),
      nnz_(nnz)
  {
  }

  template <typename CSRView>
  sparse_matrix_t(resources const& raft_handle, CSRView const& csr_view)
    : handle_(raft_handle),
      row_offsets_(csr_view.offsets),
      col_indices_(csr_view.indices),
      values_(csr_view.edge_data),
      nrows_(csr_view.number_of_vertices),
      ncols_(csr_view.number_of_vertices),
      nnz_(csr_view.number_of_edges)
  {
  }

  virtual ~sparse_matrix_t(void) =
    default;  // virtual because used as base for following matrix types

  // y = alpha*A*x + beta*y
  //(Note: removed const-ness of x, because CUDA 11 SpMV
  // descriptor creation works with non-const, and const-casting
  // down is dangerous)
  //
  virtual void mv(value_type alpha,
                  value_type* __restrict__ x,
                  value_type beta,
                  value_type* __restrict__ y,
                  sparse_mv_alg_t alg = sparse_mv_alg_t::SPARSE_MV_ALG1,
                  bool transpose      = false,
                  bool symmetric      = false) const
  {
    using namespace sparse;

    RAFT_EXPECTS(x != nullptr, "Null x buffer.");
    RAFT_EXPECTS(y != nullptr, "Null y buffer.");

    auto cusparse_h = resource::get_cusparse_handle(handle_);
    auto stream     = resource::get_cuda_stream(handle_);

    cusparseOperation_t trans = transpose ? CUSPARSE_OPERATION_TRANSPOSE :  // transpose
                                  CUSPARSE_OPERATION_NON_TRANSPOSE;         // non-transpose

#if not defined CUDA_ENFORCE_LOWER and CUDA_VER_10_1_UP
    auto size_x = transpose ? nrows_ : ncols_;
    auto size_y = transpose ? ncols_ : nrows_;

    cusparseSpMVAlg_t spmv_alg = translate_algorithm(alg);

    // create descriptors:
    //(below casts are necessary, because
    // cusparseCreateCsr(...) takes non-const
    // void*; the casts should be harmless)
    //
    cusparseSpMatDescr_t matA;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(&matA,
                                                              nrows_,
                                                              ncols_,
                                                              nnz_,
                                                              const_cast<index_type*>(row_offsets_),
                                                              const_cast<index_type*>(col_indices_),
                                                              const_cast<value_type*>(values_)));

    cusparseDnVecDescr_t vecX;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednvec(&vecX, size_x, x));

    rmm::device_uvector<value_type> y_tmp(size_y, stream);
    raft::copy(y_tmp.data(), y, size_y, stream);

    cusparseDnVecDescr_t vecY;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednvec(&vecY, size_y, y_tmp.data()));

    // get (scratch) external device buffer size:
    //
    size_t bufferSize;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmv_buffersize(
      cusparse_h, trans, &alpha, matA, vecX, &beta, vecY, spmv_alg, &bufferSize, stream));

    // allocate external buffer:
    //
    vector_t<value_type> external_buffer(handle_, bufferSize);

    // finally perform SpMV:
    //
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmv(
      cusparse_h, trans, &alpha, matA, vecX, &beta, vecY, spmv_alg, external_buffer.raw(), stream));

    // FIXME: This is a workaround for a cusparse issue being encountered in CUDA 12
    raft::copy(y, y_tmp.data(), size_y, stream);
    // free descriptors:
    //(TODO: maybe wrap them in a RAII struct?)
    //
    RAFT_CUSPARSE_TRY(cusparseDestroyDnVec(vecY));
    RAFT_CUSPARSE_TRY(cusparseDestroyDnVec(vecX));
    RAFT_CUSPARSE_TRY(cusparseDestroySpMat(matA));
#else
    RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsesetpointermode(cusparse_h, CUSPARSE_POINTER_MODE_HOST, stream));
    cusparseMatDescr_t descr = 0;
    RAFT_CUSPARSE_TRY(cusparseCreateMatDescr(&descr));
    if (symmetric) {
      RAFT_CUSPARSE_TRY(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_SYMMETRIC));
    } else {
      RAFT_CUSPARSE_TRY(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    }
    RAFT_CUSPARSE_TRY(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecsrmv(cusparse_h,
                                                          trans,
                                                          nrows_,
                                                          ncols_,
                                                          nnz_,
                                                          &alpha,
                                                          descr,
                                                          values_,
                                                          row_offsets_,
                                                          col_indices_,
                                                          x,
                                                          &beta,
                                                          y,
                                                          stream));
    RAFT_CUSPARSE_TRY(cusparseDestroyMatDescr(descr));
#endif
  }

  resources const& get_handle(void) const { return handle_; }

#if not defined CUDA_ENFORCE_LOWER and CUDA_VER_10_1_UP
  cusparseSpMVAlg_t translate_algorithm(sparse_mv_alg_t alg) const
  {
    switch (alg) {
      case sparse_mv_alg_t::SPARSE_MV_ALG1: return CUSPARSE_SPMV_CSR_ALG1;
      case sparse_mv_alg_t::SPARSE_MV_ALG2: return CUSPARSE_SPMV_CSR_ALG2;
      default: return CUSPARSE_SPMV_ALG_DEFAULT;
    }
  }
#endif

  // private: // maybe not, keep this ASAPBNS ("as simple as possible, but not simpler"); hence,
  // aggregate

  raft::resources const& handle_;
  index_type const* row_offsets_;
  index_type const* col_indices_;
  value_type const* values_;
  index_type const nrows_;
  index_type const ncols_;
  index_type const nnz_;
};

template <typename index_type, typename value_type>
struct laplacian_matrix_t : sparse_matrix_t<index_type, value_type> {
  laplacian_matrix_t(resources const& raft_handle,
                     index_type const* row_offsets,
                     index_type const* col_indices,
                     value_type const* values,
                     index_type const nrows,
                     index_type const nnz)
    : sparse_matrix_t<index_type, value_type>(
        raft_handle, row_offsets, col_indices, values, nrows, nnz),
      diagonal_(raft_handle, nrows)
  {
    vector_t<value_type> ones{raft_handle, nrows};
    ones.fill(1.0);
    sparse_matrix_t<index_type, value_type>::mv(1, ones.raw(), 0, diagonal_.raw());
  }

  laplacian_matrix_t(resources const& raft_handle,
                     sparse_matrix_t<index_type, value_type> const& csr_m)
    : sparse_matrix_t<index_type, value_type>(raft_handle,
                                              csr_m.row_offsets_,
                                              csr_m.col_indices_,
                                              csr_m.values_,
                                              csr_m.nrows_,
                                              csr_m.nnz_),
      diagonal_(raft_handle, csr_m.nrows_)
  {
    vector_t<value_type> ones{raft_handle, csr_m.nrows_};
    ones.fill(1.0);
    sparse_matrix_t<index_type, value_type>::mv(1, ones.raw(), 0, diagonal_.raw());
  }

  // y = alpha*A*x + beta*y
  //
  void mv(value_type alpha,
          value_type* __restrict__ x,
          value_type beta,
          value_type* __restrict__ y,
          sparse_mv_alg_t alg = sparse_mv_alg_t::SPARSE_MV_ALG1,
          bool transpose      = false,
          bool symmetric      = false) const override
  {
    constexpr int BLOCK_SIZE = 1024;
    auto n                   = sparse_matrix_t<index_type, value_type>::nrows_;

    auto handle   = sparse_matrix_t<index_type, value_type>::get_handle();
    auto cublas_h = resource::get_cublas_handle(handle);
    auto stream   = resource::get_cuda_stream(handle);

    // scales y by beta:
    //
    if (beta == 0) {
      RAFT_CUDA_TRY(cudaMemsetAsync(y, 0, n * sizeof(value_type), stream));
    } else if (beta != 1) {
      // TODO: Call from public API when ready
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasscal(cublas_h, n, &beta, y, 1, stream));
    }

    // Apply diagonal matrix
    //
    dim3 gridDim{std::min<unsigned int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535), 1, 1};

    dim3 blockDim{BLOCK_SIZE, 1, 1};
    diagmv<<<gridDim, blockDim, 0, stream>>>(n, alpha, diagonal_.raw(), x, y);
    RAFT_CHECK_CUDA(stream);

    // Apply adjacency matrix
    //
    sparse_matrix_t<index_type, value_type>::mv(-alpha, x, 1, y, alg, transpose, symmetric);
  }

  vector_t<value_type> diagonal_;
};

template <typename index_type, typename value_type>
struct modularity_matrix_t : laplacian_matrix_t<index_type, value_type> {
  modularity_matrix_t(resources const& raft_handle,
                      index_type const* row_offsets,
                      index_type const* col_indices,
                      value_type const* values,
                      index_type const nrows,
                      index_type const nnz)
    : laplacian_matrix_t<index_type, value_type>(
        raft_handle, row_offsets, col_indices, values, nrows, nnz)
  {
    edge_sum_ = laplacian_matrix_t<index_type, value_type>::diagonal_.nrm1();
  }

  modularity_matrix_t(resources const& raft_handle,
                      sparse_matrix_t<index_type, value_type> const& csr_m)
    : laplacian_matrix_t<index_type, value_type>(raft_handle, csr_m)
  {
    edge_sum_ = laplacian_matrix_t<index_type, value_type>::diagonal_.nrm1();
  }

  // y = alpha*A*x + beta*y
  //
  void mv(value_type alpha,
          value_type* __restrict__ x,
          value_type beta,
          value_type* __restrict__ y,
          sparse_mv_alg_t alg = sparse_mv_alg_t::SPARSE_MV_ALG1,
          bool transpose      = false,
          bool symmetric      = false) const override
  {
    auto n = sparse_matrix_t<index_type, value_type>::nrows_;

    auto handle   = sparse_matrix_t<index_type, value_type>::get_handle();
    auto cublas_h = resource::get_cublas_handle(handle);
    auto stream   = resource::get_cuda_stream(handle);

    // y = A*x
    //
    sparse_matrix_t<index_type, value_type>::mv(alpha, x, 0, y, alg, transpose, symmetric);
    value_type dot_res;

    // gamma = d'*x
    //
    // Cublas::dot(this->n, D.raw(), 1, x, 1, &dot_res);
    // TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(
      raft::linalg::detail::cublasdot(cublas_h,
                                      n,
                                      laplacian_matrix_t<index_type, value_type>::diagonal_.raw(),
                                      1,
                                      x,
                                      1,
                                      &dot_res,
                                      stream));

    // y = y -(gamma/edge_sum)*d
    //
    value_type gamma_ = -dot_res / edge_sum_;
    // TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(
      raft::linalg::detail::cublasaxpy(cublas_h,
                                       n,
                                       &gamma_,
                                       laplacian_matrix_t<index_type, value_type>::diagonal_.raw(),
                                       1,
                                       y,
                                       1,
                                       stream));
  }

  value_type edge_sum_;
};

}  // namespace detail
}  // namespace matrix
}  // namespace spectral
}  // namespace raft
