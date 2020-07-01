/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/handle.hpp>
#include <raft/spectral/sm_utils.hpp>

#include <thrust/fill.h>
#include <thrust/reduce.h>

#include <algorithm>

// =========================================================
// Useful macros
// =========================================================

// Get index of matrix entry
#define IDX(i, j, lda) ((i) + (j) * (lda))

namespace raft {
namespace matrix {

using size_type = int;  // for now; TODO: move it in appropriate header

// Vector "view"-like aggregate for linear algebra purposes
//
template <typename value_type>
struct vector_view_t {
  value_type* buffer_;
  size_type size_;

  vector_view_t(value_type* buffer, size_type sz)
    : buffer_(buffer), size_(sz) {}

  vector_view_t(vector_view_t&& other)
    : buffer_(other.buffer_), size_(other.size_) {
    other.buffer_ = nullptr;
    other.size_ = 0;
  }

  vector_view_t& operator=(vector_view_t&& other) {
    buffer_ = other.buffer_;
    size_ = other.size_;

    other.buffer_ = nullptr;
    other.size_ = 0;
  }
};

// allocatable vector, using raft handle allocator
//
template <typename value_type>
class vector_t {
  handle_t const& handle_;
  value_type* buffer_;
  size_type size_;
  cudaStream_t stream_;

 public:
  vector_t(handle_t const& raft_handle, size_type sz)
    : handle_(raft_handle),
      buffer_(
        static_cast<value_type*>(raft_handle.get_device_allocator()->allocate(
          sz * sizeof(value_type), raft_handle.get_stream()))),
      size_(sz),
      stream_(raft_handle.get_stream()) {}

  ~vector_t(void) {
    handle_.get_device_allocator()->deallocate(buffer_, size_, stream_);
  }

  size_type size(void) const { return size_; }

  value_type* raw(void) { return buffer_; }

  value_type const* raw(void) const { return buffer_; }

  template <typename ThrustExecPolicy>
  value_type nrm1(ThrustExecPolicy t_exe_pol) const {
    return thrust::reduce(t_exe_pol, buffer_, buffer_ + size_, value_type{0},
                          [] __device__(auto left, auto right) {
                            auto abs_left = left > 0 ? left : -left;
                            auto abs_right = right > 0 ? right : -right;
                            return abs_left + abs_right;
                          });
  }

  template <typename ThrustExecPolicy>
  void fill(ThrustExecPolicy t_exe_pol, value_type value) {
    thrust::fill_n(t_exe_pol, buffer_, size_, value);
  }
};

template <typename index_type, typename value_type>
struct sparse_matrix_t {
  sparse_matrix_t(handle_t const& raft_handle, index_type const* row_offsets,
                  index_type const* col_indices, value_type const* values,
                  index_type const nrows, index_type const nnz)
    : handle_(raft_handle),
      row_offsets_(row_offsets),
      col_indices_(col_indices),
      values_(values),
      nrows_(nrows),
      nnz_(nnz) {}

  template <typename CSRView>
  sparse_matrix_t(handle_t const& raft_handle, CSRView const& csr_view)
    : handle_(raft_handle),
      row_offsets_(csr_view.offsets),
      col_indices_(csr_view.indices),
      values_(csr_view.edge_data),
      nrows_(csr_view.number_of_vertices),
      nnz_(csr_view.number_of_edges) {}

  virtual ~sparse_matrix_t(void) =
    default;  // virtual because used as base for following matrix types

  // y = alpha*A*x + beta*y
  //(Note: removed const-ness of x, because CUDA 11 SpMV
  // descriptor creation works with non-const, and const-casting
  // down is dangerous)
  //
  virtual void mv(value_type alpha, value_type* __restrict__ x, value_type beta,
                  value_type* __restrict__ y, bool transpose = false,
                  bool symmetric = false) const {
    using namespace sparse;

    RAFT_EXPECTS(x != nullptr, "Null x buffer.");
    RAFT_EXPECTS(y != nullptr, "Null y buffer.");

    auto cusparse_h = handle_.get_cusparse_handle();
    auto stream = handle_.get_stream();

    cusparseOperation_t trans =
      transpose ? CUSPARSE_OPERATION_TRANSPOSE :  // transpose
        CUSPARSE_OPERATION_NON_TRANSPOSE;         //non-transpose

#if __CUDACC_VER_MAJOR__ > 10

    //create descriptors:
    //
    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparsecreatecsr(&matA, nrows_, nrows_, nnz_, row_offsets_,
                                     col_indices_, values_));

    cusparseDnVecDescr_t vecX;
    CUSPARSE_CHECK(cusparsecreatednvec(&vecX, nrows_, x));

    cusparseDnVecDescr_t vecY;
    CUSPARSE_CHECK(cusparsecreatednvec(&vecY, nrows_, y));

    //get (scratch) external device buffer size:
    //
    size_t bufferSize;
    CUSPARSE_CHECK(cusparsespmv_buffersize(cusparse_h, opA, &alpha, matA, vecX,
                                           &beta, vecY, alg, &bufferSize,
                                           stream));

    //allocate external buffer:
    //
    vector_t<value_type> external_buffer(handle_, bufferSize);

    //finally perform SpMV:
    //
    CUSPARSE_CHECK(cusparsespmv(cusparse_h, trans, &alpha, matA, vecX, &beta,
                                vecY, CUSPARSE_CSRMV_ALG1,
                                external_buffer.raw(), stream));

    //free descriptors:
    //(TODO: maybe wrap them in a RAII struct?)
    //
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
#else
    CUSPARSE_CHECK(
      cusparsesetpointermode(cusparse_h, CUSPARSE_POINTER_MODE_HOST, stream));
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    if (symmetric) {
      CUSPARSE_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_SYMMETRIC));
    } else {
      CUSPARSE_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    }
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparsecsrmv(cusparse_h, trans, nrows_, nrows_, nnz_,
                                 &alpha, descr, values_, row_offsets_,
                                 col_indices_, x, &beta, y, stream));
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));
#endif
  }

  handle_t const& get_handle(void) const { return handle_; }

  //private: // maybe not, keep this ASAPBNS ("as simple as possible, but not simpler"); hence, aggregate

  handle_t const& handle_;
  index_type const* row_offsets_;
  index_type const* col_indices_;
  value_type const* values_;
  index_type const nrows_;
  index_type const nnz_;
};

template <typename index_type, typename value_type>
struct laplacian_matrix_t : sparse_matrix_t<index_type, value_type> {
  template <typename ThrustExePolicy>
  laplacian_matrix_t(handle_t const& raft_handle,
                     ThrustExePolicy thrust_exec_policy,
                     index_type const* row_offsets,
                     index_type const* col_indices, value_type const* values,
                     index_type const nrows, index_type const nnz)
    : sparse_matrix_t<index_type, value_type>(raft_handle, row_offsets,
                                              col_indices, values, nrows, nnz),
      diagonal_(raft_handle, nrows) {
    vector_t<value_type> ones{raft_handle, nrows};
    ones.fill(thrust_exec_policy, 1.0);
    sparse_matrix_t<index_type, value_type>::mv(1, ones.raw(), 0,
                                                diagonal_.raw());
  }

  template <typename ThrustExePolicy>
  laplacian_matrix_t(handle_t const& raft_handle,
                     ThrustExePolicy thrust_exec_policy,
                     sparse_matrix_t<index_type, value_type> const& csr_m)
    : sparse_matrix_t<index_type, value_type>(raft_handle, csr_m.row_offsets_,
                                              csr_m.col_indices_, csr_m.values_,
                                              csr_m.nrows_, csr_m.nnz_),
      diagonal_(raft_handle, csr_m.nrows_) {
    vector_t<value_type> ones{raft_handle, csr_m.nrows_};
    ones.fill(thrust_exec_policy, 1.0);
    sparse_matrix_t<index_type, value_type>::mv(1, ones.raw(), 0,
                                                diagonal_.raw());
  }

  // y = alpha*A*x + beta*y
  //
  void mv(value_type alpha, value_type* __restrict__ x, value_type beta,
          value_type* __restrict__ y, bool transpose = false,
          bool symmetric = false) const override {
    constexpr int BLOCK_SIZE = 1024;
    auto n = sparse_matrix_t<index_type, value_type>::nrows_;

    auto cublas_h =
      sparse_matrix_t<index_type, value_type>::get_handle().get_cublas_handle();
    auto stream =
      sparse_matrix_t<index_type, value_type>::get_handle().get_stream();

    // scales y by beta:
    //
    if (beta == 0) {
      CUDA_TRY(cudaMemsetAsync(y, 0, n * sizeof(value_type), stream));
    } else if (beta != 1) {
      CUBLAS_CHECK(linalg::cublasscal(cublas_h, n, &beta, y, 1, stream));
    }

    // Apply diagonal matrix
    //
    dim3 gridDim, blockDim;
    gridDim.x = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);
    gridDim.y = 1;
    gridDim.z = 1;
    blockDim.x = BLOCK_SIZE;
    blockDim.y = 1;
    blockDim.z = 1;
    utils::diagmv<<<gridDim, blockDim, 0, stream>>>(n, alpha, diagonal_.raw(),
                                                    x, y);
    CHECK_CUDA(stream);

    // Apply adjacency matrix
    //
    sparse_matrix_t<index_type, value_type>::mv(-alpha, x, 1, y);
  }

  vector_t<value_type> diagonal_;
};

template <typename index_type, typename value_type>
struct modularity_matrix_t : laplacian_matrix_t<index_type, value_type> {
  template <typename ThrustExePolicy>
  modularity_matrix_t(handle_t const& raft_handle,
                      ThrustExePolicy thrust_exec_policy,
                      index_type const* row_offsets,
                      index_type const* col_indices, value_type const* values,
                      index_type const nrows, index_type const nnz)
    : laplacian_matrix_t<index_type, value_type>(
        raft_handle, thrust_exec_policy, row_offsets, col_indices, values,
        nrows, nnz) {
    edge_sum_ = laplacian_matrix_t<index_type, value_type>::diagonal_.nrm1(
      thrust_exec_policy);
  }

  template <typename ThrustExePolicy>
  modularity_matrix_t(handle_t const& raft_handle,
                      ThrustExePolicy thrust_exec_policy,
                      sparse_matrix_t<index_type, value_type> const& csr_m)
    : laplacian_matrix_t<index_type, value_type>(raft_handle,
                                                 thrust_exec_policy, csr_m) {
    edge_sum_ = laplacian_matrix_t<index_type, value_type>::diagonal_.nrm1(
      thrust_exec_policy);
  }

  // y = alpha*A*x + beta*y
  //
  void mv(value_type alpha, value_type* __restrict__ x, value_type beta,
          value_type* __restrict__ y, bool transpose = false,
          bool symmetric = false) const override {
    auto n = sparse_matrix_t<index_type, value_type>::nrows_;

    auto cublas_h =
      sparse_matrix_t<index_type, value_type>::get_handle().get_cublas_handle();
    auto stream =
      sparse_matrix_t<index_type, value_type>::get_handle().get_stream();

    // y = A*x
    //
    sparse_matrix_t<index_type, value_type>::mv(alpha, x, 0, y);
    value_type dot_res;

    // gamma = d'*x
    //
    // Cublas::dot(this->n, D.raw(), 1, x, 1, &dot_res);
    CUBLAS_CHECK(linalg::cublasdot(
      cublas_h, n, laplacian_matrix_t<index_type, value_type>::diagonal_.raw(),
      1, x, 1, &dot_res, stream));

    // y = y -(gamma/edge_sum)*d
    //
    value_type gamma_ = -dot_res / edge_sum_;
    CUBLAS_CHECK(linalg::cublasaxpy(
      cublas_h, n, &gamma_,
      laplacian_matrix_t<index_type, value_type>::diagonal_.raw(), 1, y, 1,
      stream));
  }

  value_type edge_sum_;
};

}  // namespace matrix
}  // namespace raft
