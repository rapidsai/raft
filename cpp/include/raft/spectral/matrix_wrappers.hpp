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

// Apply diagonal matrix to vector:
//
template <typename IndexType, typename ValueType>
static __global__ void diagmv(IndexType n, ValueType alpha,
                              const ValueType* __restrict__ D,
                              const ValueType* __restrict__ x,
                              ValueType* __restrict__ y) {
  IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] += alpha * D[i] * x[i];
    i += blockDim.x * gridDim.x;
  }
}

// specifies type of algorithm used
// for SpMv:
//
enum struct SparseMvAlgoT : int {
  kUndefined = -1,
  kDefault,  // generic, for any sparse matrix
  kAlgo1,    // typical for CSR
  kAlgo2     // may provide better performamce for irregular sparse matrices
};

// Vector "view"-like aggregate for linear algebra purposes
//
template <typename ValueT>
struct vector_view_t {
  ValueT* buffer;
  size_type size;

  vector_view_t(ValueT* buffer, size_type sz)
    : buffer(buffer), size(sz) {}

  vector_view_t(vector_view_t&& other)
    : buffer(other.buffer), size(other.size) {
    other.buffer = nullptr;
    other.size = 0;
  }

  vector_view_t& operator=(vector_view_t&& other) {
    buffer = other.buffer;
    size = other.size;

    other.buffer = nullptr;
    other.size = 0;
  }
};

// allocatable vector, using raft handle allocator
//
template <typename ValueT>
class vector_t {
  handle_t const& handle_;
  ValueT* buffer_;
  size_type size_;
  cudaStream_t stream_;

 public:
  vector_t(handle_t const& raft_handle, size_type sz)
    : handle_(raft_handle),
      buffer_(
        static_cast<ValueT*>(raft_handle.get_device_allocator()->allocate(
          sz * sizeof(ValueT), raft_handle.get_stream()))),
      size_(sz),
      stream_(raft_handle.get_stream()) {}

  ~vector_t() {
    handle_.get_device_allocator()->deallocate(buffer_, size_, stream_);
  }

  size_type size() const { return size_; }

  ValueT* raw() { return buffer_; }

  ValueT const* raw() const { return buffer_; }

  template <typename ThrustExecPolicy>
  ValueT nrm1(ThrustExecPolicy t_exe_pol) const {
    return thrust::reduce(t_exe_pol, buffer_, buffer_ + size_, ValueT{0},
                          [] __device__(auto left, auto right) {
                            auto abs_left = left > 0 ? left : -left;
                            auto abs_right = right > 0 ? right : -right;
                            return abs_left + abs_right;
                          });
  }

  template <typename ThrustExecPolicy>
  void fill(ThrustExecPolicy t_exe_pol, ValueT value) {
    thrust::fill_n(t_exe_pol, buffer_, size_, value);
  }
};

template <typename IndexType, typename ValueT>
struct sparse_matrix_t {
  sparse_matrix_t(handle_t const& raft_handle, IndexType const* row_offsets,
                  IndexType const* col_indices, ValueT const* values,
                  IndexType const nrows, IndexType const ncols,
                  IndexType const nnz)
    : handle(raft_handle),
      row_offsets(row_offsets),
      col_indices(col_indices),
      values(values),
      nrows(nrows),
      ncols(ncols),
      nnz(nnz) {}

  sparse_matrix_t(handle_t const& raft_handle, IndexType const* row_offsets,
                  IndexType const* col_indices, ValueT const* values,
                  IndexType const nrows, IndexType const nnz)
    : handle(raft_handle),
      row_offsets(row_offsets),
      col_indices(col_indices),
      values(values),
      nrows(nrows),
      ncols(nrows),
      nnz(nnz) {}

  template <typename CSRView>
  sparse_matrix_t(handle_t const& raft_handle, CSRView const& csr_view)
    : handle(raft_handle),
      row_offsets(csr_view.offsets),
      col_indices(csr_view.indices),
      values(csr_view.edge_data),
      nrows(csr_view.number_of_vertices),
      ncols(csr_view.number_of_vertices),
      nnz(csr_view.number_of_edges) {}

  virtual ~sparse_matrix_t() =
    default;  // virtual because used as base for following matrix types

  // y = alpha*A*x + beta*y
  //(Note: removed const-ness of x, because CUDA 11 SpMV
  // descriptor creation works with non-const, and const-casting
  // down is dangerous)
  //
  virtual void mv(ValueT alpha, ValueT* __restrict__ x, ValueT beta,
                  ValueT* __restrict__ y,
                  SparseMvAlgoT alg = SparseMvAlgoT::kAlgo1,
                  bool transpose = false, bool symmetric = false) const {
    using namespace sparse;

    RAFT_EXPECTS(x != nullptr, "Null x buffer.");
    RAFT_EXPECTS(y != nullptr, "Null y buffer.");

    auto cusparse_h = handle.get_cusparse_handle();
    auto stream = handle.get_stream();

    cusparseOperation_t trans =
      transpose ? CUSPARSE_OPERATION_TRANSPOSE :  // transpose
        CUSPARSE_OPERATION_NON_TRANSPOSE;         //non-transpose

#if not defined CUDA_ENFORCE_LOWER and CUDA_VER_10_1_UP
    auto size_x = transpose ? nrows : ncols;
    auto size_y = transpose ? ncols : nrows;

    cusparseSpMVAlg_t spmv_alg = translate_algorithm(alg);

    //create descriptors:
    //(below casts are necessary, because
    // cusparseCreateCsr(...) takes non-const
    // void*; the casts should be harmless)
    //
    cusparseSpMatDescr_t mat_a;
    CUSPARSE_CHECK(cusparsecreatecsr(
      &mat_a, nrows, ncols, nnz, const_cast<IndexType*>(row_offsets),
      const_cast<IndexType*>(col_indices), const_cast<ValueT*>(values)));

    cusparseDnVecDescr_t vec_x;
    CUSPARSE_CHECK(cusparsecreatednvec(&vec_x, size_x, x));

    cusparseDnVecDescr_t vec_y;
    CUSPARSE_CHECK(cusparsecreatednvec(&vec_y, size_y, y));

    //get (scratch) external device buffer size:
    //
    size_t buffer_size;
    CUSPARSE_CHECK(cusparsespmv_buffersize(cusparse_h, trans, &alpha, mat_a,
                                           vec_x, &beta, vec_y, spmv_alg,
                                           &buffer_size, stream));

    //allocate external buffer:
    //
    vector_t<ValueT> external_buffer(handle, buffer_size);

    //finally perform SpMV:
    //
    CUSPARSE_CHECK(cusparsespmv(cusparse_h, trans, &alpha, mat_a, vec_x, &beta,
                                vec_y, spmv_alg, external_buffer.raw(), stream));

    //free descriptors:
    //(TODO: maybe wrap them in a RAII struct?)
    //
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x));
    CUSPARSE_CHECK(cusparseDestroySpMat(mat_a));
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
    CUSPARSE_CHECK(cusparsecsrmv(cusparse_h, trans, nrows, ncols, nnz,
                                 &alpha, descr, values, row_offsets,
                                 col_indices, x, &beta, y, stream));
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));
#endif
  }

  handle_t const& get_handle() const { return handle; }

#if not defined CUDA_ENFORCE_LOWER and CUDA_VER_10_1_UP
  cusparseSpMVAlg_t translate_algorithm(SparseMvAlgoT alg) const {
    switch (alg) {
      case SparseMvAlgoT::kAlgo1:
        return CUSPARSE_CSRMV_ALG1;
      case SparseMvAlgoT::kAlgo2:
        return CUSPARSE_CSRMV_ALG2;
      default:
        return CUSPARSE_MV_ALG_DEFAULT;
    }
  }
#endif

  //private: // maybe not, keep this ASAPBNS ("as simple as possible, but not simpler"); hence, aggregate

  handle_t const& handle;
  IndexType const* row_offsets;
  IndexType const* col_indices;
  ValueT const* values;
  IndexType const nrows;
  IndexType const ncols;
  IndexType const nnz;
};

template <typename IndexType, typename ValueT>
struct laplacian_matrix_t : sparse_matrix_t<IndexType, ValueT> {
  template <typename ThrustExePolicy>
  laplacian_matrix_t(handle_t const& raft_handle,
                     ThrustExePolicy thrust_exec_policy,
                     IndexType const* row_offsets,
                     IndexType const* col_indices, ValueT const* values,
                     IndexType const nrows, IndexType const nnz)
    : sparse_matrix_t<IndexType, ValueT>(raft_handle, row_offsets,
                                              col_indices, values, nrows, nnz),
      diagonal(raft_handle, nrows) {
    vector_t<ValueT> ones{raft_handle, nrows};
    ones.fill(thrust_exec_policy, 1.0);
    sparse_matrix_t<IndexType, ValueT>::mv(1, ones.raw(), 0,
                                                diagonal.raw());
  }

  template <typename ThrustExePolicy>
  laplacian_matrix_t(handle_t const& raft_handle,
                     ThrustExePolicy thrust_exec_policy,
                     sparse_matrix_t<IndexType, ValueT> const& csr_m)
    : sparse_matrix_t<IndexType, ValueT>(raft_handle, csr_m.row_offsets,
                                              csr_m.col_indices, csr_m.values,
                                              csr_m.nrows, csr_m.nnz),
      diagonal(raft_handle, csr_m.nrows) {
    vector_t<ValueT> ones{raft_handle, csr_m.nrows};
    ones.fill(thrust_exec_policy, 1.0);
    sparse_matrix_t<IndexType, ValueT>::mv(1, ones.raw(), 0,
                                                diagonal.raw());
  }

  // y = alpha*A*x + beta*y
  //
  void mv(ValueT alpha, ValueT* __restrict__ x, ValueT beta,
          ValueT* __restrict__ y,
          SparseMvAlgoT alg = SparseMvAlgoT::kAlgo1,
          bool transpose = false, bool symmetric = false) const override {
    constexpr int kBlockSize = 1024;
    auto n = sparse_matrix_t<IndexType, ValueT>::nrows;

    auto cublas_h =
      sparse_matrix_t<IndexType, ValueT>::get_handle().get_cublas_handle();
    auto stream =
      sparse_matrix_t<IndexType, ValueT>::get_handle().get_stream();

    // scales y by beta:
    //
    if (beta == 0) {
      CUDA_TRY(cudaMemsetAsync(y, 0, n * sizeof(ValueT), stream));
    } else if (beta != 1) {
      CUBLAS_CHECK(linalg::cublasscal(cublas_h, n, &beta, y, 1, stream));
    }

    // Apply diagonal matrix
    //
    dim3 grid_dim{
      std::min<unsigned int>((n + kBlockSize - 1) / kBlockSize, 65535), 1, 1};

    dim3 block_dim{kBlockSize, 1, 1};
    diagmv<<<grid_dim, block_dim, 0, stream>>>(n, alpha, diagonal.raw(), x, y);
    CHECK_CUDA(stream);

    // Apply adjacency matrix
    //
    sparse_matrix_t<IndexType, ValueT>::mv(-alpha, x, 1, y, alg, transpose,
                                                symmetric);
  }

  vector_t<ValueT> diagonal;
};

template <typename IndexType, typename ValueT>
struct modularity_matrix_t : laplacian_matrix_t<IndexType, ValueT> {
  template <typename ThrustExePolicy>
  modularity_matrix_t(handle_t const& raft_handle,
                      ThrustExePolicy thrust_exec_policy,
                      IndexType const* row_offsets,
                      IndexType const* col_indices, ValueT const* values,
                      IndexType const nrows, IndexType const nnz)
    : laplacian_matrix_t<IndexType, ValueT>(
        raft_handle, thrust_exec_policy, row_offsets, col_indices, values,
        nrows, nnz) {
    edge_sum = laplacian_matrix_t<IndexType, ValueT>::diagonal.nrm1(
      thrust_exec_policy);
  }

  template <typename ThrustExePolicy>
  modularity_matrix_t(handle_t const& raft_handle,
                      ThrustExePolicy thrust_exec_policy,
                      sparse_matrix_t<IndexType, ValueT> const& csr_m)
    : laplacian_matrix_t<IndexType, ValueT>(raft_handle,
                                                 thrust_exec_policy, csr_m) {
    edge_sum = laplacian_matrix_t<IndexType, ValueT>::diagonal.nrm1(
      thrust_exec_policy);
  }

  // y = alpha*A*x + beta*y
  //
  void mv(ValueT alpha, ValueT* __restrict__ x, ValueT beta,
          ValueT* __restrict__ y,
          SparseMvAlgoT alg = SparseMvAlgoT::kAlgo1,
          bool transpose = false, bool symmetric = false) const override {
    auto n = sparse_matrix_t<IndexType, ValueT>::nrows;

    auto cublas_h =
      sparse_matrix_t<IndexType, ValueT>::get_handle().get_cublas_handle();
    auto stream =
      sparse_matrix_t<IndexType, ValueT>::get_handle().get_stream();

    // y = A*x
    //
    sparse_matrix_t<IndexType, ValueT>::mv(alpha, x, 0, y, alg, transpose,
                                                symmetric);
    ValueT dot_res;

    // gamma = d'*x
    //
    // Cublas::dot(this->n, D.raw(), 1, x, 1, &dot_res);
    CUBLAS_CHECK(linalg::cublasdot(
      cublas_h, n, laplacian_matrix_t<IndexType, ValueT>::diagonal.raw(),
      1, x, 1, &dot_res, stream));

    // y = y -(gamma/edge_sum)*d
    //
    ValueT gamma = -dot_res / edge_sum;
    CUBLAS_CHECK(linalg::cublasaxpy(
      cublas_h, n, &gamma,
      laplacian_matrix_t<IndexType, ValueT>::diagonal.raw(), 1, y, 1,
      stream));
  }

  ValueT edge_sum;
};

}  // namespace matrix
}  // namespace raft
