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

#include <raft/sparse/cusparse_wrappers.h>
#include <raft/graph.hpp>
#include <raft/handle.hpp>
#include <raft/spectral/sm_utils.hpp>

// =========================================================
// Useful macros
// =========================================================

// Get index of matrix entry
#define IDX(i, j, lda) ((i) + (j) * (lda))

namespace raft {

namespace {

template <typename IndexType_, typename ValueType_>
static __global__ void scale_obs_kernel(IndexType_ m, IndexType_ n,
                                        ValueType_* obs) {
  IndexType_ i, j, k, index, mm;
  ValueType_ alpha, v, last;
  bool valid;
  // ASSUMPTION: kernel is launched with either 2, 4, 8, 16 or 32 threads in x-dimension

  // compute alpha
  mm = (((m + blockDim.x - 1) / blockDim.x) *
        blockDim.x);  // m in multiple of blockDim.x
  alpha = 0.0;
  // printf("[%d,%d,%d,%d] n=%d, li=%d, mn=%d \n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y, n,
  // li, mn);
  for (j = threadIdx.y + blockIdx.y * blockDim.y; j < n;
       j += blockDim.y * gridDim.y) {
    for (i = threadIdx.x; i < mm; i += blockDim.x) {
      // check if the thread is valid
      valid = i < m;

      // get the value of the last thread
      last = utils::shfl(alpha, blockDim.x - 1, blockDim.x);

      // if you are valid read the value from memory, otherwise set your value to 0
      alpha = (valid) ? obs[i + j * m] : 0.0;
      alpha = alpha * alpha;

      // do prefix sum (of size warpSize=blockDim.x =< 32)
      for (k = 1; k < blockDim.x; k *= 2) {
        v = utils::shfl_up(alpha, k, blockDim.x);
        if (threadIdx.x >= k) alpha += v;
      }
      // shift by last
      alpha += last;
    }
  }

  // scale by alpha
  alpha = utils::shfl(alpha, blockDim.x - 1, blockDim.x);
  alpha = std::sqrt(alpha);
  for (j = threadIdx.y + blockIdx.y * blockDim.y; j < n;
       j += blockDim.y * gridDim.y) {
    for (i = threadIdx.x; i < m; i += blockDim.x) {  // blockDim.x=32
      index = i + j * m;
      obs[index] = obs[index] / alpha;
    }
  }
}

template <typename IndexType_>
IndexType_ next_pow2(IndexType_ n) {
  IndexType_ v;
  // Reference:
  // http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float
  v = n - 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

template <typename IndexType_, typename ValueType_>
cudaError_t scale_obs(IndexType_ m, IndexType_ n, ValueType_* obs) {
  IndexType_ p2m;
  dim3 nthreads, nblocks;

  // find next power of 2
  p2m = next_pow2<IndexType_>(m);
  // setup launch configuration
  nthreads.x = max(2, min(p2m, 32));
  nthreads.y = 256 / nthreads.x;
  nthreads.z = 1;
  nblocks.x = 1;
  nblocks.y = (n + nthreads.y - 1) / nthreads.y;
  nblocks.z = 1;
  // printf("m=%d(%d),n=%d,obs=%p,
  // nthreads=(%d,%d,%d),nblocks=(%d,%d,%d)\n",m,p2m,n,obs,nthreads.x,nthreads.y,nthreads.z,nblocks.x,nblocks.y,nblocks.z);

  // launch scaling kernel (scale each column of obs by its norm)
  scale_obs_kernel<IndexType_, ValueType_><<<nblocks, nthreads>>>(m, n, obs);
  CUDA_CHECK_LAST();

  return cudaSuccess;
}

template <typename vertex_t, typename edge_t, typename weight_t,
          typename ThrustExePolicy>
void transform_eigen_matrix(handle_t handle, ThrustExePolicy thrust_exec_policy,
                            edge_t n, vertex_t nEigVecs, weight_t* eigVecs) {
  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  // Whiten eigenvector matrix
  for (auto i = 0; i < nEigVecs; ++i) {
    weight_t mean, std;

    mean = thrust::reduce(
      thrust_exec_policy, thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
      thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)));
    CUDA_CHECK_LAST();
    mean /= n;
    thrust::transform(thrust_exec_policy,
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)),
                      thrust::make_constant_iterator(mean),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::minus<weight_t>());
    CUDA_CHECK_LAST();

    CUBLAS_CHECK(
      cublasnrm2(cublas_h, n, eigVecs + IDX(0, i, n), 1, &std, stream));

    std /= std::sqrt(static_cast<weight_t>(n));

    thrust::transform(thrust_exec_policy,
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)),
                      thrust::make_constant_iterator(std),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::divides<weight_t>());
    CUDA_CHECK_LAST();
  }

  // Transpose eigenvector matrix
  //   TODO: in-place transpose
  {
    vector_t<weight_t> work(handle, nEigVecs * n);
    CUBLAS_CHECK(
      cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST, stream));

    CUBLAS_CHECK(cublasgeam(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, nEigVecs, n,
                            &one, eigVecs, n, &zero, (weight_t*)NULL, nEigVecs,
                            work.raw(), nEigVecs, stream));

    CUDA_TRY(cudaMemcpyAsync(eigVecs, work.raw(),
                             nEigVecs * n * sizeof(weight_t),
                             cudaMemcpyDeviceToDevice, stream));
  }
}

}  // namespace

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
};

template <typename index_type, typename value_type>
struct sparse_matrix_t {
  sparse_matrix_t(index_type const* row_offsets, index_type const* col_indices,
                  value_type const* values, index_type const nnz,
                  index_type const nrows)
    : row_offsets_(row_offsets),
      col_indices_(col_indices),
      values_(values),
      nrows_(nrows),
      nnz_(nnz) {}

  sparse_matrix_t(
    GraphCSRView<index_type, index_type, value_type> const& csr_view)
    : row_offsets_(csr_view.offsets_),
      col_indices_(csr_view.indices_),
      values_(csr_view.edge_data_),
      nrows_(csr_view.number_of_vertices_),
      nnz_(csr_view.number_of_edges_) {}

  virtual ~sparse_matrix_t(void) =
    default;  // virtual because used as base for following matrix types

  // y = alpha*A*x + beta*y
  //
  virtual void mv(value_type alpha, value_type const* __restrict__ x,
                  value_type beta, value_type* __restrict__ y) const {
    //TODO:
    //
    //Cusparse::set_pointer_mode_host();
    //cusparsecsrmv(...);
  }

  //private: // maybe not, keep this ASAPBNS ("as simple as possible, but not simpler"); hence, aggregate

  index_type const* row_offsets_;
  index_type const* col_indices_;
  value_type const*
    values_;  // TODO: const-ness of this is debatable; cusparse primitives may not accept it...
  index_type const nrows_;
  index_type const nnz_;
};

template <typename index_type, typename value_type>
struct laplacian_matrix_t : sparse_matrix_t<index_type, value_type> {
  laplacian_matrix_t(handle_t const& raft_handle, index_type const* row_offsets,
                     index_type const* col_indices, value_type const* values,
                     index_type const nrows, index_type const nnz)
    : sparse_matrix_t<index_type, value_type>(row_offsets, col_indices, values,
                                              nrows, nnz),
      diagonal_(raft_handle, nrows) {
    auto* v = diagonal_.raw();
    //TODO: more work, here:
    //
    // vector_t<value_type> ones(nrows);
    // ones.fill(1.0);
    // sparse_matrix_t::mv(1, ones.raw(), 0, diagonal_.raw());
  }

  laplacian_matrix_t(
    handle_t const& raft_handle,
    GraphCSRView<index_type, index_type, value_type> const& csr_view)
    : sparse_matrix_t<index_type, value_type>(csr_view),
      diagonal_(raft_handle, csr_view.number_of_vertices_) {
    //TODO: more work, here:
    //
    // vector_t<value_type> ones(csr_view.number_of_vertices_);
    // ones.fill(1.0);
    // sparse_matrix_t::mv(1, ones.raw(), 0, diagonal_.raw());
  }

  // y = alpha*A*x + beta*y
  //
  void mv(value_type alpha, value_type const* __restrict__ x, value_type beta,
          value_type* __restrict__ y) const override {
    //TODO: call cusparse::csrmv ... and more:
    //
    // if (beta == 0)
    //   CHECK_CUDA(cudaMemset(y, 0, (this->n) * sizeof(ValueType_)))
    //   else if (beta != 1)
    //     thrust::transform(thrust::device_pointer_cast(y),
    //                       thrust::device_pointer_cast(y + this->n),
    //                       thrust::make_constant_iterator(beta),
    //                       thrust::device_pointer_cast(y),
    //                       thrust::multiplies<ValueType_>());

    // // Apply diagonal matrix
    // dim3 gridDim, blockDim;
    // gridDim.x  = min(((this->n) + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);
    // gridDim.y  = 1;
    // gridDim.z  = 1;
    // blockDim.x = BLOCK_SIZE;
    // blockDim.y = 1;
    // blockDim.z = 1;
    // diagmv<<<gridDim, blockDim, 0, A->s>>>(this->n, alpha, D.raw(), x, y);
    // cudaCheckError();

    // // Apply adjacency matrix
    // sparse_matrix_t::mv(-alpha, x, 1, y);
  }

  vector_t<value_type> diagonal_;
};

template <typename index_type, typename value_type>
struct modularity_matrix_t : laplacian_matrix_t<index_type, value_type> {
  modularity_matrix_t(handle_t const& raft_handle,
                      index_type const* row_offsets,
                      index_type const* col_indices, value_type const* values,
                      index_type const nrows, index_type const nnz)
    : laplacian_matrix_t<index_type, value_type>(
        raft_handle, row_offsets, col_indices, values, nrows, nnz) {
    auto* v = laplacian_matrix_t<index_type, value_type>::diagonal_.raw();
    //TODO: more work, here:
    //
    // diag_nrm1_ = diagonal_.nrm1();
  }

  modularity_matrix_t(
    handle_t const& raft_handle,
    GraphCSRView<index_type, index_type, value_type> const& csr_view)
    : laplacian_matrix_t<index_type, value_type>(raft_handle, csr_view) {
    //TODO: more work, here:
    //
    // diag_nrm1_ = diagonal_.nrm1();
  }

  // y = alpha*A*x + beta*y
  //
  void mv(value_type alpha, value_type const* __restrict__ x, value_type beta,
          value_type* __restrict__ y) const override {
    //TODO: call cusparse::csrmv ... and more:
    //
    // // y = A*x
    // sparse_matrix_t::mv(alpha, x, 0, y);
    // value_type dot_res;
    // // gamma = d'*x
    // Cublas::dot(this->n, D.raw(), 1, x, 1, &dot_res);
    // // y = y -(gamma/edge_sum)*d
    // Cublas::axpy(this->n, -(dot_res / this->edge_sum), D.raw(), 1, y, 1);
  }

  value_type get_diag_nrm1(void) const { return diag_nrm1_; }

  value_type diag_nrm1_;
};

}  // namespace matrix
}  // namespace raft
