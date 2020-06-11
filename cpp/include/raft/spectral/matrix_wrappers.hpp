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
    //TODO: call cusparse::csrmv
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
    //TODO: more work, here...
  }

  laplacian_matrix_t(
    handle_t const& raft_handle,
    GraphCSRView<index_type, index_type, value_type> const& csr_view)
    : sparse_matrix_t<index_type, value_type>(csr_view),
      diagonal_(raft_handle, csr_view.number_of_vertices_) {}

  // y = alpha*A*x + beta*y
  //
  void mv(value_type alpha, value_type const* __restrict__ x, value_type beta,
          value_type* __restrict__ y) const override {
    //TODO: call cusparse::csrmv ... and more
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
    //TODO: more work, here...
  }

  modularity_matrix_t(
    handle_t const& raft_handle,
    GraphCSRView<index_type, index_type, value_type> const& csr_view)
    : laplacian_matrix_t<index_type, value_type>(raft_handle, csr_view) {}

  // y = alpha*A*x + beta*y
  //
  void mv(value_type alpha, value_type const* __restrict__ x, value_type beta,
          value_type* __restrict__ y) const override {
    //TODO: call cusparse::csrmv ... and more
  }
};

}  // namespace matrix
}  // namespace raft
