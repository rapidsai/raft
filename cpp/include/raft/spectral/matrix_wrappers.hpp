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

#include <raft/graph.hpp>
#include <raft/sparse/cusparse_wrappers.h>

namespace raft{
namespace matrix {

using size_type = int; // for now; TODO: move it in appropriate header
  
template <typename index_type, typename value_type>
struct sparse_matrix_t {
  sparse_matrix_t(index_type const* row_offsets,
                  index_type const* col_indices,
                  value_type const* values,
                  index_type const nnz,
                  index_type const nrows) :
    row_offsets_(row_offsets),
    col_indices_(col_indices),
    values_(values),
    nrows_(nrows),
    nnz_(nnz)
  {
  }

  sparse_matrix_t(const GraphCSRView<index_type, index_type, value_type>& csr_view): 
    row_offsets_(csr_view.offsets_),
    col_indices_(csr_view.indices_),
    values_(csr_view.edge_data_),
    nrows_(csr_view.number_of_vertices_),
    nnz_(csr_view.number_of_edges_)
  {
  }
    

  virtual ~sparse_matrix_t(void) = default; // virtual because used as base for following matrix types
  
  // y = alpha*A*x + beta*y
  //
  template<typename exe_policy_t>
  void mv(value_type alpha,
          value_type const* __restrict__ x,
          value_type beta,
          value_type* __restrict__ y,
          exe_policy_t&& policy,
          cudaStream_t stream = nullptr) const
  {
    //TODO: call cusparse::csrmv
  }
  
  //private: // maybe not, keep this ASAP ("as simple as possible"); hence, aggregate
  
  index_type const* row_offsets_;
  index_type const* col_indices_;
  value_type const* values_; // TODO: const-ness of this is debatable; cusparse primitives may not accept it...
  index_type const nrows_;
  index_type const nnz_;
};

} // namespace matrix
} // namespace raft
