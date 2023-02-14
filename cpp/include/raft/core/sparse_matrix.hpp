/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <raft/core/span.hpp>

namespace raft {

// TODO: Define container_policy interface

template <typename NZType>
class sparse_structure {
  virtual NZType get_nnz();
};

/**
 * A non-owning view into a compressed sparse structure
 *
 * The structure representation does not have a value/weight
 * component so that its const-ness can be varied from it.
 *
 * @tparam IndptrType
 * @tparam IndicesType
 */
template <typename IndptrType, typename IndicesType, typename NZType>
class compressed_structure_view : public sparse_structure<NZType> {
  using row_type = IndptrType;
  using col_type = IndicesType;

  virtual span<IndptrType> get_indptr();
  virtual span<IndicesType> get_indices();
};

/**
 * Represents a sparse compressed structure (or adjacency list)
 * which can be used to model both a CSR and CSC matrix.
 *
 * The structure representation does not have a value/weight
 * component so that its const-ness can be varied from it.
 *
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam ContainerPolicy
 */
template <typename IndptrType, typename IndicesType, typename NZType, typename ContainerPolicy>
class compressed_structure : public sparse_structure<NZType> {
  using view_type = compressed_structure_view<IndptrType, IndicesType>;
  using row_type  = IndptrType;
  using col_type  = IndicesType;

  virtual view_type view();
};

/**
 * A non-owning view into a coordinate structure
 *
 * The structure representation does not have a value/weight
 * component so that its const-ness can be varied from it.
 *
 * @tparam RowType
 * @tparam ColType
 */
template <typename RowType, typename ColType, typename NZType>
class coordinate_structure_view : public sparse_structure<NZType> {
  using row_type = RowType;
  using col_type = ColType;

  virtual span<RowType> get_rows();
  virtual span<ColType> get_cols();
};

/**
 * Represents a sparse coordinate structure (or edge list)
 * which can be used to model a COO matrix.
 *
 * The structure representation does not have a value/weight
 * component so that its const-ness can be varied from it.
 *
 * @tparam RowType
 * @tparam ColType
 * @tparam ContainerPolicy
 */
template <typename RowType, typename ColType, typename NZType, typename ContainerPolicy>
class coordinate_structure : sparse_structure<NZType> {
  using view_type = coordinate_structure_view<RowType, ColType>;
  using row_type  = RowType;
  using col_type  = ColType;

  virtual view_type view();
};

/**
 * A non-owning view of a sparse matrix, which includes a
 * structure component coupled with its elements/weights
 *
 * @tparam ElementType
 * @tparam sparse_structure
 */
template <typename ElementType, typename sparse_structure>
class sparse_matrix_view {
  using structure_view_type = sparse_structure::view_type;

  virtual structure_view_type get_structure();
  virtual span<ElementType> get_elements();
};

/**
 * An owning container for a sparse matrix, which includes a
 * structure component coupled with its elements/weights
 * @tparam ElementType
 * @tparam sparse_structure
 * @tparam ContainerPolicy
 */
template <typename ElementType, typename sparse_structure, typename ContainerPolicy>
class sparse_matrix {
  using view_type = sparse_matrix_view<ElementType, sparse_structure>;

  virtual view_type view();
};

template <typename ElementType, typename IndptrType, typename IndicesType>
class csr_matrix_view : sparse_matrix<ElementType, compressed_structure<IndptrType, IndicesType>> {
};

template <typename ElementType, typename IndptrType, typename IndicesType, typename ContainerPolicy>
class csr_matrix : sparse_matrix<ElementType, compressed_structure<IndptrType, IndicesType>> {
  using view_type = csr_matrix_view<ElementType, IndptrType, IndicesType>;

  csr_matrix(const csr_matrix&)  = delete;
  csr_matrix(const csr_matrix&&) = delete;

  virtual view_type view();
};

template <typename ElementType, typename RowType, typename ColType>
class coo_matrix_view : sparse_matrix<ElementType, compressed_structure<RowType, ColType>> {
};

template <typename ElementType, typename RowType, typename ColType, typename ContainerPolicy>
class coo_matrix : sparse_matrix<ElementType, compressed_structure<RowType, ColType>> {
  using view_type = coo_matrix_view<ElementType, RowType, ColType>;

  coo_matrix(const coo_matrix&)  = delete;
  coo_matrix(const coo_matrix&&) = delete;

  virtual view_type view();
};
}  // namespace raft