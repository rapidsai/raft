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

#include <raft/core/detail/device_mdarray.hpp>
#include <raft/core/sparse_matrix.hpp>

namespace raft {

template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          typename ContainerPolicy = detail::device_uvector_policy<ElementType>>
using device_structure_owning_csr_matrix =
  structure_owning_csr_matrix<ElementType, IndptrType, IndicesType, NZType, true, ContainerPolicy>

  template <typename ElementType,
            typename IndptrType,
            typename IndicesType,
            typename NZType,
            typename ContainerPolicy = detail::device_uvector_policy<ElementType>>
  using device_structure_preserving_csr_matrix = structure_preserving_csr_matrix<ElementType,
                                                                                 IndptrType,
                                                                                 IndicesType,
                                                                                 NZType,
                                                                                 true,
                                                                                 ContainerPolicy>

  template <typename ElementType,
            typename RowType,
            typename ColType,
            typename NZType,
            typename ContainerPolicy = detail::device_uvector_policy<ElementType>>
  using device_structure_owning_coo_matrix =
    structure_owning_coo_matrix<ElementType, RowType, ColType, NZType, true, ContainerPolicy>

  template <typename ElementType,
            typename RowType,
            typename ColType,
            typename NZType,
            typename ContainerPolicy = detail::device_uvector_policy<ElementType>>
  using device_structure_preserving_coo_matrix =
    structure_preserving_coo_matrix<ElementType, RowType, ColType, NZType, true, ContainerPolicy>

  /**
   * Create a structure-owning sparse matrix in the compressed-sparse row format. Structure-owning
   * means that all of the underlying vectors (data, indptr, indices) are owned by the csr_matrix
   * instance. If not known up front, the sparsity can be ignored in this factory function and
   * `resize()` invoked on the instance once the sparsity is known.
   * @tparam ElementType
   * @tparam IndptrType
   * @tparam IndicesType
   * @tparam NZType
   * @param handle
   * @param n_rows
   * @param n_cols
   * @param nnz
   * @return
   */
  template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
  auto make_csr_matrix(raft::device_resources const& handle,
                       IndptrType n_rows,
                       IndicesType n_cols,
                       NZType nnz = 0)
{
  using csr_matrix_t =
    device_structure_owning_csr_matrix<ElementType, IndptrType, IndicesType, NZType>;
  return csr_matrix_t(handle, n_rows, n_cols, nnz);
}

/**
 * Create a structure-preserving sparse matrix in the compressed-sparse row format.
 * Structure-preserving means that a view of the csr structure is supplied, allowing the values in
 * the structure to change but not the structure itself. The csr_matrix instance does not own the
 * structure, the sparsity must be known up front, and cannot be resized later.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param handle
 * @param structure_
 * @return
 */
template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
auto make_csr_matrix(raft::device_resources const& handle,
                     compressed_structure_view<IndptrType, IndicesType, NZType> structure_)
{
  using csr_matrix_t =
    device_structure_preserving_csr_matrix<ElementType, IndptrType, IndicesType, NZType>;
  return csr_matrix_t(handle, std::make_shared(structure_));
}

/**
 * Create a structure-owning sparse matrix in the coordinate format. Structure-owning means that
 * all of the underlying vectors (data, indptr, indices) are owned by the coo_matrix instance. If
 * not known up front, the sparsity can be ignored in this factory function and `resize()` invoked
 * on the instance once the sparsity is known.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param handle
 * @param n_rows
 * @param n_cols
 * @param nnz
 * @return
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_coo_matrix(raft::device_resources const& handle,
                     RowType n_rows,
                     ColType n_cols,
                     NZType nnz = 0)
{
  using coo_matrix_t = device_structure_owning_coo_matrix<ElementType, RowType, ColType, NZType>;
  return coo_matrix_t(handle, n_rows, n_cols, nnz);
}

/**
 * Create a structure-preserving sparse matrix in the coordinate format. Structure-preserving means
 * that a view of the coo structure is supplied, allowing the values in the structure to change but
 * not the structure itself. The csr_matrix instance does not own the structure, the sparsity must
 * be known up front, and cannot be resized later.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param handle
 * @param structure_
 * @return
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_coo_matrix(raft::device_resources const& handle,
                     coordinate_structure_view<RowType, ColType, NZType> structure_)
{
  using coo_matrix_t =
    device_structure_preserving_coo_matrix<ElementType, RowType, ColType, NZType>;
  return coo_matrix_t(handle, std::make_shared(structure_));
}
};  // namespace raft