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
#include <raft/core/device_span.hpp>
#include <raft/core/sparse_matrix.hpp>

namespace raft {

template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          typename ContainerPolicy = detail::device_uvector_policy<ElementType>>
using device_sparsity_owning_csr_matrix =
  sparsity_owning_csr_matrix<ElementType, IndptrType, IndicesType, NZType, true, ContainerPolicy>

  template <typename ElementType,
            typename IndptrType,
            typename IndicesType,
            typename NZType,
            typename ContainerPolicy = detail::device_uvector_policy<ElementType>>
  using device_sparsity_preserving_csr_matrix = sparsity_preserving_csr_matrix<ElementType,
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
  using device_sparsity_owning_coo_matrix =
    sparsity_owning_coo_matrix<ElementType, RowType, ColType, NZType, true, ContainerPolicy>

  template <typename ElementType, typename RowType, typename ColType, typename NZType>
  using device_coo_matrix_view = coo_matrix_view<ElementType, RowType, ColType, NZType, true>

  template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
  using device_csr_matrix_view = csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, true>

  template <typename ElementType,
            typename RowType,
            typename ColType,
            typename NZType,
            typename ContainerPolicy = detail::device_uvector_policy<ElementType>>
  using device_sparsity_preserving_coo_matrix =
    sparsity_preserving_coo_matrix<ElementType, RowType, ColType, NZType, true, ContainerPolicy>

  template <typename ElementType,
            typename RowType,
            typename ColType,
            typename NZType,
            typename ContainerPolicy = detail::device_uvector_policy<ElementType>>
  using device_coordinate_structure =
    coordinate_structure<ElementType, RowType, ColType, NZType, true, ContainerPolicy>

  template <typename ElementType,
            typename IndptrType,
            typename IndicesType,
            typename NZType,
            typename ContainerPolicy = detail::device_uvector_policy<ElementType>>
  using device_compressed_structure =
    compressed_structure<ElementType, IndptrType, IndicesType, NZType, true, ContainerPolicy>

  template <typename ElementType, typename RowType, typename ColType, typename NZType>
  using device_coordinate_structure_view =
    coordinate_structure_view<ElementType, RowType, ColType, NZType, true>

  template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
  using device_compressed_structure_view =
    compressed_structure_view<ElementType, IndptrType, IndicesType, NZType, true>

  /**
   * Create a sparsity-owning sparse matrix in the compressed-sparse row format. sparsity-owning
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
    device_sparsity_owning_csr_matrix<ElementType, IndptrType, IndicesType, NZType>;
  return csr_matrix_t(handle, n_rows, n_cols, nnz);
}

/**
 * Create a sparsity-preserving sparse matrix in the compressed-sparse row format.
 * sparsity-preserving means that a view of the csr sparsity is supplied, allowing the values in
 * the sparsity to change but not the sparsity itself. The csr_matrix instance does not own the
 * sparsity, the sparsity must be known up front, and cannot be resized later.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param handle
 * @param sparsity_
 * @return
 */
template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
auto make_csr_matrix(raft::device_resources const& handle,
                     compressed_sparsity_view<IndptrType, IndicesType, NZType> sparsity_)
{
  using csr_matrix_t =
    device_sparsity_preserving_csr_matrix<ElementType, IndptrType, IndicesType, NZType>;
  return csr_matrix_t(handle, std::make_shared(sparsity_));
}

/**
 * Create a sparsity-owning sparse matrix in the coordinate format. sparsity-owning means that
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
  using coo_matrix_t = device_sparsity_owning_coo_matrix<ElementType, RowType, ColType, NZType>;
  return coo_matrix_t(handle, n_rows, n_cols, nnz);
}

/**
 * Create a sparsity-preserving sparse matrix in the coordinate format. sparsity-preserving means
 * that a view of the coo sparsity is supplied, allowing the values in the sparsity to change but
 * not the sparsity itself. The csr_matrix instance does not own the sparsity, the sparsity must
 * be known up front, and cannot be resized later.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param handle
 * @param sparsity_
 * @return
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_coo_matrix(raft::device_resources const& handle,
                     coordinate_structure_view<RowType, ColType, NZType> sparsity_)
{
  using coo_matrix_t = device_sparsity_preserving_coo_matrix<ElementType, RowType, ColType, NZType>;
  return coo_matrix_t(handle, std::make_shared(sparsity_));
}

/**
 * Create a non-owning sparse matrix view in the coordinate format. This is sparsity-preserving,
 * meaning that the underlying sparsity is known and cannot be changed. Use the sparsity-owning
 * coo_matrix if sparsity needs to be mutable.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param handle
 * @param sparsity_
 * @return
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_coo_matrix_view(ElementType* ptr,
                          coordinate_structure_view<RowType, ColType, NZType> sparsity_)
{
  using coo_matrix_view_t = device_coo_matrix_view<ElementType, RowType, ColType, NZType>;
  return coo_matrix_t(, std::make_shared(sparsity_));
}

/**
 * Create a non-owning sparse matrix view in the coordinate format. This is sparsity-preserving,
 * meaning that the underlying sparsity is known and cannot be changed. Use the sparsity-owning
 * coo_matrix if sparsity needs to be mutable.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param handle
 * @param sparsity_
 * @return
 */
template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
auto make_csr_matrix_view(ElementType* ptr,
                          compressed_structure_view<IndptrType, IndicesType, NZType> structure_)
{
  using csr_matrix_view_t = device_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType>;
  return csr_matrix_t(raft::device_span(ptr, sparsity_.get_nnz()), std::make_shared(structure_));
}

/**
 * Create a sparsity-owning coordinate structure object. If not known up front, this object can be
 * resized() once the sparsity (number of non-zeros) is known, postponing the allocation of the
 * underlying data arrays.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param handle
 * @param sparsity_
 * @return
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_coordinate_structure(RowType n_rows, ColType n_cols, NZType nnz = 0)
{
  using coordinate_structure_t = device_coordinate_structure<ElementType, RowType, ColType, NZType>;
  return coordinate_structure_t(n_rows, n_cols, nnz);
}

/**
 * Create a non-owning sparsity-preserved coordinate structure view. Sparsity-preserving means that
 * the underlying sparsity is known and cannot be changed. Use the sparsity-owning version if the
 * sparsity is not known up front.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param handle
 * @param sparsity_
 * @return
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_coordinate_structure_view(
  RowType* rows, ColType* cols, RowType n_rows, ColType n_cols, NZType nnz = 0)
{
  using coordinate_structure_view_t =
    device_coordinate_structure_view<ElementType, RowType, ColType, NZType>;
  return coordinate_structure_view_t(
    raft::device_span(rows, nnz), raft::device_span(cols, nnz), n_rows, n_cols);
}

/**
 * Create a sparsity-owning compressed structure. This is not sparsity-preserving, meaning that
 * the underlying sparsity does not need to be known upon construction. When not known up front,
 * the allocation of the underlying indices array is delayed until `resize(nnz)` is invoked.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param handle
 * @param sparsity_
 * @return
 */
template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
auto make_compressed_structure(IndptrType n_rows, IndicesType n_cols, NZType nnz = 0)
{
  using compressed_structure_t =
    device_compressed_structure<ElementType, IndptrType, IndicesType, NZType>;
    return compressed_structure_t(n_rows, n_cols, nnz));
}

/**
 * Create a non-owning sparsity-preserved compressed structure view. Sparsity-preserving means that
 * the underlying sparsity is known and cannot be changed. Use the sparsity-owning version if the
 * sparsity is not known up front.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param handle
 * @param sparsity_
 * @return
 */
template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
auto make_compressed_structure_view(IndptrType* indptr, IndicesType* indices, NZType nnz)
{
  using compressed_structure_t =
    device_compressed_structure_view<ElementType, IndptrType, IndicesType, NZType>;
    return compressed_structure_t(raft::device_span(indptr, n_rows+1), raft::device_span(indices, nnz), n_cols));
}

};  // namespace raft