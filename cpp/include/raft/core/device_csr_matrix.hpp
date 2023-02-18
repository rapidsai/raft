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

#include <raft/core/csr_matrix.hpp>
#include <raft/core/detail/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_span.hpp>
#include <raft/core/sparse_types.hpp>
#include <type_traits>

namespace raft {

template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy = detail::device_uvector_policy,
          SparsityType type_enum                         = SparsityType::OWNING>
using device_csr_matrix =
  csr_matrix<ElementType, IndptrType, IndicesType, NZType, true, ContainerPolicy, type_enum>;

/**
 * Specialization for a sparsity-owning csr matrix which uses device memory
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy = detail::device_uvector_policy>
using device_sparsity_owning_csr_matrix =
  csr_matrix<ElementType, IndptrType, IndicesType, NZType, true, ContainerPolicy>;

template <typename T>
struct is_device_csr_matrix : std::false_type {
};

template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T>
          typename ContainerPolicy,
          SparsityType type_enum>
struct is_device_csr_matrix<
  device_csr_matrix<ElementType, IndptrType, IndicesType, NZType, ContainerPolicy, type_enum>>
  : std::true_type {
};

template <typename T>
constexpr bool is_device_csr_matrix_v = is_device_csr_matrix<T>::value;

template <typename T>
constexpr bool is_device_csr_sparsity_owning_v =
  is_device_csr_matrix<T>::value and T::get_type_enum() == OWNING;

template <typename T>
constexpr bool is_device_csr_sparsity_preserving_v =
  is_device_csr_matrix<T>::value and T::get_type_enum() == PRESERVING;

/**
 * Specialization for a csr matrix view which uses device memory
 */
template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
using device_csr_matrix_view = csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, true>;

/**
 * Specialization for a sparsity-preserving csr matrix which uses device memory
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy = detail::device_uvector_policy>
using device_sparsity_preserving_csr_matrix = csr_matrix<ElementType,
                                                         IndptrType,
                                                         IndicesType,
                                                         NZType,
                                                         true,
                                                         ContainerPolicy,
                                                         SparsityType::PRESERVING>;

/**
 * Specialization for a csr matrix view which uses device memory
 */
template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
using device_csr_matrix_view = csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, true>;

/**
 * Specialization for a sparsity-owning compressed structure which uses device memory
 */
template <typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy = detail::device_uvector_policy>
using device_compressed_structure =
  compressed_structure<IndptrType, IndicesType, NZType, true, ContainerPolicy>;

/**
 * Specialization for a sparsity-preserving compressed structure view which uses device memory
 */
template <typename IndptrType, typename IndicesType, typename NZType>
using device_compressed_structure_view =
  compressed_structure_view<IndptrType, IndicesType, NZType, true>;

/**
 * Create a sparsity-owning sparse matrix in the compressed-sparse row format. sparsity-owning
 * means that all of the underlying vectors (data, indptr, indices) are owned by the csr_matrix
 * instance. If not known up front, the sparsity can be ignored in this factory function and
 * `resize()` invoked on the instance once the sparsity is known.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] handle a raft handle for managing expensive device resources
 * @param[in] n_rows total number of rows in the matrix
 * @param[in] n_cols total number of columns in the matrix
 * @param[in] nnz number of non-zeros in the matrix if known [optional]
 * @return a sparsity-owning sparse matrix in compressed (csr) format
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType = uint64_t>
auto make_device_csr_matrix(raft::device_resources const& handle,
                            IndptrType n_rows,
                            IndicesType n_cols,
                            NZType nnz = 0)
{
  return device_sparsity_owning_csr_matrix<ElementType, IndptrType, IndicesType, NZType>(
    handle, n_rows, n_cols, nnz);
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
 * @param[in] handle raft handle for managing expensive device resources
 * @param[in] structure_ a sparsity-preserving compressed structural view
 * @return a sparsity-preserving sparse matrix in compressed (csr) format
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType = uint64_t>
auto make_device_csr_matrix(
  raft::device_resources const& handle,
  device_compressed_structure_view<IndptrType, IndicesType, NZType> structure_)
{
  return device_sparsity_preserving_csr_matrix<ElementType, IndptrType, IndicesType, NZType>(
    handle,
    std::make_shared<device_compressed_structure_view<IndptrType, IndicesType, NZType>>(
      structure_));
}

/**
 * Create a non-owning sparse matrix view in the coordinate format. This is sparsity-preserving,
 * meaning that the underlying sparsity is known and cannot be changed. Use the sparsity-owning
 * coo_matrix if sparsity needs to be mutable.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] ptr a pointer to array of nonzero matrix elements on device (size nnz)
 * @param[in] structure_ a sparsity-preserving compressed sparse structural view
 * @return a sparsity-preserving csr matrix view
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType = uint64_t>
auto make_device_csr_matrix_view(
  ElementType* ptr, device_compressed_structure_view<IndptrType, IndicesType, NZType> structure_)
{
  return device_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType>(
    raft::device_span<ElementType>(ptr, structure_.get_nnz()), std::make_shared(structure_));
}

/**
 * Create a non-owning sparse matrix view in the compressed-sparse row format. This is
 * sparsity-preserving, meaning that the underlying sparsity is known and cannot be changed. Use the
 * sparsity-owning coo_matrix if sparsity needs to be mutable.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] elements device span containing array of matrix elements (size nnz)
 * @param[in] structure_ a sparsity-preserving structural view
 * @return a sparsity-preserving csr matrix view
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType = uint64_t>
auto make_device_csr_matrix_view(
  raft::device_span<ElementType> elements,
  device_compressed_structure_view<IndptrType, IndicesType, NZType> structure_)
{
  RAFT_EXPECTS(elements.size() == structure_.get_nnz(),
               "Size of elements must be equal to the nnz from the structure");
  return device_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType>(
    elements, std::make_shared(structure_));
}

/**
 * Create a sparsity-owning compressed structure. This is not sparsity-preserving, meaning that
 * the underlying sparsity does not need to be known upon construction. When not known up front,
 * the allocation of the underlying indices array is delayed until `resize(nnz)` is invoked.
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] handle raft handle for managing expensive device resources
 * @param[in] n_rows total number of rows
 * @param[in] n_cols total number of cols
 * @param[in] nnz total number of nonzeros, if known
 * @return a sparsity-owning compressed structure instance
 */
template <typename IndptrType, typename IndicesType, typename NZType = uint64_t>
auto make_compressed_structure(raft::device_resources const& handle,
                               IndptrType n_rows,
                               IndicesType n_cols,
                               NZType nnz = 0)
{
  return device_compressed_structure<IndptrType, IndicesType, NZType>(handle, n_rows, n_cols, nnz);
}

/**
 * Create a non-owning sparsity-preserved compressed structure view. Sparsity-preserving means that
 * the underlying sparsity is known and cannot be changed. Use the sparsity-owning version if the
 * sparsity is not known up front.
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] indptr structural indptr (size n_rows+1)
 * @param[in] indices structural indices (size nnz)
 * @param[in] n_rows total number of rows
 * @param[in] n_cols total number of columns
 * @param[in] nnz number of non-zeros
 * @return a sparsity-preserving compressed structural view
 */
template <typename IndptrType, typename IndicesType, typename NZType = uint64_t>
auto make_device_csr_structure_view(
  IndptrType* indptr, IndicesType* indices, IndptrType n_rows, IndicesType n_cols, NZType nnz)
{
  return device_compressed_structure_view<IndptrType, IndicesType, NZType>(
    raft::device_span<IndptrType>(indptr, n_rows + 1),
    raft::device_span<IndicesType>(indices, nnz),
    n_cols);
}

/**
 * Create a non-owning sparsity-preserved compressed structure view. Sparsity-preserving means that
 * the underlying sparsity is known and cannot be changed. Use the sparsity-owning version if the
 * sparsity is not known up front.
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] indptr structural indptr (size n_rows+1)
 * @param[in] indices structural indices (size nnz)
 * @param[in] n_cols total number of columns
 * @return a sparsity-preserving compressed structural view
 *
 */
template <typename IndptrType, typename IndicesType, typename NZType = uint64_t>
auto make_device_csr_structure_view(raft::device_span<IndptrType> indptr,
                                    raft::device_span<IndicesType> indices,
                                    IndicesType n_cols)
{
  return device_compressed_structure_view<IndptrType, IndicesType, NZType>(indptr, indices, n_cols);
}

};  // namespace raft