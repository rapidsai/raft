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
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_span.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/sparse_types.hpp>
#include <type_traits>

namespace raft {

/**
 * \defgroup device_csr_matrix Device CSR Matrix Types
 * @{
 */

/**
 * Specialization for a sparsity-preserving compressed structure view which uses device memory
 */
template <typename IndptrType, typename IndicesType, typename NZType>
using device_compressed_structure_view =
  compressed_structure_view<IndptrType, IndicesType, NZType, true>;

/**
 * Specialization for a sparsity-owning compressed structure which uses device memory
 */
template <typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_uvector_policy>
using device_compressed_structure =
  compressed_structure<IndptrType, IndicesType, NZType, true, ContainerPolicy>;

/**
 * Specialization for a csr matrix view which uses device memory
 */
template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
using device_csr_matrix_view = csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, true>;

template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_uvector_policy,
          SparsityType sparsity_type                     = SparsityType::OWNING>
using device_csr_matrix =
  csr_matrix<ElementType, IndptrType, IndicesType, NZType, true, ContainerPolicy, sparsity_type>;

/**
 * Specialization for a sparsity-owning csr matrix which uses device memory
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_uvector_policy>
using device_sparsity_owning_csr_matrix =
  csr_matrix<ElementType, IndptrType, IndicesType, NZType, true, ContainerPolicy>;

/**
 * Specialization for a sparsity-preserving csr matrix which uses device memory
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_uvector_policy>
using device_sparsity_preserving_csr_matrix = csr_matrix<ElementType,
                                                         IndptrType,
                                                         IndicesType,
                                                         NZType,
                                                         true,
                                                         ContainerPolicy,
                                                         SparsityType::PRESERVING>;

template <typename T>
struct is_device_csr_matrix_view : std::false_type {};

template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
struct is_device_csr_matrix_view<
  device_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType>> : std::true_type {};

template <typename T>
constexpr bool is_device_csr_matrix_view_v = is_device_csr_matrix_view<T>::value;

template <typename T>
struct is_device_csr_matrix : std::false_type {};

template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T>
          typename ContainerPolicy,
          SparsityType sparsity_type>
struct is_device_csr_matrix<
  device_csr_matrix<ElementType, IndptrType, IndicesType, NZType, ContainerPolicy, sparsity_type>>
  : std::true_type {};

template <typename T>
constexpr bool is_device_csr_matrix_v = is_device_csr_matrix<T>::value;

template <typename T>
constexpr bool is_device_csr_sparsity_owning_v =
  is_device_csr_matrix<T>::value and T::get_sparsity_type() == OWNING;

template <typename T>
constexpr bool is_device_csr_sparsity_preserving_v =
  is_device_csr_matrix<T>::value and T::get_sparsity_type() == PRESERVING;

/**
 * Create a sparsity-owning sparse matrix in the compressed-sparse row format. sparsity-owning
 * means that all of the underlying vectors (data, indptr, indices) are owned by the csr_matrix
 * instance. If not known up front, the sparsity can be ignored in this factory function and
 * `resize()` invoked on the instance once the sparsity is known.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 *
 * raft::resources handle;
 * csr_matrix = raft::make_device_csr_matrix(handle, n_rows, n_cols);
 * ...
 * // compute expected sparsity
 * ...
 * int nnz = 5000;
 * csr_matrix.initialize_sparsity(nnz);
 * @endcode

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
auto make_device_csr_matrix(raft::resources const& handle,
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
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 *
 * raft::resources handle;
 * coo_structure = raft::make_device_compressed_structure(handle, n_rows, n_cols);
 * ...
 * // compute expected sparsity
 * ...
 * csr_structure.initialize_sparsity(nnz);
 * csr_matrix = raft::make_device_csr_matrix(handle, csr_structure.view());
 * @endcode
 *
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] handle raft handle for managing expensive device resources
 * @param[in] structure a sparsity-preserving compressed structural view
 * @return a sparsity-preserving sparse matrix in compressed (csr) format
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType = uint64_t>
auto make_device_csr_matrix(
  raft::resources const& handle,
  device_compressed_structure_view<IndptrType, IndicesType, NZType> structure)
{
  return device_sparsity_preserving_csr_matrix<ElementType, IndptrType, IndicesType, NZType>(
    handle, structure);
}

/**
 * Create a non-owning sparse matrix view in the coordinate format. This is sparsity-preserving,
 * meaning that the underlying sparsity is known and cannot be changed. Use the sparsity-owning
 * coo_matrix if sparsity needs to be mutable.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following pointer is assumed to reference device memory for a size of nnz
 * float* d_elm_ptr = ...;
 *
 * raft::resources handle;
 * csr_structure = raft::make_device_compressed_structure(handle, n_rows, n_cols, nnz);
 * csr_matrix_view = raft::make_device_csr_matrix_view(handle, d_elm_ptr, csr_structure.view());
 * @endcode
 *
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] ptr a pointer to array of nonzero matrix elements on device (size nnz)
 * @param[in] structure a sparsity-preserving compressed sparse structural view
 * @return a sparsity-preserving csr matrix view
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType = uint64_t>
auto make_device_csr_matrix_view(
  ElementType* ptr, device_compressed_structure_view<IndptrType, IndicesType, NZType> structure)
{
  return device_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType>(
    raft::device_span<ElementType>(ptr, structure.get_nnz()), structure);
}

/**
 * Create a non-owning sparse matrix view in the compressed-sparse row format. This is
 * sparsity-preserving, meaning that the underlying sparsity is known and cannot be changed. Use the
 * sparsity-owning coo_matrix if sparsity needs to be mutable.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_span.hpp>
 * #include <raft/core/device_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following span is assumed to be of size nnz
 * raft::device_span<float> d_elm_ptr;
 *
 * raft::resources handle;
 * csr_structure = raft::make_device_compressed_structure(handle, n_rows, n_cols, nnz);
 * csr_matrix_view = raft::make_device_csr_matrix_view(handle, d_elm_ptr, csr_structure.view());
 * @endcode
 *
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] elements device span containing array of matrix elements (size nnz)
 * @param[in] structure a sparsity-preserving structural view
 * @return a sparsity-preserving csr matrix view
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType = uint64_t>
auto make_device_csr_matrix_view(
  raft::device_span<ElementType> elements,
  device_compressed_structure_view<IndptrType, IndicesType, NZType> structure)
{
  RAFT_EXPECTS(elements.size() == structure.get_nnz(),
               "Size of elements must be equal to the nnz from the structure");
  return device_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType>(elements, structure);
}

/**
 * Create a sparsity-owning compressed structure. This is not sparsity-preserving, meaning that
 * the underlying sparsity does not need to be known upon construction. When not known up front,
 * the allocation of the underlying indices array is delayed until `resize(nnz)` is invoked.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * raft::resources handle;
 * csr_structure = raft::make_device_compressed_structure(handle, n_rows, n_cols, nnz);
 * ...
 * // compute expected sparsity
 * ...
 * csr_structure.initialize_sparsity(nnz);
 * @endcode
 *
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
auto make_device_compressed_structure(raft::resources const& handle,
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
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following pointer is assumed to reference device memory of size n_rows+1
 * int *indptr = ...;
 *
 * // The following pointer is assumed to reference device memory of size nnz
 * int *indices = ...;
 *
 * raft::resources handle;
 * csr_structure = raft::make_device_compressed_structure_view(handle, indptr, indices, n_rows,
 * n_cols, nnz);
 * @endcode *
 *
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
auto make_device_compressed_structure_view(
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
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following device spans is assumed to be of size n_rows+1
 * raft::device_span<int> indptr;
 *
 * // The following device span is assumed to be of size nnz
 * raft::device_span<int> indices;
 *
 * raft::resources handle;
 * csr_structure = raft::make_device_compressed_structure_view(handle, indptr, indices, n_rows,
 * n_cols);
 * @endcode
 *
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
auto make_device_compressed_structure_view(raft::device_span<IndptrType> indptr,
                                           raft::device_span<IndicesType> indices,
                                           IndicesType n_cols)
{
  return device_compressed_structure_view<IndptrType, IndicesType, NZType>(indptr, indices, n_cols);
}

/** @} */

};  // namespace raft