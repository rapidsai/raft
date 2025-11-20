/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/init.cuh>
#include <raft/sparse/matrix/detail/diagonal.cuh>
#include <raft/util/input_validation.hpp>

namespace raft::sparse::matrix {

/**
 * @brief Get the diagonal vector from a CSR matrix
 *
 * This function extracts the diagonal elements from a CSR matrix and stores them in a vector.
 * The diagonal elements are the elements where the row index and column index are the same.
 *
 *
 * @tparam T The data type of the matrix elements
 * @tparam IndexType The data type of the indices
 *
 * @param[in]  res             RAFT resources for managing device memory and streams
 * @param[in]  csr_matrix_view View of the input CSR matrix from which to extract the diagonal
 * @param[out] diagonal        View of the output vector where diagonal elements will be stored
 *
 */
template <typename T, typename IndptrType, typename IndexType, typename NNZType>
void diagonal(raft::resources const& res,
              raft::device_csr_matrix_view<T, IndptrType, IndexType, NNZType> csr_matrix_view,
              raft::device_vector_view<T, IndexType> diagonal)
{
  detail::diagonal(res, csr_matrix_view, diagonal);
}

/**
 * @brief Scale a CSR matrix by its diagonal elements
 *
 * This function scales each element of the CSR matrix by the corresponding diagonal element.
 * The diagonal elements are assumed to be stored in the diagonal vector.
 *
 *
 * @tparam T The data type of the matrix elements
 * @tparam IndexType The data type of the indices
 *
 * @param[in]  res             RAFT resources for managing device memory and streams
 * @param[in]  csr_matrix_view View of the input CSR matrix to scale
 * @param[in]  diagonal        View of the input vector containing diagonal elements
 *
 *
 */
template <typename T, typename IndptrType, typename IndexType, typename NNZType>
void scale_by_diagonal_symmetric(
  raft::resources const& res,
  const raft::device_vector_view<T, IndexType> diagonal,
  raft::device_csr_matrix_view<T, IndptrType, IndexType, NNZType> csr_matrix_view)
{
  detail::scale_by_diagonal_symmetric(res, diagonal, csr_matrix_view);
}

/**
 * @brief Set the diagonal elements of a CSR matrix to a scalar value
 *
 * This function sets the diagonal elements of a CSR matrix to a scalar value.
 * The diagonal elements are the elements where the row index and column index are the same.
 *
 *
 * @tparam T The data type of the matrix elements
 * @tparam IndexType The data type of the indices
 *
 * @param[in]  res             RAFT resources for managing device memory and streams
 * @param[in]  csr_matrix_view View of the input CSR matrix to modify
 * @param[in]  scalar          The scalar value to set the diagonal elements to
 *
 */
template <typename T, typename IndptrType, typename IndexType, typename NNZType>
void set_diagonal(raft::resources const& res,
                  raft::device_csr_matrix_view<T, IndptrType, IndexType, NNZType> csr_matrix_view,
                  T scalar)
{
  detail::set_diagonal(res, csr_matrix_view, scalar);
}

/**
 * @brief Get the diagonal vector from a COO matrix
 *
 * This function extracts the diagonal elements from a COO matrix and stores them in a vector.
 * The diagonal elements are the elements where the row index and column index are the same.
 *
 * @tparam T The data type of the matrix elements
 * @tparam RowType The data type of the row indices
 * @tparam ColType The data type of the column indices
 * @tparam NNZType The data type for representing nonzero counts
 *
 * @param[in]  res             RAFT resources for managing device memory and streams
 * @param[in]  coo_matrix_view View of the input COO matrix from which to extract the diagonal
 * @param[out] diagonal        View of the output vector where diagonal elements will be stored
 */
template <typename T, typename RowType, typename ColType, typename NNZType>
void diagonal(raft::resources const& res,
              raft::device_coo_matrix_view<T, RowType, ColType, NNZType> coo_matrix_view,
              raft::device_vector_view<T, RowType> diagonal)
{
  detail::diagonal(res, coo_matrix_view, diagonal);
}

/**
 * @brief Scale a COO matrix by its diagonal elements
 *
 * This function scales each element of the COO matrix by the corresponding diagonal element.
 * The diagonal elements are assumed to be stored in the diagonal vector.
 *
 * @tparam T The data type of the matrix elements
 * @tparam RowType The data type of the row indices
 * @tparam ColType The data type of the column indices
 * @tparam NNZType The data type for representing nonzero counts
 *
 * @param[in]  res             RAFT resources for managing device memory and streams
 * @param[in]  diagonal        View of the input vector containing diagonal elements
 * @param[in]  coo_matrix_view View of the input COO matrix to scale
 */
template <typename T, typename RowType, typename ColType, typename NNZType>
void scale_by_diagonal_symmetric(
  raft::resources const& res,
  const raft::device_vector_view<T, RowType> diagonal,
  raft::device_coo_matrix_view<T, RowType, ColType, NNZType> coo_matrix_view)
{
  detail::scale_by_diagonal_symmetric(res, diagonal, coo_matrix_view);
}

/**
 * @brief Set the diagonal elements of a COO matrix to a scalar value
 *
 * This function sets the diagonal elements of a COO matrix to a scalar value.
 * The diagonal elements are the elements where the row index and column index are the same.
 *
 * @tparam T The data type of the matrix elements
 * @tparam RowType The data type of the row indices
 * @tparam ColType The data type of the column indices
 * @tparam NNZType The data type for representing nonzero counts
 *
 * @param[in]  res             RAFT resources for managing device memory and streams
 * @param[in]  coo_matrix_view View of the input COO matrix to modify
 * @param[in]  scalar          The scalar value to set the diagonal elements to
 */
template <typename T, typename RowType, typename ColType, typename NNZType>
void set_diagonal(raft::resources const& res,
                  raft::device_coo_matrix_view<T, RowType, ColType, NNZType> coo_matrix_view,
                  T scalar)
{
  detail::set_diagonal(res, coo_matrix_view, scalar);
}

}  // namespace raft::sparse::matrix
