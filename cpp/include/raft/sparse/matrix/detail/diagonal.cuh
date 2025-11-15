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
#include <raft/util/input_validation.hpp>

namespace raft::sparse::matrix::detail {

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
  auto structure = csr_matrix_view.structure_view();
  auto n_rows    = structure.get_n_rows();

  auto values      = csr_matrix_view.get_elements().data();
  auto col_indices = structure.get_indices().data();
  auto row_offsets = structure.get_indptr().data();
  auto diag_ptr    = diagonal.data_handle();

  raft::matrix::fill(res, diagonal, T(0));

  raft::linalg::map_offset(res, diagonal, [=] __device__(auto idx) {
    for (auto j = row_offsets[idx]; j < row_offsets[idx + 1]; j++) {
      if (col_indices[j] == idx) {
        diag_ptr[idx] = values[j];
        break;
      }
    }
    return diag_ptr[idx];
  });
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
  raft::device_csr_matrix_view<T, IndptrType, IndexType, NNZType> csr_matrix)
{
  auto structure = csr_matrix.structure_view();
  auto nnz       = structure.get_nnz();

  auto values      = csr_matrix.get_elements().data();
  auto col_indices = structure.get_indices().data();
  auto row_offsets = structure.get_indptr().data();
  auto diag_ptr    = diagonal.data_handle();

  raft::linalg::map_offset(res, diagonal, [=] __device__(auto idx) {
    T row_scale = diag_ptr[idx] == 0 ? 0 : 1.0f / diag_ptr[idx];  // Scale factor for this row
    for (auto j = row_offsets[idx]; j < row_offsets[idx + 1]; j++) {
      IndexType col = col_indices[j];
      T col_scale   = diag_ptr[col] == 0 ? 0 : 1.0f / diag_ptr[col];  // Scale factor for the column
      values[j]     = row_scale * values[j] * col_scale;
    }
    return diag_ptr[idx];
  });
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
                  raft::device_csr_matrix_view<T, IndptrType, IndexType, NNZType> csr_matrix,
                  T scalar)
{
  auto structure = csr_matrix.structure_view();
  auto n_rows    = structure.get_n_rows();

  auto values      = csr_matrix.get_elements().data();
  auto col_indices = structure.get_indices().data();
  auto row_offsets = structure.get_indptr().data();

  raft::linalg::map_offset(
    res, make_device_vector_view(row_offsets, n_rows), [=] __device__(auto idx) {
      for (auto j = row_offsets[idx]; j < row_offsets[idx + 1]; j++) {
        if (col_indices[j] == idx) {
          values[j] = scalar;
          break;
        }
      }
      return row_offsets[idx];
    });
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
  auto structure = coo_matrix_view.structure_view();
  auto nnz       = structure.get_nnz();
  auto n_rows    = structure.get_n_rows();

  auto values   = coo_matrix_view.get_elements().data();
  auto rows     = structure.get_rows().data();
  auto cols     = structure.get_cols().data();
  auto diag_ptr = diagonal.data_handle();

  raft::matrix::fill(res, diagonal, T(0));

  auto values_view = raft::make_device_vector_view(values, nnz);

  raft::linalg::map_offset(res, values_view, [=] __device__(auto idx) {
    if (rows[idx] == cols[idx]) { diag_ptr[rows[idx]] = values[idx]; }
    return values[idx];
  });
}

/**
 * @brief Scale a COO matrix by its diagonal elements
 *
 * This function scales each element of the COO matrix by the corresponding diagonal element.
 *
 * @tparam T The data type of the matrix elements
 * @tparam RowType The data type of the row indices
 * @tparam ColType The data type of the column indices
 * @tparam NNZType The data type for representing nonzero counts
 *
 * @param[in]     res             RAFT resources for managing device memory and streams
 * @param[in]     diagonal        View of the input vector containing diagonal elements
 * @param[in,out] coo_matrix      View of the COO matrix to scale
 */
template <typename T, typename RowType, typename ColType, typename NNZType>
void scale_by_diagonal_symmetric(
  raft::resources const& res,
  const raft::device_vector_view<T, RowType> diagonal,
  raft::device_coo_matrix_view<T, RowType, ColType, NNZType> coo_matrix)
{
  auto structure = coo_matrix.structure_view();
  auto nnz       = structure.get_nnz();

  auto values   = coo_matrix.get_elements().data();
  auto rows     = structure.get_rows().data();
  auto cols     = structure.get_cols().data();
  auto diag_ptr = diagonal.data_handle();

  auto values_view = raft::make_device_vector_view(values, nnz);

  raft::linalg::map_offset(res, values_view, [=] __device__(auto idx) {
    auto row    = rows[idx];
    auto col    = cols[idx];
    T row_scale = diag_ptr[row] == 0 ? 0 : 1.0f / diag_ptr[row];  // Scale factor for this row
    T col_scale = diag_ptr[col] == 0 ? 0 : 1.0f / diag_ptr[col];  // Scale factor for the column

    return row_scale * values[idx] * col_scale;
  });
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
 * @param[in]     res             RAFT resources for managing device memory and streams
 * @param[in,out] coo_matrix      View of the COO matrix to modify
 * @param[in]     scalar          The scalar value to set the diagonal elements to
 */
template <typename T, typename RowType, typename ColType, typename NNZType>
void set_diagonal(raft::resources const& res,
                  raft::device_coo_matrix_view<T, RowType, ColType, NNZType> coo_matrix,
                  T scalar)
{
  auto structure = coo_matrix.structure_view();
  auto nnz       = structure.get_nnz();

  auto values = coo_matrix.get_elements().data();
  auto rows   = structure.get_rows().data();
  auto cols   = structure.get_cols().data();

  auto values_view = raft::make_device_vector_view(values, nnz);

  raft::linalg::map_offset(res, values_view, [=] __device__(auto idx) {
    if (rows[idx] == cols[idx]) { return scalar; }
    return values[idx];
  });
}

}  // namespace raft::sparse::matrix::detail
