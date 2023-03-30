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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>

namespace raft {
namespace sparse {
namespace linalg {
namespace detail {

/**
 * @brief create a cuSparse dense descriptor
 * @tparam ValueType Data type of dense_view (float/double)
 * @tparam IndexType Type of dense_view
 * @tparam LayoutPolicy layout of dense_view
 * @param[in] handle raft handle
 * @param[in] dense_view input raft::device_matrix_view
 * @returns dense matrix descriptor to be used by cuSparse API
 */
template <typename ValueType, typename IndexType, typename LayoutPolicy>
cusparseDnMatDescr_t create_descriptor(
  raft::device_matrix_view<ValueType, IndexType, LayoutPolicy>& dense_view)
{
  ASSERT(dense_view.stride(0) == 1 || dense_view.stride(1) == 1, "Smallest stride needs to be 1");
  bool is_row_major = dense_view.stride(1) == 1;
  auto order        = is_row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
  IndexType ld      = is_row_major ? dense_view.stride(0) : dense_view.stride(1);
  cusparseDnMatDescr_t descr;
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
    &descr,
    dense_view.extent(0),
    dense_view.extent(1),
    ld,
    const_cast<std::remove_const_t<ValueType>*>(dense_view.data_handle()),
    order));
  return descr;
}

/**
 * @brief create a cuSparse sparse descriptor
 * @tparam ValueType Data type of sparse_view (float/double)
 * @tparam NZType Type of sparse_view
 * @param[in] handle raft handle
 * @param[in] sparse_view input raft::device_csr_matrix_view of size M rows x K columns
 * @returns sparse matrix descriptor to be used by cuSparse API
 */
template <typename ValueType, typename NZType>
cusparseSpMatDescr_t create_descriptor(
  raft::device_csr_matrix_view<ValueType, int, int, NZType>& sparse_view)
{
  cusparseSpMatDescr_t descr;
  auto csr_structure = sparse_view.get_structure();
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(
    &descr,
    csr_structure.get_n_rows(),
    csr_structure.get_n_cols(),
    csr_structure.get_nnz(),
    const_cast<int*>(csr_structure.get_indptr().data()),
    const_cast<int*>(csr_structure.get_indices().data()),
    const_cast<std::remove_const_t<ValueType>*>(sparse_view.get_elements().data())));
  return descr;
}

/**
 * @brief SPMM function designed for handling all CSR * DENSE
 * combinations of operand layouts for cuSparse.
 * It computes the following equation: Z = alpha . X * Y + beta . Z
 * where X is a CSR device matrix view and Y,Z are device matrix views
 * @tparam ValueType Data type of input/output matrices (float/double)
 * @tparam IndexType Type of Y and Z
 * @tparam NZType Type of X
 * @tparam LayoutPolicyY layout of Y
 * @tparam LayoutPolicyZ layout of Z
 * @param[in] handle raft handle
 * @param[in] trans_x transpose operation for X
 * @param[in] trans_y transpose operation for Y
 * @param[in] alpha scalar
 * @param[in] descr_x input sparse descriptor
 * @param[in] descr_y input dense descriptor
 * @param[in] beta scalar
 * @param[out] descr_z output dense descriptor
 */
template <typename ValueType>
void spmm(raft::device_resources const& handle,
          const bool trans_x,
          const bool trans_y,
          const ValueType* alpha,
          cusparseSpMatDescr_t& descr_x,
          cusparseDnMatDescr_t& descr_y,
          const ValueType* beta,
          cusparseDnMatDescr_t& descr_z)
{
  auto opX = trans_x ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto opY = trans_y ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto alg = CUSPARSE_SPMM_CSR_ALG1;
  size_t bufferSize;
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm_bufferSize(handle.get_cusparse_handle(),
                                                                  opX,
                                                                  opY,
                                                                  alpha,
                                                                  descr_x,
                                                                  descr_y,
                                                                  beta,
                                                                  descr_z,
                                                                  alg,
                                                                  &bufferSize,
                                                                  handle.get_stream()));

  raft::interruptible::synchronize(handle.get_stream());

  rmm::device_uvector<ValueType> tmp(bufferSize, handle.get_stream());

  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(handle.get_cusparse_handle(),
                                                       opX,
                                                       opY,
                                                       alpha,
                                                       descr_x,
                                                       descr_y,
                                                       beta,
                                                       descr_z,
                                                       alg,
                                                       tmp.data(),
                                                       handle.get_stream()));
}

}  // end namespace detail
}  // end namespace linalg
}  // end namespace sparse
}  // end namespace raft
