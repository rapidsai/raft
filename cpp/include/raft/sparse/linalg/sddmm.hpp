/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/sparse/linalg/detail/cusparse_utils.hpp>
#include <raft/sparse/linalg/detail/sddmm.hpp>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace sparse {
namespace linalg {

/**
 * @brief This function performs the multiplication of dense matrix A and dense matrix B,
 * followed by an element-wise multiplication with the sparsity pattern of C.
 * It computes the following equation: C = alpha · (opA(A) * opB(B) ∘ spy(C)) + beta · C
 * where A,B are device matrix views and C is a CSR device matrix view
 * @tparam ValueType Data type of input/output matrices (float/double)
 * @tparam IndexType Type of C
 * @tparam NZType Type of C
 * @tparam LayoutPolicyA layout of A
 * @tparam LayoutPolicyB layout of B
 * @param[in] handle raft handle
 * @param[in] A input raft::device_matrix_view
 * @param[in] B input raft::device_matrix_view
 * @param[inout] C output raft::device_csr_matrix_view
 * @param[in] opA input Operation op(A)
 * @param[in] opB input Operation op(B)
 * @param[in] alpha input raft::host_scalar_view
 * @param[in] beta input raft::host_scalar_view
 */
template <typename ValueType,
          typename IndexType,
          typename NZType,
          typename LayoutPolicyA,
          typename LayoutPolicyB>
void sddmm(raft::resources const& handle,
           raft::device_matrix_view<const ValueType, IndexType, LayoutPolicyA> A,
           raft::device_matrix_view<const ValueType, IndexType, LayoutPolicyB> B,
           raft::device_csr_matrix_view<ValueType, IndexType, IndexType, NZType> C,
           const raft::linalg::Operation opA,
           const raft::linalg::Operation opB,
           raft::host_scalar_view<ValueType> alpha,
           raft::host_scalar_view<ValueType> beta)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(A), "A is not contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(B), "B is not contiguous");

  static_assert(std::is_same_v<ValueType, float> || std::is_same_v<ValueType, double>,
                "The `ValueType` of sddmm only supports float/double.");

  auto descrA = detail::create_descriptor(A);
  auto descrB = detail::create_descriptor(B);
  auto descrC = detail::create_descriptor(C);
  auto op_A   = detail::convert_operation(opA);
  auto op_B   = detail::convert_operation(opB);

  detail::sddmm(
    handle, descrA, descrB, descrC, op_A, op_B, alpha.data_handle(), beta.data_handle());

  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(descrA));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(descrB));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(descrC));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // end namespace linalg
}  // end namespace sparse
}  // end namespace raft
