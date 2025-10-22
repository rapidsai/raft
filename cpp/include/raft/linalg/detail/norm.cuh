/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/operators.hpp>
#include <raft/linalg/norm_types.hpp>
#include <raft/linalg/reduce.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <NormType norm_type,
          bool rowMajor,
          typename Type,
          typename IdxType,
          typename Lambda,
          typename OutType = Type>
void rowNormCaller(
  OutType* dots, const Type* data, IdxType D, IdxType N, cudaStream_t stream, Lambda fin_op)
{
  if constexpr (norm_type == L1Norm) {
    raft::linalg::reduce<rowMajor, true, Type, OutType, IdxType>(
      dots, data, D, N, (OutType)0, stream, false, raft::abs_op(), raft::add_op(), fin_op);
  } else if constexpr (norm_type == L2Norm) {
    raft::linalg::reduce<rowMajor, true, Type, OutType, IdxType>(
      dots, data, D, N, (OutType)0, stream, false, raft::sq_op(), raft::add_op(), fin_op);
  } else if constexpr (norm_type == LinfNorm) {
    raft::linalg::reduce<rowMajor, true, Type, OutType, IdxType>(
      dots, data, D, N, (OutType)0, stream, false, raft::abs_op(), raft::max_op(), fin_op);
  } else {
    THROW("Unsupported norm type: %d", norm_type);
  }
}

template <NormType norm_type,
          bool rowMajor,
          typename Type,
          typename IdxType,
          typename Lambda,
          typename OutType = Type>
void colNormCaller(
  OutType* dots, const Type* data, IdxType D, IdxType N, cudaStream_t stream, Lambda fin_op)
{
  if constexpr (norm_type == L1Norm) {
    raft::linalg::reduce<rowMajor, false, Type, OutType, IdxType>(
      dots, data, D, N, (OutType)0, stream, false, raft::abs_op(), raft::add_op(), fin_op);
  } else if constexpr (norm_type == L2Norm) {
    raft::linalg::reduce<rowMajor, false, Type, OutType, IdxType>(
      dots, data, D, N, (OutType)0, stream, false, raft::sq_op(), raft::add_op(), fin_op);
  } else if constexpr (norm_type == LinfNorm) {
    raft::linalg::reduce<rowMajor, false, Type, OutType, IdxType>(
      dots, data, D, N, (OutType)0, stream, false, raft::abs_op(), raft::max_op(), fin_op);
  } else {
    THROW("Unsupported norm type: %d", norm_type);
  }
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
