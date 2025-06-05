/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
