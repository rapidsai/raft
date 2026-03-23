/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace stats {
namespace detail {

template <bool rowMajor, typename Type, typename IdxType = int>
void sum(Type* output, const Type* input, IdxType D, IdxType N, cudaStream_t stream)
{
  raft::linalg::reduce<rowMajor, false>(output, input, D, N, Type(0), stream);
}

}  // namespace detail
}  // namespace stats
}  // namespace raft
