/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include <raft/spatial/knn/detail/faiss_select/DeviceDefs.cuh>
#include <raft/spatial/knn/detail/faiss_select/PtxUtils.cuh>
#include <raft/spatial/knn/detail/faiss_select/ReductionOperators.cuh>
#include <raft/spatial/knn/detail/faiss_select/StaticUtils.h>
#include <raft/spatial/knn/detail/faiss_select/WarpShuffles.cuh>

namespace raft::spatial::knn::detail::faiss_select {

template <typename T, typename Op, int ReduceWidth = kWarpSize>
__device__ inline T warpReduceAll(T val, Op op)
{
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val = op(val, shfl_xor(val, mask));
  }

  return val;
}
}  // namespace raft::spatial::knn::detail::faiss_select
