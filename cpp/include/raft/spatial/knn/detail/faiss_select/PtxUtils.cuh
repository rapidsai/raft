/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>

namespace raft::spatial::knn::detail::faiss_select {
__device__ __forceinline__ int getLaneId()
{
  int laneId;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
  return laneId;
}
}  // namespace raft::spatial::knn::detail::faiss_select
