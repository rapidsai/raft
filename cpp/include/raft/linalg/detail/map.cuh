/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cub/cub.cuh>
#include <raft/core/device_resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/vectorized.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <typename InType,
          typename OutType,
          typename IdxType,
          typename MapOp,
          int TPB,
          typename... Args>
__global__ void mapKernel(OutType* out, IdxType len, MapOp map, const InType* in, Args... args)
{
  auto idx = (threadIdx.x + (blockIdx.x * blockDim.x));

  if (idx < len) { out[idx] = map(in[idx], args[idx]...); }
}

template <typename InType,
          typename OutType,
          typename IdxType,
          typename MapOp,
          int TPB,
          typename... Args>
void mapImpl(
  OutType* out, IdxType len, MapOp map, cudaStream_t stream, const InType* in, Args... args)
{
  const int nblks = raft::ceildiv(len, (IdxType)TPB);
  mapKernel<InType, OutType, IdxType, MapOp, TPB, Args...>
    <<<nblks, TPB, 0, stream>>>(out, len, map, in, args...);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace detail
}  // namespace linalg
};  // namespace raft
