/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/vectorized.cuh>

namespace raft {
namespace linalg {

template <typename InType, typename OutType, typename MapOp, int TPB, typename... Args>
__global__ void mapKernel(OutType* out, size_t len, MapOp map, const InType* in, Args... args)
{
  auto idx = (threadIdx.x + (blockIdx.x * blockDim.x));

  if (idx < len) { out[idx] = map(in[idx], args[idx]...); }
}

template <typename InType, typename OutType, typename MapOp, int TPB, typename... Args>
void mapImpl(
  OutType* out, size_t len, MapOp map, cudaStream_t stream, const InType* in, Args... args)
{
  const int nblks = raft::ceildiv(len, (size_t)TPB);
  mapKernel<InType, OutType, MapOp, TPB, Args...>
    <<<nblks, TPB, 0, stream>>>(out, len, map, in, args...);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief CUDA version of map
 * @tparam InType data-type upon which the math operation will be performed
 * @tparam MapOp the device-lambda performing the actual operation
 * @tparam TPB threads-per-block in the final kernel launched
 * @tparam Args additional parameters
 * @tparam OutType data-type in which the result will be stored
 * @param out the output of the map operation (assumed to be a device pointer)
 * @param len number of elements in the input array
 * @param map the device-lambda
 * @param stream cuda-stream where to launch this kernel
 * @param in the input array
 * @param args additional input arrays
 */

template <typename InType,
          typename MapOp,
          int TPB = 256,
          typename... Args,
          typename OutType = InType>
void map(OutType* out, size_t len, MapOp map, cudaStream_t stream, const InType* in, Args... args)
{
  mapImpl<InType, OutType, MapOp, TPB, Args...>(out, len, map, stream, in, args...);
}

}  // namespace linalg
};  // namespace raft
