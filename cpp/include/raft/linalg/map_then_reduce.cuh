/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

template <typename Type, int TPB>
__device__ void reduce(Type *out, const Type acc) {
  using block_reduce_t = cub::BlockReduce<Type, TPB>;
  __shared__ typename block_reduce_t::TempStorage temp_storage;  // NOLINT
  Type tmp = block_reduce_t(temp_storage).Sum(acc);
  if (threadIdx.x == 0) {
    raft::myAtomicAdd(out, tmp);
  }
}

template <typename Type, typename MapOp, int TPB, typename... Args>
__global__ void map_then_sum_reduce_kernel(Type *out, size_t len, MapOp map,
                                       const Type *in, Args... args) {
  auto acc = static_cast<Type>(0);
  auto idx = (threadIdx.x + (blockIdx.x * blockDim.x));
  if (idx < len) {
    acc = map(in[idx], args[idx]...);
  }
  __syncthreads();
  reduce<Type, TPB>(out, acc);
}

template <typename Type, typename MapOp, int TPB, typename... Args>
void map_then_sum_reduce_impl(Type *out, size_t len, MapOp map, cudaStream_t stream,
                          const Type *in, Args... args) {
  CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(Type), stream));
  const int nblks = raft::ceildiv(len, (size_t)TPB);
  map_then_sum_reduce_kernel<Type, MapOp, TPB, Args...>
    <<<nblks, TPB, 0, stream>>>(out, len, map, in, args...);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief CUDA version of map and then sum reduction operation
 * @tparam Type data-type upon which the math operation will be performed
 * @tparam MapOp the device-lambda performing the actual operation
 * @tparam TPB threads-per-block in the final kernel launched
 * @tparam Args additional parameters
 * @param out the output sum-reduced value (assumed to be a device pointer)
 * @param len number of elements in the input array
 * @param map the device-lambda
 * @param stream cuda-stream where to launch this kernel
 * @param in the input array
 * @param args additional input arrays
 */
template <typename Type, typename MapOp, int TPB = 256, typename... Args>
void mapThenSumReduce(Type *out, size_t len, MapOp map,  // NOLINT
                      cudaStream_t stream,
                      const Type *in, Args... args) {
  map_then_sum_reduce_impl<Type, MapOp, TPB, Args...>(out, len, map, stream, in,
                                                  args...);
}

};  // end namespace linalg
};  // end namespace raft
