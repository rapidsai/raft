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

struct sum_tag {};

template <typename Type, int TPB>
__device__ void reduce(Type *out, const Type acc, sum_tag) {
  typedef cub::BlockReduce<Type, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  Type tmp = BlockReduce(temp_storage).Sum(acc);
  if (threadIdx.x == 0) {
    raft::myAtomicAdd(out, tmp);
  }
}

template <typename Type, int TPB, typename ReduceLambda>
__device__ void reduce(Type *out, const Type acc, ReduceLambda op) {
  typedef cub::BlockReduce<Type, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  Type tmp = BlockReduce(temp_storage).Reduce(acc, op);
  if (threadIdx.x == 0) {
    raft::myAtomicReduce(out, tmp, op);
  }
}

template <typename Type, typename MapOp, typename ReduceLambda, int TPB,
          typename... Args>
__global__ void mapThenReduceKernel(Type *out, size_t len, Type neutral,
                                    MapOp map, ReduceLambda op, const Type *in,
                                    Args... args) {
  Type acc = neutral;
  auto idx = (threadIdx.x + (blockIdx.x * blockDim.x));

  if (idx < len) {
    acc = map(in[idx], args[idx]...);
  }

  __syncthreads();

  reduce<Type, TPB>(out, acc, op);
}

template <typename Type, typename MapOp, typename ReduceLambda, int TPB,
          typename... Args>
void mapThenReduceImpl(Type *out, size_t len, Type neutral, MapOp map,
                       ReduceLambda op, cudaStream_t stream, const Type *in,
                       Args... args) {
  raft::update_device(out, &neutral, 1, stream);
  const int nblks = raft::ceildiv(len, (size_t)TPB);
  mapThenReduceKernel<Type, MapOp, ReduceLambda, TPB, Args...>
    <<<nblks, TPB, 0, stream>>>(out, len, neutral, map, op, in, args...);
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
void mapThenSumReduce(Type *out, size_t len, MapOp map, cudaStream_t stream,
                      const Type *in, Args... args) {
  mapThenReduceImpl<Type, MapOp, sum_tag, TPB, Args...>(
    out, len, (Type)0, map, sum_tag(), stream, in, args...);
}

/**
 * @brief CUDA version of map and then generic reduction operation
 * @tparam Type data-type upon which the math operation will be performed
 * @tparam MapOp the device-lambda performing the actual map operation
 * @tparam ReduceLambda the device-lambda performing the actual reduction
 * @tparam TPB threads-per-block in the final kernel launched
 * @tparam Args additional parameters
 * @param out the output reduced value (assumed to be a device pointer)
 * @param len number of elements in the input array
 * @param neutral The neutral element of the reduction operation. For example:
 *    0 for sum, 1 for multiply, +Inf for Min, -Inf for Max
 * @param map the device-lambda
 * @param op the reduction device lambda
 * @param stream cuda-stream where to launch this kernel
 * @param in the input array
 * @param args additional input arrays
 */

template <typename Type, typename MapOp, typename ReduceLambda, int TPB = 256,
          typename... Args>
void mapThenReduce(Type *out, size_t len, Type neutral, MapOp map,
                   ReduceLambda op, cudaStream_t stream, const Type *in,
                   Args... args) {
  mapThenReduceImpl<Type, MapOp, ReduceLambda, TPB, Args...>(
    out, len, neutral, map, op, stream, in, args...);
}
};  // end namespace linalg
};  // end namespace raft
