/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

#ifndef __MAP_THEN_REDUCE_H
#define __MAP_THEN_REDUCE_H

#pragma once

#include "detail/map_then_reduce.cuh"

namespace raft {
namespace linalg {

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

template <typename InType,
          typename MapOp,
          int TPB = 256,
          typename... Args,
          typename OutType = InType>
void mapThenSumReduce(
  OutType* out, size_t len, MapOp map, cudaStream_t stream, const InType* in, Args... args)
{
  detail::mapThenReduceImpl<InType, OutType, MapOp, detail::sum_tag, TPB, Args...>(
    out, len, (OutType)0, map, detail::sum_tag(), stream, in, args...);
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

template <typename InType,
          typename MapOp,
          typename ReduceLambda,
          int TPB          = 256,
          typename OutType = InType,
          typename... Args>
void mapThenReduce(OutType* out,
                   size_t len,
                   OutType neutral,
                   MapOp map,
                   ReduceLambda op,
                   cudaStream_t stream,
                   const InType* in,
                   Args... args)
{
  detail::mapThenReduceImpl<InType, OutType, MapOp, ReduceLambda, TPB, Args...>(
    out, len, neutral, map, op, stream, in, args...);
}
};  // end namespace linalg
};  // end namespace raft

#endif