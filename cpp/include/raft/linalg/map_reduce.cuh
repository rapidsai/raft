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
#ifndef __MAP_REDUCE_H
#define __MAP_REDUCE_H

#pragma once

#include "detail/map_then_reduce.cuh"

#include <raft/core/mdarray.hpp>

namespace raft::linalg {

/**
 * @defgroup map_reduce Map-Reduce ops
 * @{
 */

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
          typename IdxType = std::uint32_t,
          int TPB          = 256,
          typename OutType = InType,
          typename... Args>
void mapReduce(OutType* out,
               size_t len,
               OutType neutral,
               MapOp map,
               ReduceLambda op,
               cudaStream_t stream,
               const InType* in,
               Args... args)
{
  detail::mapThenReduceImpl<InType, OutType, IdxType, MapOp, ReduceLambda, TPB, Args...>(
    out, len, neutral, map, op, stream, in, args...);
}

/**
 * @brief CUDA version of map and then generic reduction operation
 * @tparam InValueType the data-type of the input
 * @tparam MapOp the device-lambda performing the actual map operation
 * @tparam ReduceLambda the device-lambda performing the actual reduction
 * @tparam IndexType the index type
 * @tparam OutValueType the data-type of the output
 * @tparam Args additional parameters
 * @param[in] handle raft::handle_t
 * @param[in] in the input of type raft::device_vector_view
 * @param[in] neutral The neutral element of the reduction operation. For example:
 *    0 for sum, 1 for multiply, +Inf for Min, -Inf for Max
 * @param[out] out the output reduced value assumed to be a raft::device_scalar_view
 * @param[in] map the fused device-lambda
 * @param[in] op the fused reduction device lambda
 * @param[in] args additional input arrays
 */
template <typename InValueType,
          typename MapOp,
          typename ReduceLambda,
          typename IndexType,
          typename OutValueType,
          typename... Args>
void map_reduce(const raft::handle_t& handle,
                raft::device_vector_view<const InValueType, IndexType> in,
                raft::device_scalar_view<OutValueType> out,
                OutValueType neutral,
                MapOp map,
                ReduceLambda op,
                Args... args)
{
  RAFT_EXPECTS(out.is_exhaustive(), "Output is not exhaustive");
  RAFT_EXPECTS(in.is_exhaustive(), "Input is not exhaustive");

  mapReduce<InValueType, MapOp, ReduceLambda, IndexType, 256, OutValueType, Args...>(
    out.data_handle(),
    in.extent(0),
    neutral,
    map,
    op,
    handle.get_stream(),
    in.data_handle(),
    args...);
}

/** @} */  // end of map_reduce

}  // end namespace raft::linalg

#endif