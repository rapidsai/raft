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
#ifndef __MAP_H
#define __MAP_H

#pragma once

#include "detail/map.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

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
          typename IdxType = std::uint32_t,
          int TPB          = 256,
          typename OutType = InType,
          typename... Args>
void map_k(
  OutType* out, IdxType len, MapOp map, cudaStream_t stream, const InType* in, Args... args)
{
  detail::mapImpl<InType, OutType, IdxType, MapOp, TPB, Args...>(
    out, len, map, stream, in, args...);
}

/**
 * @defgroup map Mapping ops
 * @{
 */

/**
 * @brief CUDA version of map
 * @tparam InType data-type for math operation of type raft::device_mdspan
 * @tparam MapOp the device-lambda performing the actual operation
 * @tparam TPB threads-per-block in the final kernel launched
 * @tparam OutType data-type of result of type raft::device_mdspan
 * @tparam Args additional parameters
 * @param[in] handle raft::handle_t
 * @param[in] in the input of type raft::device_mdspan
 * @param[out] out the output of the map operation of type raft::device_mdspan
 * @param[in] map the device-lambda
 * @param[in] args additional input arrays
 */
template <typename InType,
          typename MapOp,
          typename OutType,
          int TPB = 256,
          typename... Args,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void map(const raft::handle_t& handle, InType in, OutType out, MapOp map, Args... args)
{
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output is not exhaustive");
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input is not exhaustive");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Input and Output");

  if (out.size() <= std::numeric_limits<std::uint32_t>::max()) {
    map_k<in_value_t, MapOp, std::uint32_t, TPB, out_value_t, Args...>(
      out.data_handle(), out.size(), map, handle.get_stream(), in.data_handle(), args...);
  } else {
    map_k<in_value_t, MapOp, std::uint64_t, TPB, out_value_t, Args...>(
      out.data_handle(), out.size(), map, handle.get_stream(), in.data_handle(), args...);
  }
}

/**
 * @brief Perform an element-wise unary operation on the input offset into the output array
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/device_mdarray.hpp>
 *  #include <raft/core/handle.hpp>
 *  #include <raft/core/operators.hpp>
 *  #include <raft/linalg/map.cuh>
 *  ...
 *  raft::handle_t handle;
 *  auto squares = raft::make_device_vector<int>(handle, n);
 *  raft::linalg::map_offset(handle, squares.view(), raft::sq_op());
 * @endcode
 *
 * @tparam OutType Output mdspan type
 * @tparam MapOp   The unary operation type with signature `OutT func(const IdxT& idx);`
 * @param[in]  handle The raft handle
 * @param[out] out    Output array
 * @param[in]  op     The unary operation
 */
template <typename OutType,
          typename MapOp,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void map_offset(const raft::handle_t& handle, OutType out, MapOp op)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");

  using out_value_t = typename OutType::value_type;

  thrust::tabulate(
    handle.get_thrust_policy(), out.data_handle(), out.data_handle() + out.size(), op);
}

/** @} */  // end of map

}  // namespace linalg
};  // namespace raft

#endif