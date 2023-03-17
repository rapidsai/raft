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

/**
 * This file is deprecated and will be removed in a future release.
 * Please use versions in individual header files instead.
 */

#ifndef __UNARY_OP_H
#define __UNARY_OP_H

#pragma once

#pragma message(__FILE__                                                  \
                " is deprecated and will be removed in a future release." \
                " Please use raft/linalg/map.cuh.")

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/linalg/map.cuh>

namespace raft {
namespace linalg {

/**
 * @brief perform element-wise unary operation in the input array
 *
 * @note This operation is deprecated. Please use `map` in `raft/linalg/map.cuh` instead.
 *
 * @tparam InType input data-type
 * @tparam Lambda Device lambda performing the actual operation, with the signature
 *         `OutType func(const InType& val);`
 * @tparam OutType output data-type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads-per-block in the final kernel launched
 * @param[out] out    Output array [on device], dim = [len]
 * @param[in]  in     Input array [on device], dim = [len]
 * @param[in]  len    Number of elements in the input array
 * @param[in]  op     Device lambda
 * @param[in]  stream cuda stream where to launch work
 */
template <typename InType,
          typename Lambda,
          typename IdxType = int,
          typename OutType = InType,
          int TPB          = 256>
[[deprecated("Use function `linalg::map` from raft/linalg/map.cuh")]] void unaryOp(
  OutType* out, const InType* in, IdxType len, Lambda op, cudaStream_t stream)
{
  return detail::map<false>(stream, out, len, op, in);
}

/**
 * @brief Perform an element-wise unary operation into the output array
 *
 * @note This operation is deprecated. Please use `map_offset` in `raft/linalg/map.cuh` instead.
 *
 * @tparam OutType output data-type
 * @tparam Lambda  Device lambda performing the actual operation, with the signature
 *                 `void func(OutType* p, IdxType idx);`
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB     threads-per-block in the final kernel launched
 *
 * @param[out] out    Output array [on device], dim = [len]
 * @param[in]  len    Number of elements in the input array
 * @param[in]  op     Device lambda
 * @param[in]  stream cuda stream where to launch work
 */
template <typename OutType, typename Lambda, typename IdxType = int, int TPB = 256>
[[deprecated("Use function `linalg::map_offset` from raft/linalg/map.cuh")]] void writeOnlyUnaryOp(
  OutType* out, IdxType len, Lambda op, cudaStream_t stream)
{
  RAFT_LOG_WARN(
    "Using the deprecated function raft::linalg::writeOnlyUnaryOp with dangerous semantics. "
    "Please replace it with raft::linalg::map_offset from raft/linalg/map.cuh");
  return detail::map<true>(stream, out, len, [out, op] __device__(IdxType offset) {
    OutType& r = out[offset];
    op(&r, offset);
    return r;
  });
}

/**
 * @defgroup unary_op Element-Wise Unary Operations
 * @{
 */

/**
 * @brief Perform an element-wise unary operation into the output array
 *
 * @note This operation is deprecated. Please use `map` in `raft/linalg/map.cuh` instead.
 *
 * @tparam InType Input Type raft::device_mdspan
 * @tparam Lambda Device lambda performing the actual operation, with the signature
 *                `out_value_t func(const in_value_t& val);`
 * @tparam OutType Output Type raft::device_mdspan
 * @param[in]  handle The raft handle
 * @param[in]  in     Input
 * @param[out] out    Output
 * @param[in]  op     Device lambda
 */
template <typename InType,
          typename Lambda,
          typename OutType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
[[deprecated("Use function `linalg::map` from raft/linalg/map.cuh")]] void unary_op(
  raft::device_resources const& handle, InType in, OutType out, Lambda op)
{
  return map(handle, in, out, op);
}

/**
 * @brief Perform an element-wise unary operation on the input index into the output array
 *
 * @note This operation is deprecated. Please use `map_offset` in `raft/linalg/map.cuh` instead.
 *
 * @tparam OutType Output Type raft::device_mdspan
 * @tparam Lambda  Device lambda performing the actual operation, with the signature
 *                 `void func(out_value_t* out_location, index_t idx);`
 * @param[in]  handle The raft handle
 * @param[out] out    Output
 * @param[in]  op     Device lambda
 */
template <typename OutType,
          typename Lambda,
          typename = raft::enable_if_output_device_mdspan<OutType>>
[[deprecated("Use function `linalg::map_offset` from raft/linalg/map.cuh")]] void
write_only_unary_op(const raft::device_resources& handle, OutType out, Lambda op)
{
  return writeOnlyUnaryOp(out.data_handle(), out.size(), op, handle.get_stream());
}

/** @} */  // end of group unary_op

};  // end namespace linalg
};  // end namespace raft

#endif
