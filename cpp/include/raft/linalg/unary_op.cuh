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
#ifndef __UNARY_OP_H
#define __UNARY_OP_H

#pragma once

#include "detail/unary_op.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

/**
 * @brief perform element-wise unary operation in the input array
 * @tparam InType input data-type
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam OutType output data-type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output array
 * @param in the input array
 * @param len number of elements in the input array
 * @param op the device-lambda
 * @param stream cuda stream where to launch work
 * @note Lambda must be a functor with the following signature:
 *       `OutType func(const InType& val);`
 */
template <typename InType,
          typename Lambda,
          typename IdxType = int,
          typename OutType = InType,
          int TPB          = 256>
void unaryOp(OutType* out, const InType* in, IdxType len, Lambda op, cudaStream_t stream)
{
  detail::unaryOpCaller(out, in, len, op, stream);
}

/**
 * @brief Perform an element-wise unary operation into the output array
 *
 * Compared to `unaryOp()`, this method does not do any reads from any inputs
 *
 * @tparam OutType output data-type
 * @tparam Lambda  the device-lambda performing the actual operation
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB     threads-per-block in the final kernel launched
 *
 * @param[out] out    the output array [on device] [len = len]
 * @param[in]  len    number of elements in the input array
 * @param[in]  op     the device-lambda which must be of the form:
 *                    `void func(OutType* outLocationOffset, IdxType idx);`
 *                    where outLocationOffset will be out + idx.
 * @param[in]  stream cuda stream where to launch work
 */
template <typename OutType, typename Lambda, typename IdxType = int, int TPB = 256>
void writeOnlyUnaryOp(OutType* out, IdxType len, Lambda op, cudaStream_t stream)
{
  detail::writeOnlyUnaryOpCaller(out, len, op, stream);
}

/**
 * @defgroup unary_op Element-Wise Unary Operations
 * @{
 */

/**
 * @brief perform element-wise binary operation on the input arrays
 * @tparam InType Input Type raft::device_mdspan
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam OutType Output Type raft::device_mdspan
 * @param[in] handle raft::handle_t
 * @param[in] in Input
 * @param[out] out Output
 * @param[in] op the device-lambda
 * @note Lambda must be a functor with the following signature:
 *       `InType func(const InType& val);`
 */
template <typename InType,
          typename Lambda,
          typename OutType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void unary_op(const raft::handle_t& handle, InType in, OutType out, Lambda op)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  if (out.size() <= std::numeric_limits<std::uint32_t>::max()) {
    unaryOp<in_value_t, Lambda, std::uint32_t, out_value_t>(
      out.data_handle(), in.data_handle(), out.size(), op, handle.get_stream());
  } else {
    unaryOp<in_value_t, Lambda, std::uint64_t, out_value_t>(
      out.data_handle(), in.data_handle(), out.size(), op, handle.get_stream());
  }
}

/**
 * @brief perform element-wise binary operation on the input arrays
 * This function does not read from the input
 * @tparam InType Input Type raft::device_mdspan
 * @tparam Lambda the device-lambda performing the actual operation
 * @param[in] handle raft::handle_t
 * @param[inout] in Input/Output
 * @param[in] op the device-lambda
 * @note Lambda must be a functor with the following signature:
 *       `InType func(const InType& val);`
 */
template <typename InType, typename Lambda, typename = raft::enable_if_output_device_mdspan<InType>>
void write_only_unary_op(const raft::handle_t& handle, InType in, Lambda op)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input must be contiguous");

  using in_value_t = typename InType::value_type;

  if (in.size() <= std::numeric_limits<std::uint32_t>::max()) {
    writeOnlyUnaryOp<in_value_t, Lambda, std::uint32_t>(
      in.data_handle(), in.size(), op, handle.get_stream());
  } else {
    writeOnlyUnaryOp<in_value_t, Lambda, std::uint64_t>(
      in.data_handle(), in.size(), op, handle.get_stream());
  }
}

/** @} */  // end of group unary_op

};  // end namespace linalg
};  // end namespace raft

#endif