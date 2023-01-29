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
#ifndef __BINARY_OP_H
#define __BINARY_OP_H

#pragma once

#include "detail/binary_op.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

/**
 * @brief perform element-wise binary operation on the input arrays
 * @tparam InType input data-type
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam OutType output data-type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output array
 * @param in1 the first input array
 * @param in2 the second input array
 * @param len number of elements in the input array
 * @param op the device-lambda
 * @param stream cuda stream where to launch work
 * @note Lambda must be a functor with the following signature:
 *       `OutType func(const InType& val1, const InType& val2);`
 */
template <typename InType,
          typename Lambda,
          typename OutType = InType,
          typename IdxType = int,
          int TPB          = 256>
void binaryOp(
  OutType* out, const InType* in1, const InType* in2, IdxType len, Lambda op, cudaStream_t stream)
{
  detail::binaryOp(out, in1, in2, len, op, stream);
}

/**
 * @defgroup binary_op Element-Wise Binary Operation
 * @{
 */

/**
 * @brief perform element-wise binary operation on the input arrays
 * @tparam InType Input Type raft::device_mdspan
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam OutType Output Type raft::device_mdspan
 * @param[in] handle raft::device_resources
 * @param[in] in1 First input
 * @param[in] in2 Second input
 * @param[out] out Output
 * @param[in] op the device-lambda
 * @note Lambda must be a functor with the following signature:
 *       `OutType func(const InType& val1, const InType& val2);`
 */
template <typename InType,
          typename Lambda,
          typename OutType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void binary_op(raft::device_resources const& handle, InType in1, InType in2, OutType out, Lambda op)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in1), "Input 1 must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in2), "Input 2 must be contiguous");
  RAFT_EXPECTS(out.size() == in1.size() && in1.size() == in2.size(),
               "Size mismatch between Output and Inputs");

  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  if (out.size() <= std::numeric_limits<std::uint32_t>::max()) {
    binaryOp<in_value_t, Lambda, out_value_t, std::uint32_t>(
      out.data_handle(), in1.data_handle(), in2.data_handle(), out.size(), op, handle.get_stream());
  } else {
    binaryOp<in_value_t, Lambda, out_value_t, std::uint64_t>(
      out.data_handle(), in1.data_handle(), in2.data_handle(), out.size(), op, handle.get_stream());
  }
}

/** @} */  // end of group binary_op

};  // end namespace linalg
};  // end namespace raft

#endif