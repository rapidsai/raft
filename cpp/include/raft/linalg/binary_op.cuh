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
#ifndef __BINARY_OP_H
#define __BINARY_OP_H

#pragma once

#include "detail/binary_op.cuh"

#include <raft/core/mdarray.hpp>
#include <raft/cuda_utils.cuh>

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
 * @tparam InType Input Type raft::mdspan
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam OutType Output Type raft::mdspan
 * @tparam TPB threads-per-block in the final kernel launched
 * @param handle raft::handle_t
 * @param out Output
 * @param in1 First input
 * @param in2 Second input
 * @param op the device-lambda
 * @note Lambda must be a functor with the following signature:
 *       `OutType func(const InType& val1, const InType& val2);`
 */
template <typename InType,
          typename Lambda,
          typename OutType = InType,
          int TPB          = 256,
          typename         = raft::enable_if_mdspan<InType, OutType>>
void binary_op(
  const raft::handle_t& handle, OutType out, const InType in1, const InType in2, Lambda op)
{
  RAFT_EXPECTS(out.is_contiguous(), "Output must be contiguous");
  RAFT_EXPECTS(in1.is_contiguous(), "Input 1 must be contiguous");
  RAFT_EXPECTS(in2.is_contiguous(), "Input 2 must be contiguous");
  RAFT_EXPECTS(out.size() == in1.size() && in1.size() == in2.size(),
               "Size mismatch between Output and Inputs");

  using in_element_t  = typename InType::element_type;
  using out_element_t = typename OutType::element_type;
  binaryOp<in_element_t, Lambda, out_element_t, std::size_t, TPB>(
    out.data(), in1.data(), in2.data(), out.size(), op, handle.get_stream());
}

/** @} */  // end of group binary_op

};  // end namespace linalg
};  // end namespace raft

#endif