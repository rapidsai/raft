/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
#ifndef __SQRT_H
#define __SQRT_H

#pragma once

#include <raft/core/mdarray.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/unary_op.cuh>

namespace raft {
namespace linalg {

/**
 * @defgroup ScalarOps Scalar operations on the input buffer
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param out the output buffer
 * @param in the input buffer
 * @param len number of elements in the input buffer
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename in_t, typename out_t = in_t, typename IdxType = int>
void sqrt(out_t* out, const in_t* in, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(
    out, in, len, [] __device__(in_t in) { return raft::mySqrt(in); }, stream);
}
/** @} */

/**
 * @defgroup power Sqrt Arithmetic
 * @{
 */

/**
 * @brief Elementwise sqrt operation on the input buffer
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @param handle raft::handle_t
 * @param out    Output
 * @param in     Input
 */
template <typename InType,
          typename OutType = InType,
          typename         = raft::enable_if_device_mdspan<OutType, InType>>
void sqrt(const raft::handle_t& handle, OutType out, const InType in)
{
  using in_element_t  = typename InType::element_type;
  using out_element_t = typename OutType::element_type;

  RAFT_EXPECTS(out.is_exhaustive(), "Output must be contiguous");
  RAFT_EXPECTS(in.is_exhaustive(), "Input 1 must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Inputs");

  if (out.size() <= std::numeric_limits<std::uint32_t>::max()) {
    sqrt<in_element_t, out_element_t, std::uint32_t>(out.data_handle(),
                                                     in.data_handle(),
                                                     static_cast<std::uint32_t>(out.size()),
                                                     handle.get_stream());
  } else {
    sqrt<in_element_t, out_element_t, std::uint64_t>(out.data_handle(),
                                                     in.data_handle(),
                                                     static_cast<std::uint64_t>(out.size()),
                                                     handle.get_stream());
  }
}

/** @} */  // end of group add

};  // end namespace linalg
};  // end namespace raft

#endif