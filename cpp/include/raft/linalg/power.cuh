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
#ifndef __POWER_H
#define __POWER_H

#pragma once

#include <raft/cuda_utils.cuh>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/unary_op.cuh>

namespace raft {
namespace linalg {

/**
 * @defgroup ScalarOps Scalar operations on the input buffer
 * @tparam in_t Input data-type
 * @tparam out_t Output data-type
 * @param out the output buffer
 * @param in the input buffer
 * @param scalar the scalar used in the operations
 * @param len number of elements in the input buffer
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename in_t, typename out_t = in_t, typename IdxType = int>
void powerScalar(out_t* out, const in_t* in, in_t scalar, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(
    out, in, len, [scalar] __device__(in_t in) { return raft::myPow(in, scalar); }, stream);
}
/** @} */

/**
 * @defgroup BinaryOps Element-wise binary operations on the input buffers
 * @tparam in_t Input data-type
 * @tparam out_t Output data-type
 * @tparam IdxType Integer type used to for addressing
 * @param out the output buffer
 * @param in1 the first input buffer
 * @param in2 the second input buffer
 * @param len number of elements in the input buffers
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename in_t, typename out_t = in_t, typename IdxType = int>
void power(out_t* out, const in_t* in1, const in_t* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(
    out, in1, in2, len, [] __device__(in_t a, in_t b) { return raft::myPow(a, b); }, stream);
}
/** @} */

/**
 * @defgroup power Power Arithmetic
 * @{
 */

/**
 * @brief Elementwise power operation on the input buffers
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @param[in] handle raft::handle_t
 * @param[in] in1    First Input
 * @param[in] in2    Second Input
 * @param[out] out    Output
 */
template <typename InType,
          typename OutType,
          typename = raft::enable_if_device_mdspan<OutType, InType>>
void power(const raft::handle_t& handle, InType in1, InType in2, OutType out)
{
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(out.is_exhaustive(), "Output must be contiguous");
  RAFT_EXPECTS(in1.is_exhaustive(), "Input 1 must be contiguous");
  RAFT_EXPECTS(in2.is_exhaustive(), "Input 2 must be contiguous");
  RAFT_EXPECTS(out.size() == in1.size() && in1.size() == in2.size(),
               "Size mismatch between Output and Inputs");

  if (out.size() <= std::numeric_limits<std::uint32_t>::max()) {
    power<in_value_t, out_value_t, std::uint32_t>(out.data_handle(),
                                                  in1.data_handle(),
                                                  in2.data_handle(),
                                                  static_cast<std::uint32_t>(out.size()),
                                                  handle.get_stream());
  } else {
    power<in_value_t, out_value_t, std::uint64_t>(out.data_handle(),
                                                  in1.data_handle(),
                                                  in2.data_handle(),
                                                  static_cast<std::uint64_t>(out.size()),
                                                  handle.get_stream());
  }
}

/**
 * @brief Elementwise power of host scalar to input
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @tparam ScalarIdxType Index Type of scalar
 * @param[in] handle raft::handle_t
 * @param[in] in    Input
 * @param[out] out    Output
 * @param[in] scalar    raft::host_scalar_view
 */
template <typename InType,
          typename OutType,
          typename ScalarIdxType,
          typename = raft::enable_if_device_mdspan<OutType, InType>>
void power_scalar(const raft::handle_t& handle,
                  InType in,
                  OutType out,
                  const raft::host_scalar_view<typename InType::element_type, ScalarIdxType> scalar)
{
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(out.is_exhaustive(), "Output must be contiguous");
  RAFT_EXPECTS(in.is_exhaustive(), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  if (out.size() <= std::numeric_limits<std::uint32_t>::max()) {
    powerScalar<in_value_t, out_value_t, std::uint32_t>(out.data_handle(),
                                                        in.data_handle(),
                                                        *scalar.data_handle(),
                                                        static_cast<std::uint32_t>(out.size()),
                                                        handle.get_stream());
  } else {
    powerScalar<in_value_t, out_value_t, std::uint64_t>(out.data_handle(),
                                                        in.data_handle(),
                                                        *scalar.data_handle(),
                                                        static_cast<std::uint64_t>(out.size()),
                                                        handle.get_stream());
  }
}

/** @} */  // end of group add

};  // end namespace linalg
};  // end namespace raft

#endif