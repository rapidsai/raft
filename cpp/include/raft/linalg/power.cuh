/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include <raft/core/host_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/input_validation.hpp>

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
void powerScalar(out_t* out, const in_t* in, const in_t scalar, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(out, in, len, raft::pow_const_op<in_t>(scalar), stream);
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
  raft::linalg::binaryOp(out, in1, in2, len, raft::pow_op(), stream);
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
 * @param[in] handle raft::resources
 * @param[in] in1    First Input
 * @param[in] in2    Second Input
 * @param[out] out    Output
 */
template <typename InType,
          typename OutType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void power(raft::resources const& handle, InType in1, InType in2, OutType out)
{
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in1), "Input 1 must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in2), "Input 2 must be contiguous");
  RAFT_EXPECTS(out.size() == in1.size() && in1.size() == in2.size(),
               "Size mismatch between Output and Inputs");

  if (out.size() <= std::numeric_limits<std::uint32_t>::max()) {
    power<in_value_t, out_value_t, std::uint32_t>(out.data_handle(),
                                                  in1.data_handle(),
                                                  in2.data_handle(),
                                                  static_cast<std::uint32_t>(out.size()),
                                                  resource::get_cuda_stream(handle));
  } else {
    power<in_value_t, out_value_t, std::uint64_t>(out.data_handle(),
                                                  in1.data_handle(),
                                                  in2.data_handle(),
                                                  static_cast<std::uint64_t>(out.size()),
                                                  resource::get_cuda_stream(handle));
  }
}

/**
 * @brief Elementwise power of host scalar to input
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @tparam ScalarIdxType Index Type of scalar
 * @param[in] handle raft::resources
 * @param[in] in    Input
 * @param[out] out    Output
 * @param[in] scalar    raft::host_scalar_view
 */
template <typename InType,
          typename OutType,
          typename ScalarIdxType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void power_scalar(
  raft::resources const& handle,
  InType in,
  OutType out,
  const raft::host_scalar_view<const typename InType::value_type, ScalarIdxType> scalar)
{
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  if (out.size() <= std::numeric_limits<std::uint32_t>::max()) {
    powerScalar<in_value_t, out_value_t, std::uint32_t>(out.data_handle(),
                                                        in.data_handle(),
                                                        *scalar.data_handle(),
                                                        static_cast<std::uint32_t>(out.size()),
                                                        resource::get_cuda_stream(handle));
  } else {
    powerScalar<in_value_t, out_value_t, std::uint64_t>(out.data_handle(),
                                                        in.data_handle(),
                                                        *scalar.data_handle(),
                                                        static_cast<std::uint64_t>(out.size()),
                                                        resource::get_cuda_stream(handle));
  }
}

/** @} */  // end of group add

};  // end namespace linalg
};  // end namespace raft

#endif