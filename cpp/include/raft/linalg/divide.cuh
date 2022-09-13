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
#ifndef __DIVIDE_H
#define __DIVIDE_H

#pragma once

#include "detail/divide.cuh"

#include <raft/core/mdarray.hpp>

namespace raft {
namespace linalg {

using detail::divides_scalar;

/**
 * @defgroup ScalarOps Scalar operations on the input buffer
 * @tparam OutT output data-type upon which the math operation will be performed
 * @tparam InT input data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param out the output buffer
 * @param in the input buffer
 * @param scalar the scalar used in the operations
 * @param len number of elements in the input buffer
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename InT, typename OutT = InT, typename IdxType = int>
void divideScalar(OutT* out, const InT* in, InT scalar, IdxType len, cudaStream_t stream)
{
  detail::divideScalar(out, in, scalar, len, stream);
}
/** @} */

/**
 * @defgroup divide Division Arithmetic
 * @{
 */

/**
 * @brief Elementwise division of input by host scalar
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @tparam ScalarIdxType Index Type of scalar
 * @param[in] handle raft::handle_t
 * @param[in] in    Input
 * @param[in] scalar    raft::host_scalar_view
 * @param[out] out    Output
 */
template <typename InType,
          typename OutType,
          typename ScalarIdxType,
          typename = raft::enable_if_device_mdspan<OutType, InType>>
void divide_scalar(const raft::handle_t& handle,
                   InType in,
                   OutType out,
                   raft::host_scalar_view<const typename InType::value_type, ScalarIdxType> scalar)
{
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(out.is_exhaustive(), "Output must be contiguous");
  RAFT_EXPECTS(in.is_exhaustive(), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  if (out.size() <= std::numeric_limits<std::uint32_t>::max()) {
    divideScalar<in_value_t, out_value_t, std::uint32_t>(out.data_handle(),
                                                         in.data_handle(),
                                                         *scalar.data_handle(),
                                                         static_cast<std::uint32_t>(out.size()),
                                                         handle.get_stream());
  } else {
    divideScalar<in_value_t, out_value_t, std::uint64_t>(out.data_handle(),
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