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

#ifndef __TERNARY_OP_H
#define __TERNARY_OP_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>

namespace raft {
namespace linalg {
/**
 * @brief perform element-wise ternary operation on the input arrays
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output array
 * @param in1 the first input array
 * @param in2 the second input array
 * @param in3 the third input array
 * @param len number of elements in the input array
 * @param op the device-lambda
 * @param stream cuda stream where to launch work
 */
template <typename math_t, typename Lambda, typename out_t, typename IdxType = int, int TPB = 256>
void ternaryOp(out_t* out,
               const math_t* in1,
               const math_t* in2,
               const math_t* in3,
               IdxType len,
               Lambda op,
               cudaStream_t stream)
{
  return detail::map<false>(stream, out, len, op, in1, in2, in3);
}

/**
 * @defgroup ternary_op Element-Wise Ternary Operation
 * @{
 */

/**
 * @brief perform element-wise ternary operation on the input arrays
 * @tparam InType Input Type raft::device_mdspan
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam OutType Output Type raft::device_mdspan
 * @param[in] handle raft::resources
 * @param[in] in1 First input
 * @param[in] in2 Second input
 * @param[in] in3 Third input
 * @param[out] out Output
 * @param[in] op the device-lambda
 * @note Lambda must be a functor with the following signature:
 *       `OutType func(const InType& val1, const InType& val2, const InType& val3);`
 */
template <typename InType,
          typename Lambda,
          typename OutType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void ternary_op(
  raft::resources const& handle, InType in1, InType in2, InType in3, OutType out, Lambda op)
{
  return map(handle, in1, in2, in3, out, op);
}

/** @} */  // end of group ternary_op

};  // end namespace linalg
};  // end namespace raft

#endif
