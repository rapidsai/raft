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
#ifndef __ADD_H
#define __ADD_H

/**
 * @defgroup arithmetic Dense matrix arithmetic
 * @{
 */

#pragma once

#include "detail/add.cuh"

#include <raft/core/mdarray.hpp>

namespace raft {
namespace linalg {

using detail::adds_scalar;

/**
 * @ingroup arithmetic
 * @brief Elementwise scalar add operation on the input buffer
 *
 * @tparam InT     input data-type. Also the data-type upon which the math ops
 *                 will be performed
 * @tparam OutT    output data-type
 * @tparam IdxType Integer type used to for addressing
 *
 * @param out    the output buffer
 * @param in     the input buffer
 * @param scalar the scalar used in the operations
 * @param len    number of elements in the input buffer
 * @param stream cuda stream where to launch work
 */
template <typename InT, typename OutT = InT, typename IdxType = int>
void addScalar(OutT* out, const InT* in, const InT scalar, IdxType len, cudaStream_t stream)
{
  detail::addScalar(out, in, scalar, len, stream);
}

/**
 * @brief Elementwise add operation on the input buffers
 * @tparam InT     input data-type. Also the data-type upon which the math ops
 *                 will be performed
 * @tparam OutT    output data-type
 * @tparam IdxType Integer type used to for addressing
 *
 * @param out    the output buffer
 * @param in1    the first input buffer
 * @param in2    the second input buffer
 * @param len    number of elements in the input buffers
 * @param stream cuda stream where to launch work
 */
template <typename InT, typename OutT = InT, typename IdxType = int>
void add(OutT* out, const InT* in1, const InT* in2, IdxType len, cudaStream_t stream)
{
  detail::add(out, in1, in2, len, stream);
}

/** Substract single value pointed by singleScalarDev parameter in device memory from inDev[i] and
 * write result to outDev[i]
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param outDev the output buffer
 * @param inDev the input buffer
 * @param singleScalarDev pointer to the scalar located in device memory
 * @param len number of elements in the input and output buffer
 * @param stream cuda stream
 */
template <typename math_t, typename IdxType = int>
void addDevScalar(math_t* outDev,
                  const math_t* inDev,
                  const math_t* singleScalarDev,
                  IdxType len,
                  cudaStream_t stream)
{
  detail::addDevScalar(outDev, inDev, singleScalarDev, len, stream);
}

/**
 * @defgroup add Addition Arithmetic
 * @{
 */

/**
 * @brief Elementwise add operation on the input buffers
 * @tparam out_t   Output Type raft::mdspan
 * @tparam in_t    Input Type raft::mdspan
 * @param handle raft::handle_t
 * @param out    Output
 * @param in1    First Input
 * @param in2    Second Input
 */
template <typename out_t, typename in_t, typename = raft::enable_if_mdspan<out_t, in_t>>
void add(const raft::handle_t& handle, out_t out, const in_t in1, const in_t in2)
{
  RAFT_EXPECTS(out.is_contiguous(), "Output must be contiguous");
  RAFT_EXPECTS(in1.is_contiguous(), "Input 1 must be contiguous");
  RAFT_EXPECTS(in2.is_contiguous(), "Input 2 must be contiguous");
  RAFT_EXPECTS(out.size() == in1.size() && in1.size() == in2.size(),
               "Size mismatch between Output and Inputs");

  add(out.data(), in1.data(), in2.data(), out.size(), handle.get_stream());
}

/**
 * @brief Elementwise addition of scalar to input
 * @tparam OutType   Output Type raft::mdspan
 * @tparam InType    Input Type raft::mdspan
 * @param handle raft::handle_t
 * @param out    Output
 * @param in    Input
 * @param scalar    raft::scalar_view in either host or device memory
 */
template <typename OutType, typename InType, typename = raft::enable_if_mdspan<OutType, InType>>
void add_scalar(const raft::handle_t& handle,
                OutType out,
                const InType in,
                const raft::scalar_view<typename InType::element_type> scalar)
{
  RAFT_EXPECTS(out.is_contiguous(), "Output must be contiguous");
  RAFT_EXPECTS(in.is_contiguous(), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  if (raft::is_device_ptr(scalar.data())) {
    addDevScalar(out.data(), in.data(), scalar.data(), out.size(), handle.get_stream());
  } else {
    addScalar(out.data(), in.data(), *scalar.data(), out.size(), handle.get_stream());
  }
}

/** @} */  // end of group add

};  // end namespace linalg
};  // end namespace raft

/** @} */

#endif