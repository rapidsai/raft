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
#ifndef __MULTIPLY_H
#define __MULTIPLY_H

#pragma once

#include "detail/multiply.cuh"

#include <raft/core/mdarray.hpp>

namespace raft {
namespace linalg {

/**
 * @defgroup ScalarOps Scalar operations on the input buffer
 * @tparam out_t data-type upon which the math operation will be performed
 * @tparam in_t input data-type
 * @tparam IdxType Integer type used to for addressing
 * @param out the output buffer
 * @param in the input buffer
 * @param scalar the scalar used in the operations
 * @param len number of elements in the input buffer
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename in_t, typename out_t = in_t, typename IdxType = int>
void multiplyScalar(out_t* out, const in_t* in, in_t scalar, IdxType len, cudaStream_t stream)
{
  detail::multiplyScalar(out, in, scalar, len, stream);
}
/** @} */

/**
 * @defgroup multiply Multiplication Arithmetic
 * @{
 */

/**
 * @brief Element-wise multiplication of host scalar
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @tparam ScalarIdxType Index Type of scalar
 * @param handle raft::handle_t
 * @param out the output buffer
 * @param in the input buffer
 * @param scalar the scalar used in the operations
 * @{
 */
template <typename InType,
          typename OutType       = InType,
          typename ScalarIdxType = std::uint32_t,
          typename               = raft::enable_if_device_mdspan<InType, OutType>>
void multiply_scalar(const raft::handle_t& handle,
                     OutType out,
                     const InType in,
                     raft::host_scalar_view<typename InType::element_type, ScalarIdxType> scalar)
{
  using in_element_t  = typename InType::element_type;
  using out_element_t = typename OutType::element_type;

  RAFT_EXPECTS(out.is_exhaustive(), "Output must be contiguous");
  RAFT_EXPECTS(in.is_exhaustive(), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  if (out.size() <= std::numeric_limits<std::uint32_t>::max()) {
    multiplyScalar<in_element_t, out_element_t, std::uint32_t>(
      out.data_handle(),
      in.data_handle(),
      *scalar.data_handle(),
      static_cast<std::uint32_t>(out.size()),
      handle.get_stream());
  } else {
    multiplyScalar<in_element_t, out_element_t, std::uint64_t>(
      out.data_handle(),
      in.data_handle(),
      *scalar.data_handle(),
      static_cast<std::uint64_t>(out.size()),
      handle.get_stream());
  }
}

/** @} */  // end of group multiply

};  // end namespace linalg
};  // end namespace raft

#endif