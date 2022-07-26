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
 * @tparam InT     input data-type. Also the data-type upon which the math ops
 *                 will be performed
 * @tparam OutT    output data-type
 * @tparam IdxType Integer type used to for addressing
 * @param outDev the output buffer
 * @param inDev the input buffer
 * @param singleScalarDev pointer to the scalar located in device memory
 * @param len number of elements in the input and output buffer
 * @param stream cuda stream
 */
template <typename InT, typename OutT = InT, typename IdxType = int>
void addDevScalar(
  OutT* outDev, const InT* inDev, const InT* singleScalarDev, IdxType len, cudaStream_t stream)
{
  detail::addDevScalar(outDev, inDev, singleScalarDev, len, stream);
}

/**
 * @defgroup add Addition Arithmetic
 * @{
 */

/**
 * @brief Elementwise add operation on the input buffers
 * @tparam OutType   Output Type raft::mdspan
 * @tparam InType    Input Type raft::mdspan
 * @param handle raft::handle_t
 * @param out    Output
 * @param in1    First Input
 * @param in2    Second Input
 */
template <typename OutType, typename InType, typename = raft::enable_if_mdspan<OutType, InType>>
void add(const raft::handle_t& handle, OutType out, const InType in1, const InType in2)
{
  using in_element_t  = typename InType::element_type;
  using out_element_t = typename OutType::element_type;

  RAFT_EXPECTS(out.is_exhaustive(), "Output must be contiguous");
  RAFT_EXPECTS(in1.is_exhaustive(), "Input 1 must be contiguous");
  RAFT_EXPECTS(in2.is_exhaustive(), "Input 2 must be contiguous");
  RAFT_EXPECTS(out.size() == in1.size() && in1.size() == in2.size(),
               "Size mismatch between Output and Inputs");

  if (out.size() <= std::numeric_limits<std::uint32_t>::max()) {
    add<in_element_t, out_element_t, std::uint32_t>(out.data_handle(),
                                                    in1.data_handle(),
                                                    in2.data_handle(),
                                                    static_cast<std::uint32_t>(out.size()),
                                                    handle.get_stream());
  } else {
    add<in_element_t, out_element_t, std::uint64_t>(out.data_handle(),
                                                    in1.data_handle(),
                                                    in2.data_handle(),
                                                    static_cast<std::uint64_t>(out.size()),
                                                    handle.get_stream());
  }
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
  using in_element_t  = typename InType::element_type;
  using out_element_t = typename OutType::element_type;

  RAFT_EXPECTS(out.is_exhaustive(), "Output must be contiguous");
  RAFT_EXPECTS(in.is_exhaustive(), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  if (raft::is_device_ptr(scalar.data_handle())) {
    if (out.size() <= std::numeric_limits<std::uint32_t>::max()) {
      addDevScalar<in_element_t, out_element_t, std::uint32_t>(
        out.data_handle(),
        in.data_handle(),
        scalar.data_handle(),
        static_cast<std::uint32_t>(out.size()),
        handle.get_stream());
    } else {
      addDevScalar<in_element_t, out_element_t, std::uint64_t>(
        out.data_handle(),
        in.data_handle(),
        scalar.data_handle(),
        static_cast<std::uint64_t>(out.size()),
        handle.get_stream());
    }
  } else {
    if (out.size() <= std::numeric_limits<std::uint32_t>::max()) {
      addScalar<in_element_t, out_element_t, std::uint32_t>(out.data_handle(),
                                                            in.data_handle(),
                                                            *scalar.data_handle(),
                                                            static_cast<std::uint32_t>(out.size()),
                                                            handle.get_stream());
    } else {
      addScalar<in_element_t, out_element_t, std::uint64_t>(out.data_handle(),
                                                            in.data_handle(),
                                                            *scalar.data_handle(),
                                                            static_cast<std::uint64_t>(out.size()),
                                                            handle.get_stream());
    }
  }
}

/** @} */  // end of group add

};  // end namespace linalg
};  // end namespace raft

/** @} */

#endif