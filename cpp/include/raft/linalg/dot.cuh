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
#ifndef __DOT_H
#define __DOT_H

#pragma once

#include <raft/linalg/detail/cublas_wrappers.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>

namespace raft::linalg {

/**
 * @brief Computes the dot product of two vectors.
 * @tparam InputType1  raft::device_mdspan for the first input vector
 * @tparam InputType2  raft::device_mdspan for the second input vector
 * @tparam OutputType  Either a host_scalar_view or device_scalar_view for the output
 * @param[in] handle   raft::handle_t
 * @param[in] x        First input vector
 * @param[in] y        Second input vector
 * @param[out] out     The output dot product between the x and y vectors
 */
template <typename InputType1,
          typename InputType2,
          typename OutputType,
          typename = raft::enable_if_input_device_mdspan<InputType1>,
          typename = raft::enable_if_input_device_mdspan<InputType2>,
          typename = raft::enable_if_output_mdspan<OutputType>>
void dot(const raft::handle_t& handle, InputType1 x, InputType2 y, OutputType out)
{
  RAFT_EXPECTS(x.size() == y.size(),
               "Size mismatch between x and y input vectors in raft::linalg::dot");

  // Right now the inputs and outputs need to all have the same value_type (float/double etc).
  // Try to output a meaningful compiler error if mismatched types are passed here.
  // Note: In the future we could remove this restriction using the cublasDotEx function
  // in the cublas wrapper call, instead of the cublassdot and cublasddot functions.
  static_assert(std::is_same_v<typename InputType1::value_type, typename InputType2::value_type>,
                "Both input vectors need to have the same value_type in raft::linalg::dot call");
  static_assert(
    std::is_same_v<typename InputType1::value_type, typename OutputType::value_type>,
    "Input vectors and output scalar need to have the same value_type in raft::linalg::dot call");

  RAFT_CUBLAS_TRY(detail::cublasdot(handle.get_cublas_handle(),
                                    x.size(),
                                    x.data_handle(),
                                    x.stride(0),
                                    y.data_handle(),
                                    y.stride(0),
                                    out.data_handle(),
                                    handle.get_stream()));
}
}  // namespace raft::linalg
#endif
