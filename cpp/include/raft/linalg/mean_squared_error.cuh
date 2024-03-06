/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#ifndef __MSE_H
#define __MSE_H

#pragma once

#include "detail/mean_squared_error.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace raft {
namespace linalg {

/**
 * @brief CUDA version mean squared error function mean((A-B)**2)
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam TPB threads-per-block
 * @param out the output mean squared error value (assumed to be a device pointer)
 * @param A input array (assumed to be a device pointer)
 * @param B input array (assumed to be a device pointer)
 * @param len number of elements in the input arrays
 * @param weight weight to apply to every term in the mean squared error calculation
 * @param stream cuda-stream where to launch this kernel
 */
template <typename in_t, typename out_t, typename idx_t = size_t>
void meanSquaredError(
  out_t* out, const in_t* A, const in_t* B, idx_t len, in_t weight, cudaStream_t stream)
{
  detail::meanSquaredError(out, A, B, len, weight, stream);
}

/**
 * @defgroup mean_squared_error Mean Squared Error
 * @{
 */

/**
 * @brief CUDA version mean squared error function mean((A-B)**2)
 * @tparam InValueType Input data-type
 * @tparam IndexType Input/Output index type
 * @tparam OutValueType Output data-type
 * @tparam TPB threads-per-block
 * @param[in] handle raft::resources
 * @param[in] A input raft::device_vector_view
 * @param[in] B input raft::device_vector_view
 * @param[out] out the output mean squared error value of type raft::device_scalar_view
 * @param[in] weight weight to apply to every term in the mean squared error calculation
 */
template <typename InValueType, typename IndexType, typename OutValueType>
void mean_squared_error(raft::resources const& handle,
                        raft::device_vector_view<const InValueType, IndexType> A,
                        raft::device_vector_view<const InValueType, IndexType> B,
                        raft::device_scalar_view<OutValueType, IndexType> out,
                        OutValueType weight)
{
  RAFT_EXPECTS(A.size() == B.size(), "Size mismatch between inputs");

  meanSquaredError(out.data_handle(),
                   A.data_handle(),
                   B.data_handle(),
                   A.extent(0),
                   weight,
                   resource::get_cuda_stream(handle));
}

/** @} */  // end of group mean_squared_error

};  // end namespace linalg
};  // end namespace raft

#endif
