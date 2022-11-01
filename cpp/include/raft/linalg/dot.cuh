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
 * @param[in] handle   raft::handle_t
 * @param[in] x        First input vector
 * @param[in] y        Second input vector
 * @param[out] out     The output dot product between the x and y vectors.
 * @note The out parameter can be either a host_scalar_view or device_scalar_view
 */
template <typename ElementType,
          typename IndexType       = std::uint32_t,
          typename ScalarIndexType = std::uint32_t,
          typename LayoutPolicy1   = layout_c_contiguous,
          typename LayoutPolicy2   = layout_c_contiguous>
void dot(const raft::handle_t& handle,
         raft::device_vector_view<const ElementType, IndexType, LayoutPolicy1> x,
         raft::device_vector_view<const ElementType, IndexType, LayoutPolicy2> y,
         raft::device_scalar_view<ElementType, ScalarIndexType> out)
{
  RAFT_EXPECTS(x.size() == y.size(),
               "Size mismatch between x and y input vectors in raft::linalg::dot");

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
