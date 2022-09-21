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

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/matrix/matrix.cuh>
#include <raft/matrix/detail/matrix.cuh>

namespace raft::matrix {
/**
 * @brief set values to scalar in matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @param handle: raft handle
 * @param in input matrix
 * @param out output matrix. The result is stored in the out matrix
 * @param scalar svalar value
 */
template <typename math_t>
void fill(const raft::handle_t &handle,
          raft::device_matrix_view<const math_t> in,
          raft::device_matrix_view<math_t> out, math_t scalar) {
    RAFT_EXPECTS(in.size() == out.size(), "Input and output matrices must be the same size.");
    detail::setValue(out.data_handle(), in.data_handle(), scalar, in.size(), handle.get_stream());
}
}
