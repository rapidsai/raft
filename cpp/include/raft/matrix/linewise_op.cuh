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
#include <raft/matrix/detail/matrix.cuh>
#include <raft/matrix/matrix.cuh>

namespace raft::matrix {

// template<typename idx_t, typename arg, typename... args>
// args *extract_ptr(raft::device_vector_view<const arg, idx_t> vec, raft::device_vector_view<const
// args, idx_t>... vecs) {
//     vecs.data_handle();
//}

/**
 * Run a function over matrix lines (rows or columns) with a variable number
 * row-vectors or column-vectors.
 * The term `line` here signifies that the lines can be either columns or rows,
 * depending on the matrix layout.
 * What matters is if the vectors are applied along lines (indices of vectors correspond to
 * indices within lines), or across lines (indices of vectors correspond to line numbers).
 * @tparam m_t matrix elements type
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @tparam Lambda type of lambda function used for the operation
 * @tparam vec_t variadic types of device_vector_view vectors (size m if alongRows, size n
 * otherwise)
 * @param[in] handle raft handle for managing resources
 * @param [out] out result of the operation; can be same as `in`; should be aligned the same
 *        as `in` to allow faster vectorized memory transfers.
 * @param [in] in input matrix consisting of `nLines` lines, each `lineLen`-long.
 * @param [in] alongLines whether vectors are indices along or across lines.
 * @param [in] op the operation applied on each line:
 *    for i in [0..lineLen) and j in [0..nLines):
 *      out[i, j] = op(in[i, j], vec1[i], vec2[i], ... veck[i])   if alongLines = true
 *      out[i, j] = op(in[i, j], vec1[j], vec2[j], ... veck[j])   if alongLines = false
 *    where matrix indexing is row-major ([i, j] = [i + lineLen * j]).
 * @param [in] vecs zero or more vectors to be passed as arguments,
 *    size of each vector is `alongLines ? lineLen : nLines`.
 */
template <typename m_t,
          typename idx_t,
          typename layout,
          typename Lambda,
          typename... vec_t,
          typename = raft::enable_if_device_mdspan<vec_t...>>
void linewise_op(const raft::handle_t& handle,
                 raft::device_matrix_view<const m_t, idx_t, layout> in,
                 raft::device_matrix_view<m_t, idx_t, layout> out,
                 const bool alongLines,
                 Lambda op,
                 vec_t... vecs)
{
  constexpr auto is_rowmajor = std::is_same_v<layout, row_major>;
  constexpr auto is_colmajor = std::is_same_v<layout, col_major>;

  static_assert(is_rowmajor || is_colmajor,
                "layout for in and out must be either row or col major");

  const idx_t lineLen = is_rowmajor ? in.extent(0) : in.extent(1);
  const idx_t nLines  = is_rowmajor ? in.extent(1) : in.extent(0);

  RAFT_EXPECTS(out.extent(0) == in.extent(0) && out.extent(1) == in.extent(1),
               "Input and output must have the same shape.");

  detail::MatrixLinewiseOp<16, 256>::run<m_t, idx_t>(out.data_handle(),
                                                     in.data_handle(),
                                                     lineLen,
                                                     nLines,
                                                     alongLines,
                                                     op,
                                                     handle.get_stream(),
                                                     vecs.data_handle()...);
}
}  // namespace raft::matrix
