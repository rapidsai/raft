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
 * Run a function over matrix lines (rows or columns) with a variable number
 * row-vectors or column-vectors.
 * The term `line` here signifies that the lines can be either columns or rows,
 * depending on the matrix layout.
 * What matters is if the vectors are applied along lines (indices of vectors correspond to
 * indices within lines), or across lines (indices of vectors correspond to line numbers).
 *
 * @param [out] out result of the operation; can be same as `in`; should be aligned the same
 *        as `in` to allow faster vectorized memory transfers.
 * @param [in] in input matrix consisting of `nLines` lines, each `lineLen`-long.
 * @param [in] lineLen length of matrix line in elements (`=nCols` in row-major or `=nRows` in
 * col-major)
 * @param [in] nLines number of matrix lines (`=nRows` in row-major or `=nCols` in col-major)
 * @param [in] alongLines whether vectors are indices along or across lines.
 * @param [in] op the operation applied on each line:
 *    for i in [0..lineLen) and j in [0..nLines):
 *      out[i, j] = op(in[i, j], vec1[i], vec2[i], ... veck[i])   if alongLines = true
 *      out[i, j] = op(in[i, j], vec1[j], vec2[j], ... veck[j])   if alongLines = false
 *    where matrix indexing is row-major ([i, j] = [i + lineLen * j]).
 * @param [in] stream a cuda stream for the kernels
 * @param [in] vecs zero or more vectors to be passed as arguments,
 *    size of each vector is `alongLines ? lineLen : nLines`.
 */
template <typename m_t, typename idx_t = int, typename Lambda, typename... Vecs>
void linewise_op(const raft::handle_t &handle,
                 raft::device_matrix_view<const m_t> in,
                 raft::device_matrix_view<m_t> out,
                 const idx_t lineLen,
                 const idx_t nLines,
                 const bool alongLines,
                 Lambda op,
                 raft::device_vector_view<Vecs>... vecs) {
    detail::MatrixLinewiseOp<16, 256>::run<m_t, idx_t, Lambda, Vecs...>(
            out, in, lineLen, nLines, alongLines, op, stream, vecs...);
}
}
