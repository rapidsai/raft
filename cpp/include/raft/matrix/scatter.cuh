/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <raft/core/resources.hpp>
#include <raft/matrix/detail/scatter_inplace.cuh>

namespace raft::matrix {
/**
 * @brief In-place scatter elements in a row-major matrix according to a
 * map. The map specifies the new order in which rows of the input matrix are
 * rearranged, i.e. read the destination index from the map, and copy the row. For example,
 * the matrix [[1, 2, 3], [4, 5, 6], [7, 8, 9]] with the map [2, 0, 1] will
 * be transformed to [[4, 5, 6], [7, 8, 9], [1, 2, 3]]. Batching is done on
 * columns and an additional scratch space of shape n_rows * cols_batch_size
 * is created. For each batch, chunks of columns from each row are copied
 * into the appropriate location in the scratch space and copied back to
 * the corresponding locations in the input matrix.
 * Note: in-place scatter is not thread safe if the values in the map are not unique.
 * Users must ensure that the map indices are unique and in the range [0, n_rows).
 *
 * @tparam matrix_t     Matrix element type
 * @tparam idx_t        Integer type used for indexing
 *
 * @param[in] handle raft handle
 * @param[inout] inout input matrix (n_rows * n_cols)
 * @param[in] map Pointer to the input sequence of scatter locations. The length of the map should
 * be equal to the number of rows in the input matrix. Map indices should be unique and in the range
 * [0, n_rows). The map represents a complete permutation of indices.
 * @param[in] col_batch_size (optional) column batch size. Determines the shape of the scratch space
 * (n_rows, col_batch_size). When set to zero (default), no batching is done and an additional
 * scratch space of shape (n_rows, n_cols) is created.
 */
template <typename matrix_t, typename idx_t>
void scatter(raft::resources const& handle,
             raft::device_matrix_view<matrix_t, idx_t, raft::layout_c_contiguous> inout,
             raft::device_vector_view<const idx_t, idx_t, raft::layout_c_contiguous> map,
             idx_t col_batch_size = 0)
{
  detail::scatter(handle, inout, map, col_batch_size);
}

}  // namespace raft::matrix