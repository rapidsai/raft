/**
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
#include <raft/core/device_resources.hpp>
#include <raft/matrix/detail/scatter.cuh>

namespace raft::matrix {
/**
 * In-place scatter elements in a row-major matrix according to a
 * map. The length of the map is equal to the number of rows. The
 * map specifies the destination index for each row, i.e. in the
 * resulting matrix, row[map[i]] would be row[i]. Batching is done on
 * columns and an additional scratch space of shape n_rows * cols_batch_size
 * is created. For each batch, chunks of columns from each row are copied
 * into the appropriate location in the scratch space and copied back to
 * the corresponding locations in the input matrix.
 *
 * @tparam matrix_t
 * @tparam map_t
 * @tparam idx_t
 *
 * @param[in] handle raft handle
 * @param[inout] inout input matrix (n_rows * n_cols)
 * @param[in] map map containing the order in which rows are to be rearranged (n_rows)
 * @param[in] col_batch_size column batch size
 */
template <typename matrix_t, typename map_t, typename idx_t>
void scatter(raft::resources const& handle,
             raft::device_matrix_view<matrix_t, idx_t, raft::layout_c_contiguous> inout,
             raft::device_vector_view<map_t, idx_t, raft::layout_c_contiguous> map,
             idx_t col_batch_size)
{
  idx_t m       = inout.extent(0);
  idx_t n       = inout.extent(1);
  idx_t map_len = map.extent(0);
  RAFT_EXPECTS(0 < col_batch_size && col_batch_size <= n, "col_batch_size should be > 0 and <= n");
  RAFT_EXPECTS(map_len == m, "size of map should be equal to the number of rows in input matrix");

  detail::scatter(handle, inout, map, col_batch_size);
}

}  // namespace raft::matrix