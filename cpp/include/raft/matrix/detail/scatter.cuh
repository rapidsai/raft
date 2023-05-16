/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/fast_int_div.cuh>
#include <thrust/iterator/counting_iterator.h>

namespace raft {
namespace matrix {
namespace detail {

/**
 * In-place scatter elements in a row-major matrix according to a
 * map. The length of the map is equal to the number of rows. The
 * map specifies the destination index for each row, i.e. in the
 * resulting matrix, row[map[i]] would be row[i]. Batching is done on
 * columns and an additional scratch space of shape n_rows * cols_batch_size
 * is created. For each batch, chunks of columns from each row are copied
 * into the appropriate location in the scratch space and copied back to
 * the corresponding locations in the input matrix.
 * @tparam InputIteratorT
 * @tparam MapIteratorT
 * @tparam IndexT
 *
 * @param[in] handle raft handle
 * @param[inout] inout input matrix (n_rows * n_cols)
 * @param[in] map map containing the destination index for each row (n_rows)
 * @param[in] batch_size column batch size
 */
template <typename InputIteratorT, typename MapIteratorT, typename IndexT>
void scatter(raft::device_resources const& handle,
             raft::device_matrix_view<InputIteratorT, IndexT, raft::layout_c_contiguous> inout,
             raft::device_vector_view<const MapIteratorT, IndexT, raft::layout_c_contiguous> map,
             IndexT batch_size)
{
  IndexT m = in.extent(0);
  IndexT n = in.extent(1);

  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  IndexT n_batches = raft::ceildiv(n, batch_size);

  for (IndexT bid = 0; bid < n_batches; bid++) {
    IndexT batch_offset   = bid * batch_size;
    IndexT cols_per_batch = min(batch_size, n - batch_offset);
    auto scratch_space =
      raft::make_device_vector<InputIteratorT, IndexT>(handle, m * cols_per_batch);

    auto scatter_op = [in  = in.data_handle(),
                       map = map.data_handle(),
                       batch_offset,
                       cols_per_batch = raft::util::FastIntDiv(cols_per_batch),
                       n] __device__(auto idx) {
      IndexT row = idx / cols_per_batch;
      IndexT col = idx % cols_per_batch;
      return in[row * n + batch_offset + col];
    };
    raft::linalg::map_offset(handle, scratch_space.view(), scatter_op);
    auto copy_op = [in            = in.data_handle(),
                    map           = map.data_handle(),
                    scratch_space = scratch_space.data_handle(),
                    batch_offset,
                    cols_per_batch = raft::util::FastIntDiv(cols_per_batch),
                    n] __device__(auto idx) {
      IndexT row                            = idx / cols_per_batch;
      IndexT col                            = idx % cols_per_batch;
      in[map[row] * n + batch_offset + col] = scratch_space[idx];
    };
    auto counting = thrust::make_counting_iterator<IndexT>(0);
    thrust::for_each(exec_policy, counting, counting + m * batch_size, copy_op);
  }
}

}  // end namespace detail
}  // end namespace matrix
}  // end namespace raft