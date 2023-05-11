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
#include <cub/cub.cuh>

#include <raft/core/error.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/distance/masked_nn.cuh>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/op/reduce.cuh>

#include <raft/util/cudart_utils.hpp>
#include <raft/util/fast_int_div.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <raft/core/kvp.hpp>
#include <raft/core/nvtx.hpp>

namespace raft {
namespace matrix {
namespace detail {

/**
 * In-place gather elements in a row-major matrix according to a
 * map. The length of the map is equal to the number of rows.
 * Batching is done on columns and an additional scratch space of
 * shape n_rows * cols_batch_size is created. For each batch, chunks
 * of columns from each row are copied into the appropriate location
 * in the scratch space and copied back to the corresponding locations
 * in the input matrix.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle
 * @param[out] in input matrix (n_rows * n_cols)
 * @param[in] map map containing the order in which rows are to be rearranged (n_rows)
 * @param  D Number of columns of the input/output matrices
 * @param  N Number of rows of the input matrix
 * @param batch_size column batch size
 */
template <typename T, typename IdxT>
void batched_gather(raft::device_resources const& handle,
                    T* in,
                    IdxT* map,
                    size_t D,
                    size_t N,
                    size_t batch_size)
{
  auto exec_policy = handle.get_thrust_policy();
  size_t n_batches = raft::ceildiv(D, batch_size);
  for (size_t bid = 0; bid < n_batches; bid++) {
    size_t batch_offset   = bid * batch_size;
    size_t cols_per_batch = min(batch_size, D - (size_t)batch_offset);
    auto scratch_space = raft::make_device_vector<T, IdxT>(handle, N * cols_per_batch);

    auto scatter_op =
      [in, map, batch_offset, cols_per_batch = raft::util::FastIntDiv(cols_per_batch), D] __device__(
        auto idx) {
        IdxT row = idx / cols_per_batch;
        IdxT col = idx % cols_per_batch;
        return in[map[row] * D + batch_offset + col];
      };
    raft::linalg::map_offset(handle, scratch_space.view(), scatter_op);
    auto copy_op = [in,
                    map,
                    scratch_space = scratch_space.data_handle(),
                    batch_offset,
                    cols_per_batch = raft::util::FastIntDiv(cols_per_batch),
                    D] __device__(auto idx) {
      IdxT row                          = idx / cols_per_batch;
      IdxT col                          = idx % cols_per_batch;
      return in[row * D + batch_offset + col] = scratch_space[idx];
    };
    auto counting = thrust::make_counting_iterator<IdxT>(0);
    thrust::for_each(exec_policy, counting, counting + N * batch_size, copy_op);
    }
}

/**
 * In-place scatter elements in a row-major matrix according to a
 * map. The length of the map is equal to the number of rows.
 * Batching is done on columns and an additional scratch space of
 * shape n_rows * cols_batch_size is created. For each batch, chunks
 * of columns from each row are copied into the appropriate location
 * in the scratch space and copied back to the corresponding locations
 * in the input matrix.
 * @tparam T
 * @tparam IdxT
 * @param[in] handle raft handle
 * @param[out] in input matrix (n_rows * n_cols)
 * @param[in] map map containing the destination index for each row (n_rows)
 * @param  D Number of columns of the input/output matrices
 * @param  N Number of rows of the input matrix
 * @param batch_size column batch size
 */
template <typename T, typename IdxT>
void batched_scatter(raft::device_resources const& handle,
                     T* in,
                     IdxT* map,
                     size_t D,
                     size_t N,
                     size_t batch_size)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  size_t n_batches = raft::ceildiv(D, batch_size);

  for (size_t bid = 0; bid < n_batches; bid++) {
    size_t batch_offset   = bid * batch_size;
    size_t cols_per_batch = min(batch_size, D - batch_offset);
    auto scratch_space = raft::make_device_vector<T, IdxT>(handle, N * cols_per_batch);

    auto scatter_op =
      [in, map, batch_offset, cols_per_batch = raft::util::FastIntDiv(cols_per_batch), D] __device__(
        auto idx) {
        IdxT row = idx / cols_per_batch;
        IdxT col = idx % cols_per_batch;
        return in[row * D + batch_offset + col];
      };
    raft::linalg::map_offset(handle, scratch_space.view(), scatter_op);
    auto copy_op = [in,
                    map,
                    scratch_space = scratch_space.data_handle(),
                    batch_offset,
                    cols_per_batch = raft::util::FastIntDiv(cols_per_batch),
                    D] __device__(auto idx) {
      IdxT row                        = idx / cols_per_batch;
      IdxT col                        = idx % cols_per_batch;
      in[map[row] * D + batch_offset + col] = scratch_space[idx];
    };
    auto counting = thrust::make_counting_iterator<IdxT>(0);
    thrust::for_each(exec_policy, counting, counting + N * batch_size, copy_op);
  }
}

};  // end namespace detail
};  // end namespace matrix
};  // end namespace raft