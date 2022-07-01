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

#include "../ivf_flat_types.hpp"
#include "ann_kmeans_balanced.cuh"
#include "ann_utils.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/pow2_utils.cuh>

#include <rmm/cuda_stream_view.hpp>

namespace raft::spatial::knn::ivf_flat::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

/**
 * @brief Record the dataset into the index, one source row at a time.
 *
 * The index consists of the dataset rows, grouped by their labels (into clusters/lists).
 * Within each cluster (list), the data is grouped into blocks of `WarpSize` interleaved
 * vectors. Note, the total index length is slightly larger than the dataset length, because
 * each cluster is padded by `WarpSize` elements
 *
 * CUDA launch grid:
 *   X dimension must cover the dataset (n_rows), YZ are not used;
 *   there are no dependencies between threads, hence no constraints on the block size.
 *
 * @tparam T the element type.
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] labels device pointer to the cluster ids for each row [n_rows]
 * @param[in] list_offsets device pointer to the cluster offsets in the output (index) [n_lists]
 * @param[in] dataset device poitner to the input data [n_rows, dim]
 * @param[out] list_data device pointer to the output [index_size, dim]
 * @param[out] list_index device pointer to the source ids corr. to the output [index_size]
 * @param[out] list_sizes_ptr device pointer to the cluster sizes [n_lists];
 *                          it's used as an atomic counter, and must be initialized with zeros.
 * @param n_rows source length
 * @param dim the dimensionality of the data
 * @param veclen size of vectorized loads/stores; must satisfy `dim % veclen == 0`.
 *
 */
template <typename T, typename IdxT>
__global__ void build_index_kernel(const uint32_t* labels,
                                   const IdxT* list_offsets,
                                   const T* dataset,
                                   T* list_data,
                                   IdxT* list_index,
                                   uint32_t* list_sizes_ptr,
                                   IdxT n_rows,
                                   uint32_t dim,
                                   uint32_t veclen)
{
  const IdxT i = IdxT(blockDim.x) * IdxT(blockIdx.x) + threadIdx.x;
  if (i >= n_rows) { return; }

  auto list_id     = labels[i];
  auto inlist_id   = atomicAdd(list_sizes_ptr + list_id, 1);
  auto list_offset = list_offsets[list_id];

  // Record the source vector id in the index
  list_index[list_offset + inlist_id] = i;

  // The data is written in interleaved groups of `index::kGroupSize` vectors
  using interleaved_group = Pow2<kIndexGroupSize>;
  auto group_offset       = interleaved_group::roundDown(inlist_id);
  auto ingroup_id         = interleaved_group::mod(inlist_id) * veclen;

  // Point to the location of the interleaved group of vectors
  list_data += (list_offset + group_offset) * dim;

  // Point to the source vector
  dataset += i * dim;

  // Interleave dimensions of the source vector while recording it.
  // NB: such `veclen` is selected, that `dim % veclen == 0`
  for (uint32_t l = 0; l < dim; l += veclen) {
    for (uint32_t j = 0; j < veclen; j++) {
      list_data[l * kIndexGroupSize + ingroup_id + j] = dataset[l + j];
    }
  }
}

/** See raft::spatial::knn::ivf_flat::build docs */
template <typename T, typename IdxT>
inline auto build(const handle_t& handle,
                  const index_params& params,
                  const T* dataset,
                  IdxT n_rows,
                  uint32_t dim,
                  rmm::cuda_stream_view stream) -> index<T, IdxT>
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("ivf_flat::build(%u, %u)", n_rows, dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "unsupported data type");
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");

  // TODO: consider padding the dimensions and fixing veclen to its maximum possible value as a
  // template parameter (https://github.com/rapidsai/raft/issues/711)
  uint32_t veclen = 16 / sizeof(T);
  while (dim % veclen != 0) {
    veclen = veclen >> 1;
  }
  auto n_lists = static_cast<uint32_t>(params.n_lists);

  // kmeans cluster ids for the dataset
  rmm::device_uvector<uint32_t> labels(n_rows, stream);
  auto&& centers      = make_device_mdarray<float>(stream, n_lists, dim);
  auto&& list_sizes   = make_device_mdarray<uint32_t>(stream, n_lists);
  auto list_sizes_ptr = list_sizes.data();

  // Predict labels of the whole dataset
  kmeans::build_optimized_kmeans(handle,
                                 params.kmeans_n_iters,
                                 dim,
                                 dataset,
                                 n_rows,
                                 labels.data(),
                                 list_sizes_ptr,
                                 centers.data(),
                                 n_lists,
                                 params.kmeans_trainset_fraction,
                                 params.metric,
                                 stream);

  // Calculate offsets into cluster data using exclusive scan
  auto&& list_offsets   = make_device_mdarray<IdxT>(stream, n_lists + 1);
  auto list_offsets_ptr = list_offsets.data();

  thrust::exclusive_scan(
    rmm::exec_policy(stream),
    list_sizes_ptr,
    list_sizes_ptr + n_lists + 1,
    list_offsets_ptr,
    IdxT(0),
    [] __device__(IdxT s, uint32_t l) { return s + Pow2<WarpSize>::roundUp(l); });

  IdxT index_size;
  update_host(&index_size, list_offsets_ptr + n_lists, 1, stream);
  handle.sync_stream(stream);

  auto&& data    = make_device_mdarray<T>(stream, index_size, dim);
  auto&& indices = make_device_mdarray<IdxT>(stream, index_size);

  // we'll rebuild the `list_sizes_ptr` in the following kernel, using it as an atomic counter.
  utils::memset(list_sizes_ptr, 0, sizeof(uint32_t) * n_lists, stream);

  const dim3 block_dim(256);
  const dim3 grid_dim(raft::ceildiv<IdxT>(n_rows, block_dim.x));
  build_index_kernel<<<grid_dim, block_dim, 0, stream>>>(labels.data(),
                                                         list_offsets_ptr,
                                                         dataset,
                                                         data.data(),
                                                         indices.data(),
                                                         list_sizes_ptr,
                                                         n_rows,
                                                         dim,
                                                         veclen);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Precompute the centers vector norms for L2Expanded distance
  auto compute_norms = [&]() {
    auto&& r = make_device_mdarray<float>(stream, n_lists);
    utils::dots_along_rows(n_lists, dim, centers.data(), r.data(), stream);
    RAFT_LOG_TRACE_VEC(r.data(), 20);
    return r;
  };
  auto&& center_norms = params.metric == raft::distance::DistanceType::L2Expanded
                          ? std::optional(compute_norms())
                          : std::nullopt;

  // assemble the index
  index<T, IdxT> index{{},
                       veclen,
                       params.metric,
                       std::move(data),
                       std::move(indices),
                       std::move(list_sizes),
                       std::move(list_offsets),
                       std::move(centers),
                       std::move(center_norms)};

  // check index invariants
  index.check_consistency();

  return index;
}

}  // namespace raft::spatial::knn::ivf_flat::detail
