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
 * @tparam IdxT type of the indices in the source source_vecs
 *
 * @param[in] labels device pointer to the cluster ids for each row [n_rows]
 * @param[in] list_offsets device pointer to the cluster offsets in the output (index) [n_lists]
 * @param[in] source_vecs device poitner to the input data [n_rows, dim]
 * @param[in] source_ixs device poitner to the input indices [n_rows]
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
                                   const T* source_vecs,
                                   const IdxT* source_ixs,
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
  list_index[list_offset + inlist_id] = source_ixs == nullptr ? i : source_ixs[i];

  // The data is written in interleaved groups of `index::kGroupSize` vectors
  using interleaved_group = Pow2<kIndexGroupSize>;
  auto group_offset       = interleaved_group::roundDown(inlist_id);
  auto ingroup_id         = interleaved_group::mod(inlist_id) * veclen;

  // Point to the location of the interleaved group of vectors
  list_data += (list_offset + group_offset) * dim;

  // Point to the source vector
  source_vecs += i * dim;

  // Interleave dimensions of the source vector while recording it.
  // NB: such `veclen` is selected, that `dim % veclen == 0`
  for (uint32_t l = 0; l < dim; l += veclen) {
    for (uint32_t j = 0; j < veclen; j++) {
      list_data[l * kIndexGroupSize + ingroup_id + j] = source_vecs[l + j];
    }
  }
}

/** See raft::spatial::knn::ivf_flat::extend docs */
template <typename T, typename IdxT>
inline auto extend(const handle_t& handle,
                   const index<T, IdxT>& orig_index,
                   const T* new_vectors,
                   const IdxT* new_indices,
                   IdxT n_rows,
                   rmm::cuda_stream_view stream) -> index<T, IdxT>
{
  auto n_lists = orig_index.n_lists;
  auto dim     = orig_index.dim;
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_flat::extend(%zu, %u)", size_t(n_rows), dim);

  RAFT_EXPECTS(new_indices != nullptr || orig_index.size == 0,
               "You must pass data indices when the index is non-empty.");

  rmm::device_uvector<uint32_t> new_labels(n_rows, stream);
  kmeans::predict(handle,
                  orig_index.centers.data(),
                  n_lists,
                  dim,
                  new_vectors,
                  n_rows,
                  new_labels.data(),
                  orig_index.metric,
                  stream);

  auto&& list_sizes     = rmm::device_uvector<uint32_t>(n_lists, stream);
  auto&& list_offsets   = rmm::device_uvector<IdxT>(n_lists + 1, stream);
  auto list_sizes_ptr   = list_sizes.data();
  auto list_offsets_ptr = list_offsets.data();

  auto&& centers   = rmm::device_uvector<float>(size_t(n_lists) * size_t(dim), stream);
  auto centers_ptr = centers.data();

  // Calculate the centers and sizes on the new data, starting from the original values
  raft::copy(centers_ptr, orig_index.centers.data(), centers.size(), stream);
  raft::copy(list_sizes_ptr, orig_index.list_sizes.data(), list_sizes.size(), stream);

  kmeans::calc_centers_and_sizes(centers_ptr,
                                 list_sizes_ptr,
                                 n_lists,
                                 dim,
                                 new_vectors,
                                 n_rows,
                                 new_labels.data(),
                                 false,
                                 stream);

  // Calculate new offsets
  IdxT index_size = 0;
  update_device(list_offsets_ptr, &index_size, 1, stream);
  thrust::inclusive_scan(
    rmm::exec_policy(stream),
    list_sizes_ptr,
    list_sizes_ptr + n_lists,
    list_offsets_ptr + 1,
    [] __device__(IdxT s, uint32_t l) { return s + Pow2<WarpSize>::roundUp(l); });
  update_host(&index_size, list_offsets_ptr + n_lists, 1, stream);
  handle.sync_stream(stream);

  auto&& data    = rmm::device_uvector<T>(index_size * IdxT(dim), stream);
  auto&& indices = rmm::device_uvector<IdxT>(index_size, stream);

  // Populate index with the old data
  if (orig_index.size > 0) {
    utils::block_copy(orig_index.list_offsets.data(),
                      list_offsets_ptr,
                      IdxT(n_lists),
                      orig_index.data.data(),
                      data.data(),
                      IdxT(dim),
                      stream);

    utils::block_copy(orig_index.list_offsets.data(),
                      list_offsets_ptr,
                      IdxT(n_lists),
                      orig_index.indices.data(),
                      indices.data(),
                      IdxT(1),
                      stream);
  }

  // Copy the old sizes, so we can start from the current state of the index;
  // we'll rebuild the `list_sizes_ptr` in the following kernel, using it as an atomic counter.
  raft::copy(list_sizes_ptr, orig_index.list_sizes.data(), list_sizes.size(), stream);

  const dim3 block_dim(256);
  const dim3 grid_dim(raft::ceildiv<IdxT>(n_rows, block_dim.x));
  build_index_kernel<<<grid_dim, block_dim, 0, stream>>>(new_labels.data(),
                                                         list_offsets_ptr,
                                                         new_vectors,
                                                         new_indices,
                                                         data.data(),
                                                         indices.data(),
                                                         list_sizes_ptr,
                                                         n_rows,
                                                         dim,
                                                         orig_index.veclen);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Precompute the centers vector norms for L2Expanded distance
  auto compute_norms = [&]() {
    auto&& r = rmm::device_uvector<float>(n_lists, stream);
    utils::dots_along_rows(n_lists, dim, centers.data(), r.data(), stream);
    RAFT_LOG_TRACE_VEC(r.data(), 20);
    return std::move(r);
  };
  auto&& center_norms = orig_index.metric == raft::distance::DistanceType::L2Expanded
                          ? std::optional(compute_norms())
                          : std::nullopt;

  // assemble the index
  index<T, IdxT> new_index{{},
                           orig_index.veclen,
                           orig_index.metric,
                           index_size,
                           orig_index.dim,
                           orig_index.n_lists,
                           std::move(data),
                           std::move(indices),
                           std::move(list_sizes),
                           std::move(list_offsets),
                           std::move(centers),
                           std::move(center_norms)};

  // check index invariants
  new_index.check_consistency();

  return new_index;
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
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_flat::build(%zu, %u)", size_t(n_rows), dim);
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
  auto&& centers = rmm::device_uvector<float>(size_t(n_lists) * size_t(dim), stream);

  // Train the kmeans clustering
  {
    auto trainset_ratio = std::max<size_t>(
      1, n_rows / std::max<size_t>(params.kmeans_trainset_fraction * n_rows, n_lists));
    auto n_rows_train = n_rows / trainset_ratio;
    rmm::device_uvector<T> trainset(n_rows_train * dim, stream);
    // TODO: a proper sampling
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(trainset.data(),
                                    sizeof(T) * dim,
                                    dataset,
                                    sizeof(T) * dim * trainset_ratio,
                                    sizeof(T) * dim,
                                    n_rows_train,
                                    cudaMemcpyDefault,
                                    stream));
    kmeans::build_hierarchical(handle,
                               params.kmeans_n_iters,
                               dim,
                               trainset.data(),
                               n_rows_train,
                               centers.data(),
                               n_lists,
                               params.metric,
                               stream);
  }

  auto&& data         = rmm::device_uvector<T>(0, stream);
  auto&& indices      = rmm::device_uvector<IdxT>(0, stream);
  auto&& list_sizes   = rmm::device_uvector<uint32_t>(n_lists, stream);
  auto&& list_offsets = rmm::device_uvector<IdxT>(n_lists + 1, stream);
  utils::memzero(list_sizes.data(), list_sizes.size(), stream);
  utils::memzero(list_offsets.data(), list_offsets.size(), stream);

  // assemble the index
  index<T, IdxT> index{{},
                       veclen,
                       params.metric,
                       IdxT(0),
                       dim,
                       n_lists,
                       std::move(data),
                       std::move(indices),
                       std::move(list_sizes),
                       std::move(list_offsets),
                       std::move(centers),
                       std::nullopt};

  // check index invariants
  index.check_consistency();

  // add the data if necessary
  if (params.add_data_on_build) {
    return extend<T, IdxT>(handle, index, dataset, nullptr, n_rows, stream);
  } else {
    return index;
  }
}

}  // namespace raft::spatial::knn::ivf_flat::detail
