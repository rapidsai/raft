/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/cluster/kmeans_balanced.cuh>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/neighbors/detail/ivf_common.cuh>
#include <raft/neighbors/ivf_flat_codepacker.hpp>
#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/neighbors/ivf_list.hpp>
#include <raft/neighbors/ivf_list_types.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <raft/stats/histogram.cuh>
#include <raft/util/pow2_utils.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace raft::neighbors::ivf_flat::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

template <typename T, typename IdxT>
auto clone(const raft::resources& res, const index<T, IdxT>& source) -> index<T, IdxT>
{
  auto stream = resource::get_cuda_stream(res);

  // Allocate the new index
  index<T, IdxT> target(res,
                        source.metric(),
                        source.n_lists(),
                        source.adaptive_centers(),
                        source.conservative_memory_allocation(),
                        source.dim());

  // Copy the independent parts
  copy(target.list_sizes().data_handle(),
       source.list_sizes().data_handle(),
       source.list_sizes().size(),
       stream);
  copy(target.centers().data_handle(),
       source.centers().data_handle(),
       source.centers().size(),
       stream);
  if (source.center_norms().has_value()) {
    target.allocate_center_norms(res);
    copy(target.center_norms()->data_handle(),
         source.center_norms()->data_handle(),
         source.center_norms()->size(),
         stream);
  }
  // Copy shared pointers
  target.lists() = source.lists();

  // Make sure the device pointers point to the new lists
  ivf::detail::recompute_internal_state(res, target);

  return target;
}

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
 * @tparam T      element type.
 * @tparam IdxT   type of the indices in the source source_vecs
 * @tparam LabelT label type
 * @tparam gather_src if false, then we build the index from vectors source_vecs[i,:], otherwise
 *     we use source_vecs[source_ixs[i],:]. In both cases i=0..n_rows-1.
 *
 * @param[in] labels device pointer to the cluster ids for each row [n_rows]
 * @param[in] source_vecs device pointer to the input data [n_rows, dim]
 * @param[in] source_ixs device pointer to the input indices [n_rows]
 * @param[out] list_data_ptrs device pointer to the index data of size [n_lists][index_size, dim]
 * @param[out] list_index_ptrs device pointer to the source ids corr. to the output [n_lists]
 * [index_size]
 * @param[out] list_sizes_ptr device pointer to the cluster sizes [n_lists];
 *                          it's used as an atomic counter, and must be initialized with zeros.
 * @param n_rows source length
 * @param dim the dimensionality of the data
 * @param veclen size of vectorized loads/stores; must satisfy `dim % veclen == 0`.
 *
 */
template <typename T, typename IdxT, typename LabelT, bool gather_src = false>
RAFT_KERNEL build_index_kernel(const LabelT* labels,
                               const T* source_vecs,
                               const IdxT* source_ixs,
                               T** list_data_ptrs,
                               IdxT** list_index_ptrs,
                               uint32_t* list_sizes_ptr,
                               IdxT n_rows,
                               uint32_t dim,
                               uint32_t veclen,
                               IdxT batch_offset = 0)
{
  const IdxT i = IdxT(blockDim.x) * IdxT(blockIdx.x) + threadIdx.x;
  if (i >= n_rows) { return; }

  auto list_id     = labels[i];
  auto inlist_id   = atomicAdd(list_sizes_ptr + list_id, 1);
  auto* list_index = list_index_ptrs[list_id];
  auto* list_data  = list_data_ptrs[list_id];

  // Record the source vector id in the index
  list_index[inlist_id] = source_ixs == nullptr ? i + batch_offset : source_ixs[i];

  // The data is written in interleaved groups of `index::kGroupSize` vectors
  using interleaved_group = Pow2<kIndexGroupSize>;
  auto group_offset       = interleaved_group::roundDown(inlist_id);
  auto ingroup_id         = interleaved_group::mod(inlist_id) * veclen;

  // Point to the location of the interleaved group of vectors
  list_data += group_offset * dim;

  // Point to the source vector
  if constexpr (gather_src) {
    source_vecs += source_ixs[i] * dim;
  } else {
    source_vecs += i * dim;
  }
  // Interleave dimensions of the source vector while recording it.
  // NB: such `veclen` is selected, that `dim % veclen == 0`
  for (uint32_t l = 0; l < dim; l += veclen) {
    for (uint32_t j = 0; j < veclen; j++) {
      list_data[l * kIndexGroupSize + ingroup_id + j] = source_vecs[l + j];
    }
  }
}

/** See raft::neighbors::ivf_flat::extend docs */
template <typename T, typename IdxT>
void extend(raft::resources const& handle,
            index<T, IdxT>* index,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows)
{
  using LabelT = uint32_t;
  RAFT_EXPECTS(index != nullptr, "index cannot be empty.");

  auto stream  = resource::get_cuda_stream(handle);
  auto n_lists = index->n_lists();
  auto dim     = index->dim();
  list_spec<uint32_t, T, IdxT> list_device_spec{index->dim(),
                                                index->conservative_memory_allocation()};
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_flat::extend(%zu, %u)", size_t(n_rows), dim);

  RAFT_EXPECTS(new_indices != nullptr || index->size() == 0,
               "You must pass data indices when the index is non-empty.");

  auto new_labels = raft::make_device_vector<LabelT, IdxT>(handle, n_rows);
  raft::cluster::kmeans_balanced_params kmeans_params;
  kmeans_params.metric = index->metric();
  auto orig_centroids_view =
    raft::make_device_matrix_view<const float, IdxT>(index->centers().data_handle(), n_lists, dim);
  // Calculate the batch size for the input data if it's not accessible directly from the device
  constexpr size_t kReasonableMaxBatchSize = 65536;
  size_t max_batch_size                    = std::min<size_t>(n_rows, kReasonableMaxBatchSize);

  // Predict the cluster labels for the new data, in batches if necessary
  utils::batch_load_iterator<T> vec_batches(new_vectors,
                                            n_rows,
                                            index->dim(),
                                            max_batch_size,
                                            stream,
                                            resource::get_workspace_resource(handle));

  for (const auto& batch : vec_batches) {
    auto batch_data_view =
      raft::make_device_matrix_view<const T, IdxT>(batch.data(), batch.size(), index->dim());
    auto batch_labels_view = raft::make_device_vector_view<LabelT, IdxT>(
      new_labels.data_handle() + batch.offset(), batch.size());
    raft::cluster::kmeans_balanced::predict(handle,
                                            kmeans_params,
                                            batch_data_view,
                                            orig_centroids_view,
                                            batch_labels_view,
                                            utils::mapping<float>{});
  }

  auto* list_sizes_ptr    = index->list_sizes().data_handle();
  auto old_list_sizes_dev = raft::make_device_vector<uint32_t, IdxT>(handle, n_lists);
  copy(old_list_sizes_dev.data_handle(), list_sizes_ptr, n_lists, stream);

  // Calculate the centers and sizes on the new data, starting from the original values
  if (index->adaptive_centers()) {
    auto centroids_view = raft::make_device_matrix_view<float, IdxT>(
      index->centers().data_handle(), index->centers().extent(0), index->centers().extent(1));
    auto list_sizes_view =
      raft::make_device_vector_view<std::remove_pointer_t<decltype(list_sizes_ptr)>, IdxT>(
        list_sizes_ptr, n_lists);
    for (const auto& batch : vec_batches) {
      auto batch_data_view =
        raft::make_device_matrix_view<const T, IdxT>(batch.data(), batch.size(), index->dim());
      auto batch_labels_view = raft::make_device_vector_view<const LabelT, IdxT>(
        new_labels.data_handle() + batch.offset(), batch.size());
      raft::cluster::kmeans_balanced::helpers::calc_centers_and_sizes(handle,
                                                                      batch_data_view,
                                                                      batch_labels_view,
                                                                      centroids_view,
                                                                      list_sizes_view,
                                                                      false,
                                                                      utils::mapping<float>{});
    }
  } else {
    raft::stats::histogram<uint32_t, IdxT>(raft::stats::HistTypeAuto,
                                           reinterpret_cast<int32_t*>(list_sizes_ptr),
                                           IdxT(n_lists),
                                           new_labels.data_handle(),
                                           n_rows,
                                           1,
                                           stream);
    raft::linalg::add(
      list_sizes_ptr, list_sizes_ptr, old_list_sizes_dev.data_handle(), n_lists, stream);
  }

  // Calculate and allocate new list data
  std::vector<uint32_t> new_list_sizes(n_lists);
  std::vector<uint32_t> old_list_sizes(n_lists);
  {
    copy(old_list_sizes.data(), old_list_sizes_dev.data_handle(), n_lists, stream);
    copy(new_list_sizes.data(), list_sizes_ptr, n_lists, stream);
    resource::sync_stream(handle);
    auto& lists = index->lists();
    for (uint32_t label = 0; label < n_lists; label++) {
      ivf::resize_list(handle,
                       lists[label],
                       list_device_spec,
                       new_list_sizes[label],
                       Pow2<kIndexGroupSize>::roundUp(old_list_sizes[label]));
    }
  }
  // Update the pointers and the sizes
  ivf::detail::recompute_internal_state(handle, *index);
  // Copy the old sizes, so we can start from the current state of the index;
  // we'll rebuild the `list_sizes_ptr` in the following kernel, using it as an atomic counter.
  raft::copy(list_sizes_ptr, old_list_sizes_dev.data_handle(), n_lists, stream);

  utils::batch_load_iterator<IdxT> vec_indices(
    new_indices, n_rows, 1, max_batch_size, stream, resource::get_workspace_resource(handle));
  utils::batch_load_iterator<IdxT> idx_batch = vec_indices.begin();
  size_t next_report_offset                  = 0;
  size_t d_report_offset                     = n_rows * 5 / 100;
  for (const auto& batch : vec_batches) {
    auto batch_data_view =
      raft::make_device_matrix_view<const T, IdxT>(batch.data(), batch.size(), index->dim());
    // Kernel to insert the new vectors
    const dim3 block_dim(256);
    const dim3 grid_dim(raft::ceildiv<IdxT>(batch.size(), block_dim.x));
    build_index_kernel<T, IdxT, LabelT>
      <<<grid_dim, block_dim, 0, stream>>>(new_labels.data_handle() + batch.offset(),
                                           batch_data_view.data_handle(),
                                           idx_batch->data(),
                                           index->data_ptrs().data_handle(),
                                           index->inds_ptrs().data_handle(),
                                           list_sizes_ptr,
                                           batch.size(),
                                           dim,
                                           index->veclen(),
                                           batch.offset());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    if (batch.offset() > next_report_offset) {
      float progress = batch.offset() * 100.0f / n_rows;
      RAFT_LOG_DEBUG("ivf_flat::extend added vectors %zu, %6.1f%% complete",
                     static_cast<size_t>(batch.offset()),
                     progress);
      next_report_offset += d_report_offset;
    }
    ++idx_batch;
  }
  // Precompute the centers vector norms for L2Expanded distance
  if (!index->center_norms().has_value()) {
    index->allocate_center_norms(handle);
    if (index->center_norms().has_value()) {
      raft::linalg::rowNorm(index->center_norms()->data_handle(),
                            index->centers().data_handle(),
                            dim,
                            n_lists,
                            raft::linalg::L2Norm,
                            true,
                            stream);
      RAFT_LOG_TRACE_VEC(index->center_norms()->data_handle(), std::min<uint32_t>(dim, 20));
    }
  } else if (index->center_norms().has_value() && index->adaptive_centers()) {
    raft::linalg::rowNorm(index->center_norms()->data_handle(),
                          index->centers().data_handle(),
                          dim,
                          n_lists,
                          raft::linalg::L2Norm,
                          true,
                          stream);
    RAFT_LOG_TRACE_VEC(index->center_norms()->data_handle(), std::min<uint32_t>(dim, 20));
  }
}

/** See raft::neighbors::ivf_flat::extend docs */
template <typename T, typename IdxT>
auto extend(raft::resources const& handle,
            const index<T, IdxT>& orig_index,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) -> index<T, IdxT>
{
  auto ext_index = clone(handle, orig_index);
  detail::extend(handle, &ext_index, new_vectors, new_indices, n_rows);
  return ext_index;
}

/** See raft::neighbors::ivf_flat::build docs */
template <typename T, typename IdxT>
inline auto build(raft::resources const& handle,
                  const index_params& params,
                  const T* dataset,
                  IdxT n_rows,
                  uint32_t dim) -> index<T, IdxT>
{
  auto stream = resource::get_cuda_stream(handle);
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_flat::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "unsupported data type");
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");
  RAFT_EXPECTS(n_rows >= params.n_lists, "number of rows can't be less than n_lists");

  index<T, IdxT> index(handle, params, dim);
  utils::memzero(
    index.accum_sorted_sizes().data_handle(), index.accum_sorted_sizes().size(), stream);
  utils::memzero(index.list_sizes().data_handle(), index.list_sizes().size(), stream);
  utils::memzero(index.data_ptrs().data_handle(), index.data_ptrs().size(), stream);
  utils::memzero(index.inds_ptrs().data_handle(), index.inds_ptrs().size(), stream);

  // Train the kmeans clustering
  {
    auto trainset_ratio = std::max<size_t>(
      1, n_rows / std::max<size_t>(params.kmeans_trainset_fraction * n_rows, index.n_lists()));
    auto n_rows_train = n_rows / trainset_ratio;
    rmm::device_uvector<T> trainset(n_rows_train * index.dim(), stream);
    // TODO: a proper sampling
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(trainset.data(),
                                    sizeof(T) * index.dim(),
                                    dataset,
                                    sizeof(T) * index.dim() * trainset_ratio,
                                    sizeof(T) * index.dim(),
                                    n_rows_train,
                                    cudaMemcpyDefault,
                                    stream));
    auto trainset_const_view =
      raft::make_device_matrix_view<const T, IdxT>(trainset.data(), n_rows_train, index.dim());
    auto centers_view = raft::make_device_matrix_view<float, IdxT>(
      index.centers().data_handle(), index.n_lists(), index.dim());
    raft::cluster::kmeans_balanced_params kmeans_params;
    kmeans_params.n_iters = params.kmeans_n_iters;
    kmeans_params.metric  = index.metric();
    raft::cluster::kmeans_balanced::fit(
      handle, kmeans_params, trainset_const_view, centers_view, utils::mapping<float>{});
  }

  // add the data if necessary
  if (params.add_data_on_build) {
    detail::extend<T, IdxT>(handle, &index, dataset, nullptr, n_rows);
  }
  return index;
}

/**
 * Build an index that can be used in refinement operation.
 *
 * See raft::neighbors::refine for details on the refinement operation.
 *
 * The returned index cannot be used for a regular ivf_flat::search. The index misses information
 * about coarse clusters. Instead, the neighbor candidates are assumed to form clusters, one for
 * each query. The candidate vectors are gathered into the index dataset, that can be later used
 * in ivfflat_interleaved_scan.
 *
 * @param[in] handle the raft handle
 * @param[inout] refinement_index
 * @param[in] dataset device pointer to dataset vectors, size [n_rows, dim]. Note that n_rows is
 *   not known to this function, but each candidate_idx has to be smaller than n_rows.
 * @param[in] candidate_idx device pointer to neighbor candidates, size [n_queries, n_candidates]
 * @param[in] n_candidates  of neighbor_candidates
 */
template <typename T, typename IdxT>
inline void fill_refinement_index(raft::resources const& handle,
                                  index<T, IdxT>* refinement_index,
                                  const T* dataset,
                                  const IdxT* candidate_idx,
                                  IdxT n_queries,
                                  uint32_t n_candidates)
{
  using LabelT = uint32_t;

  auto stream      = resource::get_cuda_stream(handle);
  uint32_t n_lists = n_queries;
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_flat::fill_refinement_index(%zu, %u)", size_t(n_queries));

  rmm::device_uvector<LabelT> new_labels(n_queries * n_candidates, stream);
  auto new_labels_view =
    raft::make_device_vector_view<LabelT, IdxT>(new_labels.data(), n_queries * n_candidates);
  linalg::map_offset(
    handle,
    new_labels_view,
    raft::compose_op(raft::cast_op<LabelT>(), raft::div_const_op<IdxT>(n_candidates)));

  auto list_sizes_ptr = refinement_index->list_sizes().data_handle();
  // We do not fill centers and center norms, since we will not run coarse search.

  // Allocate new memory
  auto& lists = refinement_index->lists();
  list_spec<uint32_t, T, IdxT> list_device_spec{refinement_index->dim(), false};
  for (uint32_t label = 0; label < n_lists; label++) {
    ivf::resize_list(handle, lists[label], list_device_spec, n_candidates, uint32_t(0));
  }
  // Update the pointers and the sizes
  ivf::detail::recompute_internal_state(handle, *refinement_index);

  RAFT_CUDA_TRY(cudaMemsetAsync(list_sizes_ptr, 0, n_lists * sizeof(uint32_t), stream));

  const dim3 block_dim(256);
  const dim3 grid_dim(raft::ceildiv<IdxT>(n_queries * n_candidates, block_dim.x));
  build_index_kernel<T, IdxT, LabelT, true>
    <<<grid_dim, block_dim, 0, stream>>>(new_labels.data(),
                                         dataset,
                                         candidate_idx,
                                         refinement_index->data_ptrs().data_handle(),
                                         refinement_index->inds_ptrs().data_handle(),
                                         list_sizes_ptr,
                                         n_queries * n_candidates,
                                         refinement_index->dim(),
                                         refinement_index->veclen());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
RAFT_KERNEL pack_interleaved_list_kernel(const T* codes,
                                         T* list_data,
                                         uint32_t n_rows,
                                         uint32_t dim,
                                         uint32_t veclen,
                                         std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  uint32_t tid          = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t dst_ix = std::holds_alternative<uint32_t>(offset_or_indices)
                            ? std::get<uint32_t>(offset_or_indices) + tid
                            : std::get<const uint32_t*>(offset_or_indices)[tid];
  if (tid < n_rows) { codepacker::pack_1(codes + tid * dim, list_data, dim, veclen, dst_ix); }
}

template <typename T>
RAFT_KERNEL unpack_interleaved_list_kernel(
  const T* list_data,
  T* codes,
  uint32_t n_rows,
  uint32_t dim,
  uint32_t veclen,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  uint32_t tid          = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t src_ix = std::holds_alternative<uint32_t>(offset_or_indices)
                            ? std::get<uint32_t>(offset_or_indices) + tid
                            : std::get<const uint32_t*>(offset_or_indices)[tid];
  if (tid < n_rows) { codepacker::unpack_1(list_data, codes + tid * dim, dim, veclen, src_ix); }
}

template <typename T, typename IdxT>
void pack_list_data(
  raft::resources const& res,
  device_matrix_view<const T, uint32_t, row_major> codes,
  uint32_t veclen,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  device_mdspan<T, typename list_spec<uint32_t, T, IdxT>::list_extents, row_major> list_data)
{
  uint32_t n_rows = codes.extent(0);
  uint32_t dim    = codes.extent(1);
  if (n_rows == 0 || dim == 0) return;
  static constexpr uint32_t kBlockSize = 256;
  dim3 blocks(div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto stream = resource::get_cuda_stream(res);
  pack_interleaved_list_kernel<<<blocks, threads, 0, stream>>>(
    codes.data_handle(), list_data.data_handle(), n_rows, dim, veclen, offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T, typename IdxT>
void unpack_list_data(
  raft::resources const& res,
  device_mdspan<const T, typename list_spec<uint32_t, T, IdxT>::list_extents, row_major> list_data,
  uint32_t veclen,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  device_matrix_view<T, uint32_t, row_major> codes)
{
  uint32_t n_rows = codes.extent(0);
  uint32_t dim    = codes.extent(1);
  if (n_rows == 0 || dim == 0) return;
  static constexpr uint32_t kBlockSize = 256;
  dim3 blocks(div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto stream = resource::get_cuda_stream(res);
  unpack_interleaved_list_kernel<<<blocks, threads, 0, stream>>>(
    list_data.data_handle(), codes.data_handle(), n_rows, dim, veclen, offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace raft::neighbors::ivf_flat::detail
