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
#include "ann_serialization.h"
#include "ann_utils.cuh"

#include <raft/cluster/kmeans_balanced.cuh>
#include <raft/core/handle.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/stats/histogram.cuh>
#include <raft/util/pow2_utils.cuh>

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
 * @tparam T      element type.
 * @tparam IdxT   type of the indices in the source source_vecs
 * @tparam LabelT label type
 * @tparam gather_src if false, then we build the index from vectors source_vecs[i,:], otherwise
 *     we use source_vecs[source_ixs[i],:]. In both cases i=0..n_rows-1.
 *
 * @param[in] labels device pointer to the cluster ids for each row [n_rows]
 * @param[in] list_offsets device pointer to the cluster offsets in the output (index) [n_lists]
 * @param[in] source_vecs device pointer to the input data [n_rows, dim]
 * @param[in] source_ixs device pointer to the input indices [n_rows]
 * @param[out] list_data device pointer to the output [index_size, dim]
 * @param[out] list_index device pointer to the source ids corr. to the output [index_size]
 * @param[out] list_sizes_ptr device pointer to the cluster sizes [n_lists];
 *                          it's used as an atomic counter, and must be initialized with zeros.
 * @param n_rows source length
 * @param dim the dimensionality of the data
 * @param veclen size of vectorized loads/stores; must satisfy `dim % veclen == 0`.
 *
 */
template <typename T, typename IdxT, typename LabelT, bool gather_src = false>
__global__ void build_index_kernel(const LabelT* labels,
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

/** See raft::spatial::knn::ivf_flat::extend docs */
template <typename T, typename IdxT>
inline auto extend(const handle_t& handle,
                   const index<T, IdxT>& orig_index,
                   const T* new_vectors,
                   const IdxT* new_indices,
                   IdxT n_rows) -> index<T, IdxT>
{
  using LabelT = uint32_t;

  auto stream  = handle.get_stream();
  auto n_lists = orig_index.n_lists();
  auto dim     = orig_index.dim();
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_flat::extend(%zu, %u)", size_t(n_rows), dim);

  RAFT_EXPECTS(new_indices != nullptr || orig_index.size() == 0,
               "You must pass data indices when the index is non-empty.");

  rmm::device_uvector<LabelT> new_labels(n_rows, stream);
  raft::cluster::KMeansBalancedParams kmeans_params;
  kmeans_params.metric     = orig_index.metric();
  auto new_vectors_view    = raft::make_device_matrix_view<const T, IdxT>(new_vectors, n_rows, dim);
  auto orig_centroids_view = raft::make_device_matrix_view<const float, IdxT>(
    orig_index.centers().data_handle(), n_lists, dim);
  auto labels_view = raft::make_device_vector_view<LabelT, IdxT>(new_labels.data(), n_rows);
  raft::cluster::kmeans_balanced::predict(handle,
                                          kmeans_params,
                                          new_vectors_view,
                                          orig_centroids_view,
                                          labels_view,
                                          utils::mapping<float>{});

  index<T, IdxT> ext_index(
    handle, orig_index.metric(), n_lists, orig_index.adaptive_centers(), dim);

  auto list_sizes_ptr   = ext_index.list_sizes().data_handle();
  auto list_offsets_ptr = ext_index.list_offsets().data_handle();
  auto centers_ptr      = ext_index.centers().data_handle();

  // Calculate the centers and sizes on the new data, starting from the original values
  raft::copy(centers_ptr, orig_index.centers().data_handle(), ext_index.centers().size(), stream);

  if (ext_index.adaptive_centers()) {
    raft::copy(
      list_sizes_ptr, orig_index.list_sizes().data_handle(), ext_index.list_sizes().size(), stream);
    auto centroids_view = raft::make_device_matrix_view<float, IdxT>(centers_ptr, n_lists, dim);
    auto list_sizes_view =
      raft::make_device_vector_view<std::remove_pointer_t<decltype(list_sizes_ptr)>, IdxT>(
        list_sizes_ptr, n_lists);
    raft::cluster::kmeans_balanced::calc_centers_and_sizes(handle,
                                                           new_vectors_view,
                                                           centroids_view,
                                                           labels_view,
                                                           list_sizes_view,
                                                           false,
                                                           utils::mapping<float>{});
  } else {
    raft::stats::histogram<uint32_t, IdxT>(raft::stats::HistTypeAuto,
                                           reinterpret_cast<int32_t*>(list_sizes_ptr),
                                           IdxT(n_lists),
                                           new_labels.data(),
                                           n_rows,
                                           1,
                                           stream);
    raft::linalg::add(
      list_sizes_ptr, list_sizes_ptr, orig_index.list_sizes().data_handle(), n_lists, stream);
  }

  // Calculate new offsets
  IdxT index_size = 0;
  update_device(list_offsets_ptr, &index_size, 1, stream);
  thrust::inclusive_scan(
    rmm::exec_policy(stream),
    list_sizes_ptr,
    list_sizes_ptr + n_lists,
    list_offsets_ptr + 1,
    [] __device__(IdxT s, uint32_t l) { return s + Pow2<kIndexGroupSize>::roundUp(l); });
  update_host(&index_size, list_offsets_ptr + n_lists, 1, stream);
  handle.sync_stream(stream);

  ext_index.allocate(
    handle, index_size, ext_index.metric() == raft::distance::DistanceType::L2Expanded);

  // Populate index with the old data
  if (orig_index.size() > 0) {
    utils::block_copy(orig_index.list_offsets().data_handle(),
                      list_offsets_ptr,
                      IdxT(n_lists),
                      orig_index.data().data_handle(),
                      ext_index.data().data_handle(),
                      IdxT(dim),
                      stream);

    utils::block_copy(orig_index.list_offsets().data_handle(),
                      list_offsets_ptr,
                      IdxT(n_lists),
                      orig_index.indices().data_handle(),
                      ext_index.indices().data_handle(),
                      IdxT(1),
                      stream);
  }

  // Copy the old sizes, so we can start from the current state of the index;
  // we'll rebuild the `list_sizes_ptr` in the following kernel, using it as an atomic counter.
  raft::copy(
    list_sizes_ptr, orig_index.list_sizes().data_handle(), ext_index.list_sizes().size(), stream);

  const dim3 block_dim(256);
  const dim3 grid_dim(raft::ceildiv<IdxT>(n_rows, block_dim.x));
  build_index_kernel<<<grid_dim, block_dim, 0, stream>>>(new_labels.data(),
                                                         list_offsets_ptr,
                                                         new_vectors,
                                                         new_indices,
                                                         ext_index.data().data_handle(),
                                                         ext_index.indices().data_handle(),
                                                         list_sizes_ptr,
                                                         n_rows,
                                                         dim,
                                                         ext_index.veclen());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Precompute the centers vector norms for L2Expanded distance
  if (ext_index.center_norms().has_value()) {
    if (!ext_index.adaptive_centers() && orig_index.center_norms().has_value()) {
      raft::copy(ext_index.center_norms()->data_handle(),
                 orig_index.center_norms()->data_handle(),
                 orig_index.center_norms()->size(),
                 stream);
    } else {
      raft::linalg::rowNorm(ext_index.center_norms()->data_handle(),
                            ext_index.centers().data_handle(),
                            dim,
                            n_lists,
                            raft::linalg::L2Norm,
                            true,
                            stream,
                            raft::sqrt_op());
      RAFT_LOG_TRACE_VEC(ext_index.center_norms()->data_handle(), std::min<uint32_t>(dim, 20));
    }
  }

  // assemble the index
  return ext_index;
}

/** See raft::spatial::knn::ivf_flat::build docs */
template <typename T, typename IdxT>
inline auto build(
  const handle_t& handle, const index_params& params, const T* dataset, IdxT n_rows, uint32_t dim)
  -> index<T, IdxT>
{
  auto stream = handle.get_stream();
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_flat::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "unsupported data type");
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");

  index<T, IdxT> index(handle, params, dim);
  utils::memzero(index.list_sizes().data_handle(), index.list_sizes().size(), stream);
  utils::memzero(index.list_offsets().data_handle(), index.list_offsets().size(), stream);

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
    raft::cluster::KMeansBalancedParams kmeans_params;
    kmeans_params.n_iters = params.kmeans_n_iters;
    kmeans_params.metric  = index.metric();
    raft::cluster::kmeans_balanced::fit(
      handle, kmeans_params, trainset_const_view, index.centers().view(), utils::mapping<float>{});
  }

  // add the data if necessary
  if (params.add_data_on_build) {
    return detail::extend<T, IdxT>(handle, index, dataset, nullptr, n_rows);
  } else {
    return index;
  }
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
inline void fill_refinement_index(const handle_t& handle,
                                  index<T, IdxT>* refinement_index,
                                  const T* dataset,
                                  const IdxT* candidate_idx,
                                  IdxT n_queries,
                                  uint32_t n_candidates)
{
  using LabelT = uint32_t;

  auto stream      = handle.get_stream();
  uint32_t n_lists = n_queries;
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_flat::fill_refinement_index(%zu, %u)", size_t(n_queries));

  rmm::device_uvector<LabelT> new_labels(n_queries * n_candidates, stream);
  linalg::writeOnlyUnaryOp(
    new_labels.data(),
    n_queries * n_candidates,
    [n_candidates] __device__(LabelT * out, uint32_t i) { *out = i / n_candidates; },
    stream);

  auto list_sizes_ptr   = refinement_index->list_sizes().data_handle();
  auto list_offsets_ptr = refinement_index->list_offsets().data_handle();
  // We do not fill centers and center norms, since we will not run coarse search.

  // Calculate new offsets
  uint32_t n_roundup = Pow2<kIndexGroupSize>::roundUp(n_candidates);
  linalg::writeOnlyUnaryOp(
    refinement_index->list_offsets().data_handle(),
    refinement_index->list_offsets().size(),
    [n_roundup] __device__(IdxT * out, uint32_t i) { *out = i * n_roundup; },
    stream);

  IdxT index_size = n_roundup * n_lists;
  refinement_index->allocate(
    handle, index_size, refinement_index->metric() == raft::distance::DistanceType::L2Expanded);

  RAFT_CUDA_TRY(cudaMemsetAsync(list_sizes_ptr, 0, n_lists * sizeof(uint32_t), stream));

  const dim3 block_dim(256);
  const dim3 grid_dim(raft::ceildiv<IdxT>(n_queries * n_candidates, block_dim.x));
  build_index_kernel<T, IdxT, LabelT, true>
    <<<grid_dim, block_dim, 0, stream>>>(new_labels.data(),
                                         list_offsets_ptr,
                                         dataset,
                                         candidate_idx,
                                         refinement_index->data().data_handle(),
                                         refinement_index->indices().data_handle(),
                                         list_sizes_ptr,
                                         n_queries * n_candidates,
                                         refinement_index->dim(),
                                         refinement_index->veclen());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

static const int serialization_version = 1;

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index_ IVF-Flat index
 *
 */
template <typename T, typename IdxT>
void save(const handle_t& handle, const std::string& filename, const index<T, IdxT>& index_)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open %s", filename.c_str()); }

  RAFT_LOG_DEBUG(
    "Saving IVF-PQ index, size %zu, dim %u", static_cast<size_t>(index_.size()), index_.dim());
  write_scalar(of, serialization_version);
  write_scalar(of, index_.size());
  write_scalar(of, index_.dim());
  write_scalar(of, index_.n_lists());
  write_scalar(of, index_.metric());
  write_scalar(of, index_.veclen());
  write_scalar(of, index_.adaptive_centers());
  write_mdspan(handle, of, index_.data());
  write_mdspan(handle, of, index_.indices());
  write_mdspan(handle, of, index_.list_sizes());
  write_mdspan(handle, of, index_.list_offsets());
  write_mdspan(handle, of, index_.centers());
  if (index_.center_norms()) {
    bool has_norms = true;
    write_scalar(of, has_norms);
    write_mdspan(handle, of, *index_.center_norms());
  } else {
    bool has_norms = false;
    write_scalar(of, has_norms);
  }
  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

/** Load an index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index_ IVF-Flat index
 *
 */
template <typename T, typename IdxT>
auto load(const handle_t& handle, const std::string& filename) -> index<T, IdxT>
{
  std::ifstream infile(filename, std::ios::in | std::ios::binary);

  if (!infile) { RAFT_FAIL("Cannot open %s", filename.c_str()); }

  auto ver = read_scalar<int>(infile);
  if (ver != serialization_version) {
    RAFT_FAIL("serialization version mismatch, expected %d, got %d ", serialization_version, ver);
  }
  auto n_rows           = read_scalar<IdxT>(infile);
  auto dim              = read_scalar<uint32_t>(infile);
  auto n_lists          = read_scalar<uint32_t>(infile);
  auto metric           = read_scalar<raft::distance::DistanceType>(infile);
  auto veclen           = read_scalar<uint32_t>(infile);
  bool adaptive_centers = read_scalar<bool>(infile);

  index<T, IdxT> index_ =
    raft::spatial::knn::ivf_flat::index<T, IdxT>(handle, metric, n_lists, adaptive_centers, dim);

  index_.allocate(handle, n_rows, metric == raft::distance::DistanceType::L2Expanded);
  auto data = index_.data();
  read_mdspan(handle, infile, data);
  read_mdspan(handle, infile, index_.indices());
  read_mdspan(handle, infile, index_.list_sizes());
  read_mdspan(handle, infile, index_.list_offsets());
  read_mdspan(handle, infile, index_.centers());
  bool has_norms = read_scalar<bool>(infile);
  if (has_norms) {
    if (!index_.center_norms()) {
      RAFT_FAIL("Error inconsistent center norms");
    } else {
      auto center_norms = *index_.center_norms();
      read_mdspan(handle, infile, center_norms);
    }
  }
  infile.close();
  return index_;
}
}  // namespace raft::spatial::knn::ivf_flat::detail
