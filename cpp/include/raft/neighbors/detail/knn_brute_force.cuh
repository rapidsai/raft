/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <iostream>
#include <raft/core/device_resources.hpp>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/spatial/knn/detail/faiss_select/DistanceUtils.h>
#include <raft/spatial/knn/detail/faiss_select/Select.cuh>
#include <raft/spatial/knn/detail/fused_l2_knn.cuh>
#include <raft/spatial/knn/detail/haversine_distance.cuh>
#include <raft/spatial/knn/detail/selection_faiss.cuh>
#include <set>
#include <thrust/iterator/transform_iterator.h>

namespace raft::neighbors::detail {
using namespace raft::spatial::knn::detail;
using namespace raft::spatial::knn;

template <typename value_idx = std::int64_t,
          typename value_t   = float,
          int warp_q,
          int thread_q,
          int tpb>
__global__ void knn_merge_parts_kernel(value_t* inK,
                                       value_idx* inV,
                                       value_t* outK,
                                       value_idx* outV,
                                       size_t n_samples,
                                       int n_parts,
                                       value_t initK,
                                       value_idx initV,
                                       int k,
                                       value_idx* translations)
{
  constexpr int kNumWarps = tpb / WarpSize;

  __shared__ value_t smemK[kNumWarps * warp_q];
  __shared__ value_idx smemV[kNumWarps * warp_q];

  /**
   * Uses shared memory
   */
  faiss_select::
    BlockSelect<value_t, value_idx, false, faiss_select::Comparator<value_t>, warp_q, thread_q, tpb>
      heap(initK, initV, smemK, smemV, k);

  // Grid is exactly sized to rows available
  int row     = blockIdx.x;
  int total_k = k * n_parts;

  int i = threadIdx.x;

  // Get starting pointers for cols in current thread
  int part       = i / k;
  size_t row_idx = (row * k) + (part * n_samples * k);

  int col = i % k;

  value_t* inKStart   = inK + (row_idx + col);
  value_idx* inVStart = inV + (row_idx + col);

  int limit             = Pow2<WarpSize>::roundDown(total_k);
  value_idx translation = 0;

  for (; i < limit; i += tpb) {
    translation = translations[part];
    heap.add(*inKStart, (*inVStart) + translation);

    part    = (i + tpb) / k;
    row_idx = (row * k) + (part * n_samples * k);

    col = (i + tpb) % k;

    inKStart = inK + (row_idx + col);
    inVStart = inV + (row_idx + col);
  }

  // Handle last remainder fraction of a warp of elements
  if (i < total_k) {
    translation = translations[part];
    heap.addThreadQ(*inKStart, (*inVStart) + translation);
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    outK[row * k + i] = smemK[i];
    outV[row * k + i] = smemV[i];
  }
}

template <typename value_idx = std::int64_t, typename value_t = float, int warp_q, int thread_q>
inline void knn_merge_parts_impl(value_t* inK,
                                 value_idx* inV,
                                 value_t* outK,
                                 value_idx* outV,
                                 size_t n_samples,
                                 int n_parts,
                                 int k,
                                 cudaStream_t stream,
                                 value_idx* translations)
{
  auto grid = dim3(n_samples);

  constexpr int n_threads = (warp_q <= 1024) ? 128 : 64;
  auto block              = dim3(n_threads);

  auto kInit = std::numeric_limits<value_t>::max();
  auto vInit = -1;
  knn_merge_parts_kernel<value_idx, value_t, warp_q, thread_q, n_threads>
    <<<grid, block, 0, stream>>>(
      inK, inV, outK, outV, n_samples, n_parts, kInit, vInit, k, translations);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Calculates brute force knn, using a fixed memory budget
 * by tiling over both the rows and columns of pairwise_distances
 */
template <typename ElementType = float, typename IndexType = int64_t>
void tiled_brute_force_knn(const raft::device_resources& handle,
                           const ElementType* search,  // size (m ,d)
                           const ElementType* index,   // size (n ,d)
                           size_t m,
                           size_t n,
                           size_t d,
                           int k,
                           ElementType* distances,  // size (m, k)
                           IndexType* indices,      // size (m, k)
                           raft::distance::DistanceType metric,
                           float metric_arg         = 0.0,
                           size_t max_row_tile_size = 0,
                           size_t max_col_tile_size = 0)
{
  // Figure out the number of rows/cols to tile for
  size_t tile_rows   = 0;
  size_t tile_cols   = 0;
  auto stream        = handle.get_stream();
  auto device_memory = handle.get_workspace_resource();
  auto total_mem     = device_memory->get_mem_info(stream).second;
  raft::spatial::knn::detail::faiss_select::chooseTileSize(
    m, n, d, sizeof(ElementType), total_mem, tile_rows, tile_cols);

  // for unittesting, its convenient to be able to put a max size on the tiles
  // so we can test the tiling logic without having to use huge inputs.
  if (max_row_tile_size && (tile_rows > max_row_tile_size)) { tile_rows = max_row_tile_size; }
  if (max_col_tile_size && (tile_cols > max_col_tile_size)) { tile_cols = max_col_tile_size; }

  // tile_cols must be at least k items
  tile_cols = std::max(tile_cols, static_cast<size_t>(k));

  // stores pairwise distances for the current tile
  rmm::device_uvector<ElementType> temp_distances(tile_rows * tile_cols, stream);

  // if we're tiling over columns, we need additional buffers for temporary output
  // distances/indices
  size_t num_col_tiles = raft::ceildiv(n, tile_cols);
  size_t temp_out_cols = k * num_col_tiles;

  // the final column tile could have less than 'k' items in it
  // in which case the number of columns here is too high in the temp output.
  // adjust if necessary
  auto last_col_tile_size = n % tile_cols;
  if (last_col_tile_size && (last_col_tile_size < static_cast<size_t>(k))) {
    temp_out_cols -= k - last_col_tile_size;
  }
  rmm::device_uvector<ElementType> temp_out_distances(tile_rows * temp_out_cols, stream);
  rmm::device_uvector<IndexType> temp_out_indices(tile_rows * temp_out_cols, stream);

  bool select_min = raft::distance::is_min_close(metric);

  for (size_t i = 0; i < m; i += tile_rows) {
    size_t current_query_size = std::min(tile_rows, m - i);

    for (size_t j = 0; j < n; j += tile_cols) {
      size_t current_centroid_size = std::min(tile_cols, n - j);
      size_t current_k             = std::min(current_centroid_size, static_cast<size_t>(k));

      // calculate the top-k elements for the current tile, by calculating the
      // full pairwise distance for the tile - and then selecting the top-k from that
      // note: we're using a int32 IndexType here on purpose in order to
      // use the pairwise_distance specializations. Since the tile size will ensure
      // that the total memory is < 1GB per tile, this will not cause any issues
      distance::pairwise_distance<ElementType, int>(handle,
                                                    search + i * d,
                                                    index + j * d,
                                                    temp_distances.data(),
                                                    current_query_size,
                                                    current_centroid_size,
                                                    d,
                                                    metric,
                                                    true,
                                                    metric_arg);

      detail::select_k<IndexType, ElementType>(temp_distances.data(),
                                               nullptr,
                                               current_query_size,
                                               current_centroid_size,
                                               distances + i * k,
                                               indices + i * k,
                                               select_min,
                                               current_k,
                                               stream);

      // if we're tiling over columns, we need to do a couple things to fix up
      // the output of select_k
      // 1. The column id's in the output are relative to the tile, so we need
      // to adjust the column ids by adding the column the tile starts at (j)
      // 2. select_k writes out output in a row-major format, which means we
      // can't just concat the output of all the tiles and do a select_k on the
      // concatenation.
      // Fix both of these problems in a single pass here
      if (tile_cols != n) {
        const ElementType* in_distances = distances + i * k;
        const IndexType* in_indices     = indices + i * k;
        ElementType* out_distances      = temp_out_distances.data();
        IndexType* out_indices          = temp_out_indices.data();

        auto count = thrust::make_counting_iterator<IndexType>(0);
        thrust::for_each(handle.get_thrust_policy(),
                         count,
                         count + current_query_size * current_k,
                         [=] __device__(IndexType i) {
                           IndexType row = i / current_k, col = i % current_k;
                           IndexType out_index = row * temp_out_cols + j * k / tile_cols + col;

                           out_distances[out_index] = in_distances[i];
                           out_indices[out_index]   = in_indices[i] + j;
                         });
      }
    }

    if (tile_cols != n) {
      // select the actual top-k items here from the temporary output
      detail::select_k<IndexType, ElementType>(temp_out_distances.data(),
                                               temp_out_indices.data(),
                                               current_query_size,
                                               temp_out_cols,
                                               distances + i * k,
                                               indices + i * k,
                                               select_min,
                                               k,
                                               stream);
    }
  }
}

/**
 * @brief Merge knn distances and index matrix, which have been partitioned
 * by row, into a single matrix with only the k-nearest neighbors.
 *
 * @param inK partitioned knn distance matrix
 * @param inV partitioned knn index matrix
 * @param outK merged knn distance matrix
 * @param outV merged knn index matrix
 * @param n_samples number of samples per partition
 * @param n_parts number of partitions
 * @param k number of neighbors per partition (also number of merged neighbors)
 * @param stream CUDA stream to use
 * @param translations mapping of index offsets for each partition
 */
template <typename value_idx = std::int64_t, typename value_t = float>
inline void knn_merge_parts(value_t* inK,
                            value_idx* inV,
                            value_t* outK,
                            value_idx* outV,
                            size_t n_samples,
                            int n_parts,
                            int k,
                            cudaStream_t stream,
                            value_idx* translations)
{
  if (k == 1)
    knn_merge_parts_impl<value_idx, value_t, 1, 1>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  else if (k <= 32)
    knn_merge_parts_impl<value_idx, value_t, 32, 2>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  else if (k <= 64)
    knn_merge_parts_impl<value_idx, value_t, 64, 3>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  else if (k <= 128)
    knn_merge_parts_impl<value_idx, value_t, 128, 3>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  else if (k <= 256)
    knn_merge_parts_impl<value_idx, value_t, 256, 4>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  else if (k <= 512)
    knn_merge_parts_impl<value_idx, value_t, 512, 8>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  else if (k <= 1024)
    knn_merge_parts_impl<value_idx, value_t, 1024, 8>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
}

/**
 * Search the kNN for the k-nearest neighbors of a set of query vectors
 * @param[in] input vector of device device memory array pointers to search
 * @param[in] sizes vector of memory sizes for each device array pointer in input
 * @param[in] D number of cols in input and search_items
 * @param[in] search_items set of vectors to query for neighbors
 * @param[in] n        number of items in search_items
 * @param[out] res_I    pointer to device memory for returning k nearest indices
 * @param[out] res_D    pointer to device memory for returning k nearest distances
 * @param[in] k        number of neighbors to query
 * @param[in] userStream the main cuda stream to use
 * @param[in] internalStreams optional when n_params > 0, the index partitions can be
 *        queried in parallel using these streams. Note that n_int_streams also
 *        has to be > 0 for these to be used and their cardinality does not need
 *        to correspond to n_parts.
 * @param[in] n_int_streams size of internalStreams. When this is <= 0, only the
 *        user stream will be used.
 * @param[in] rowMajorIndex are the index arrays in row-major layout?
 * @param[in] rowMajorQuery are the query array in row-major layout?
 * @param[in] translations translation ids for indices when index rows represent
 *        non-contiguous partitions
 * @param[in] metric corresponds to the raft::distance::DistanceType enum (default is L2Expanded)
 * @param[in] metricArg metric argument to use. Corresponds to the p arg for lp norm
 */
template <typename IntType = int, typename IdxType = std::int64_t, typename value_t = float>
void brute_force_knn_impl(
  raft::device_resources const& handle,
  std::vector<value_t*>& input,
  std::vector<IntType>& sizes,
  IntType D,
  value_t* search_items,
  IntType n,
  IdxType* res_I,
  value_t* res_D,
  IntType k,
  bool rowMajorIndex                  = true,
  bool rowMajorQuery                  = true,
  std::vector<IdxType>* translations  = nullptr,
  raft::distance::DistanceType metric = raft::distance::DistanceType::L2Expanded,
  float metricArg                     = 0)
{
  auto userStream = handle.get_stream();

  ASSERT(input.size() == sizes.size(), "input and sizes vectors should be the same size");

  std::vector<IdxType>* id_ranges;
  if (translations == nullptr) {
    // If we don't have explicit translations
    // for offsets of the indices, build them
    // from the local partitions
    id_ranges       = new std::vector<IdxType>();
    IdxType total_n = 0;
    for (size_t i = 0; i < input.size(); i++) {
      id_ranges->push_back(total_n);
      total_n += sizes[i];
    }
  } else {
    // otherwise, use the given translations
    id_ranges = translations;
  }

  int device;
  RAFT_CUDA_TRY(cudaGetDevice(&device));

  rmm::device_uvector<IdxType> trans(id_ranges->size(), userStream);
  raft::update_device(trans.data(), id_ranges->data(), id_ranges->size(), userStream);

  rmm::device_uvector<value_t> all_D(0, userStream);
  rmm::device_uvector<IdxType> all_I(0, userStream);

  value_t* out_D = res_D;
  IdxType* out_I = res_I;

  if (input.size() > 1) {
    all_D.resize(input.size() * k * n, userStream);
    all_I.resize(input.size() * k * n, userStream);

    out_D = all_D.data();
    out_I = all_I.data();
  }

  // Make other streams from pool wait on main stream
  handle.wait_stream_pool_on_stream();

  for (size_t i = 0; i < input.size(); i++) {
    value_t* out_d_ptr = out_D + (i * k * n);
    IdxType* out_i_ptr = out_I + (i * k * n);

    auto stream = handle.get_next_usable_stream(i);

    if (k <= 64 && rowMajorQuery == rowMajorIndex && rowMajorQuery == true &&
        (metric == raft::distance::DistanceType::L2Unexpanded ||
         metric == raft::distance::DistanceType::L2SqrtUnexpanded ||
         metric == raft::distance::DistanceType::L2Expanded ||
         metric == raft::distance::DistanceType::L2SqrtExpanded ||
         metric == raft::distance::DistanceType::LpUnexpanded)) {
      fusedL2Knn(D,
                 out_i_ptr,
                 out_d_ptr,
                 input[i],
                 search_items,
                 sizes[i],
                 n,
                 k,
                 rowMajorIndex,
                 rowMajorQuery,
                 stream,
                 metric);

      // Perform necessary post-processing
      if (metric == raft::distance::DistanceType::L2SqrtExpanded ||
          metric == raft::distance::DistanceType::L2SqrtUnexpanded ||
          metric == raft::distance::DistanceType::LpUnexpanded) {
        float p = 0.5;  // standard l2
        if (metric == raft::distance::DistanceType::LpUnexpanded) p = 1.0 / metricArg;
        raft::linalg::unaryOp<float>(
          res_D,
          res_D,
          n * k,
          [p] __device__(float input) { return powf(fabsf(input), p); },
          userStream);
      }
    } else {
      switch (metric) {
        case raft::distance::DistanceType::Haversine:
          ASSERT(D == 2,
                 "Haversine distance requires 2 dimensions "
                 "(latitude / longitude).");

          haversine_knn(out_i_ptr, out_d_ptr, input[i], search_items, sizes[i], n, k, stream);
          break;
        default:
          // currently we don't support col_major inside tiled_brute_force_knn, because
          // of limitattions of the pairwise_distance API:
          // 1) paiwise_distance takes a single 'isRowMajor' parameter - and we have
          // multiple options here (like rowMajorQuery/rowMajorIndex)
          // 2) because of tiling, we need to be able to set a custom stride in the PW
          // api, which isn't supported
          // Instead, transpose the input matrices if they are passed as col-major.
          auto search = search_items;
          rmm::device_uvector<value_t> search_row_major(0, stream);
          if (!rowMajorQuery) {
            search_row_major.resize(n * D, stream);
            raft::linalg::transpose(handle, search, search_row_major.data(), n, D, stream);
            search = search_row_major.data();
          }
          auto index = input[i];
          rmm::device_uvector<value_t> index_row_major(0, stream);
          if (!rowMajorIndex) {
            index_row_major.resize(sizes[i] * D, stream);
            raft::linalg::transpose(handle, index, index_row_major.data(), sizes[i], D, stream);
            index = index_row_major.data();
          }

          tiled_brute_force_knn<value_t, IdxType>(
            handle, search, index, n, sizes[i], D, k, out_d_ptr, out_i_ptr, metric, metricArg);
          break;
      }
    }

    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // Sync internal streams if used. We don't need to
  // sync the user stream because we'll already have
  // fully serial execution.
  handle.sync_stream_pool();

  if (input.size() > 1 || translations != nullptr) {
    // This is necessary for proper index translations. If there are
    // no translations or partitions to combine, it can be skipped.
    knn_merge_parts(out_D, out_I, res_D, res_I, n, input.size(), k, userStream, trans.data());
  }

  if (translations == nullptr) delete id_ranges;
};

}  // namespace raft::neighbors::detail
