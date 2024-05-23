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

#include <raft/core/logger.hpp>  // RAFT_LOG_TRACE
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>                              // raft::resources
#include <raft/distance/distance_types.hpp>                     // is_min_close, DistanceType
#include <raft/linalg/gemm.cuh>                                 // raft::linalg::gemm
#include <raft/linalg/norm.cuh>                                 // raft::linalg::norm
#include <raft/linalg/unary_op.cuh>                             // raft::linalg::unary_op
#include <raft/matrix/detail/select_k.cuh>                      // matrix::detail::select_k
#include <raft/neighbors/detail/ivf_common.cuh>                 // raft::neighbors::detail::ivf
#include <raft/neighbors/detail/ivf_flat_interleaved_scan.cuh>  // interleaved_scan
#include <raft/neighbors/ivf_flat_types.hpp>                    // raft::neighbors::ivf_flat::index
#include <raft/neighbors/sample_filter_types.hpp>               // none_ivf_sample_filter
#include <raft/spatial/knn/detail/ann_utils.cuh>                // utils::mapping

#include <rmm/resource_ref.hpp>

namespace raft::neighbors::ivf_flat::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

template <typename T, typename AccT, typename IdxT, typename IvfSampleFilterT>
void search_impl(raft::resources const& handle,
                 const raft::neighbors::ivf_flat::index<T, IdxT>& index,
                 const T* queries,
                 uint32_t n_queries,
                 uint32_t queries_offset,
                 uint32_t k,
                 uint32_t n_probes,
                 uint32_t max_samples,
                 bool select_min,
                 IdxT* neighbors,
                 AccT* distances,
                 rmm::device_async_resource_ref search_mr,
                 IvfSampleFilterT sample_filter)
{
  auto stream = resource::get_cuda_stream(handle);

  std::size_t n_queries_probes = std::size_t(n_queries) * std::size_t(n_probes);

  // The norm of query
  rmm::device_uvector<float> query_norm_dev(n_queries, stream, search_mr);
  // The distance value of cluster(list) and queries
  rmm::device_uvector<float> distance_buffer_dev(n_queries * index.n_lists(), stream, search_mr);
  // The topk distance value of cluster(list) and queries
  rmm::device_uvector<float> coarse_distances_dev(n_queries_probes, stream, search_mr);
  // The topk  index of cluster(list) and queries
  rmm::device_uvector<uint32_t> coarse_indices_dev(n_queries_probes, stream, search_mr);

  // Optional structures if postprocessing is required
  // The topk distance value of candidate vectors from each cluster(list)
  rmm::device_uvector<AccT> distances_tmp_dev(0, stream, search_mr);
  // Number of samples for each query
  rmm::device_uvector<uint32_t> num_samples(0, stream, search_mr);
  // Offsets per probe for each query
  rmm::device_uvector<uint32_t> chunk_index(0, stream, search_mr);

  // The topk index of candidate vectors from each cluster(list), local index offset
  // also we might need additional storage for select_k
  rmm::device_uvector<uint32_t> indices_tmp_dev(0, stream, search_mr);
  rmm::device_uvector<uint32_t> neighbors_uint32_buf(0, stream, search_mr);

  size_t float_query_size;
  if constexpr (std::is_integral_v<T>) {
    float_query_size = n_queries * index.dim();
  } else {
    float_query_size = 0;
  }
  rmm::device_uvector<float> converted_queries_dev(float_query_size, stream, search_mr);
  float* converted_queries_ptr = converted_queries_dev.data();

  if constexpr (std::is_same_v<T, float>) {
    converted_queries_ptr = const_cast<float*>(queries);
  } else {
    linalg::unaryOp(
      converted_queries_ptr, queries, n_queries * index.dim(), utils::mapping<float>{}, stream);
  }

  float alpha = 1.0f;
  float beta  = 0.0f;

  // todo(lsugy): raft distance? (if performance is similar/better than gemm)
  switch (index.metric()) {
    case raft::distance::DistanceType::L2Expanded:
    case raft::distance::DistanceType::L2SqrtExpanded: {
      alpha = -2.0f;
      beta  = 1.0f;
      raft::linalg::rowNorm(query_norm_dev.data(),
                            converted_queries_ptr,
                            static_cast<IdxT>(index.dim()),
                            static_cast<IdxT>(n_queries),
                            raft::linalg::L2Norm,
                            true,
                            stream);
      utils::outer_add(query_norm_dev.data(),
                       (IdxT)n_queries,
                       index.center_norms()->data_handle(),
                       (IdxT)index.n_lists(),
                       distance_buffer_dev.data(),
                       stream);
      RAFT_LOG_TRACE_VEC(index.center_norms()->data_handle(), std::min<uint32_t>(20, index.dim()));
      RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), std::min<uint32_t>(20, index.n_lists()));
      break;
    }
    default: {
      alpha = 1.0f;
      beta  = 0.0f;
    }
  }

  linalg::gemm(handle,
               true,
               false,
               index.n_lists(),
               n_queries,
               index.dim(),
               &alpha,
               index.centers().data_handle(),
               index.dim(),
               converted_queries_ptr,
               index.dim(),
               &beta,
               distance_buffer_dev.data(),
               index.n_lists(),
               stream);

  RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), std::min<uint32_t>(20, index.n_lists()));
  matrix::detail::select_k<AccT, uint32_t>(handle,
                                           distance_buffer_dev.data(),
                                           nullptr,
                                           n_queries,
                                           index.n_lists(),
                                           n_probes,
                                           coarse_distances_dev.data(),
                                           coarse_indices_dev.data(),
                                           select_min);
  RAFT_LOG_TRACE_VEC(coarse_indices_dev.data(), n_probes);
  RAFT_LOG_TRACE_VEC(coarse_distances_dev.data(), n_probes);

  uint32_t grid_dim_x = 0;
  if (n_probes > 1) {
    // query the gridDimX size to store probes topK output
    ivfflat_interleaved_scan<T, typename utils::config<T>::value_t, IdxT, IvfSampleFilterT>(
      index,
      nullptr,
      nullptr,
      n_queries,
      queries_offset,
      index.metric(),
      n_probes,
      k,
      0,
      nullptr,
      select_min,
      sample_filter,
      nullptr,
      nullptr,
      grid_dim_x,
      stream);
  } else {
    grid_dim_x = 1;
  }

  num_samples.resize(n_queries, stream);
  chunk_index.resize(n_queries_probes, stream);

  ivf::detail::calc_chunk_indices::configure(n_probes, n_queries)(index.list_sizes().data_handle(),
                                                                  coarse_indices_dev.data(),
                                                                  chunk_index.data(),
                                                                  num_samples.data(),
                                                                  stream);

  auto distances_dev_ptr = distances;

  uint32_t* neighbors_uint32 = nullptr;
  if constexpr (sizeof(IdxT) == sizeof(uint32_t)) {
    neighbors_uint32 = reinterpret_cast<uint32_t*>(neighbors);
  } else {
    neighbors_uint32_buf.resize(std::size_t(n_queries) * std::size_t(k), stream);
    neighbors_uint32 = neighbors_uint32_buf.data();
  }

  uint32_t* indices_dev_ptr = nullptr;

  bool manage_local_topk = is_local_topk_feasible(k);
  if (!manage_local_topk || grid_dim_x > 1) {
    auto target_size = std::size_t(n_queries) * (manage_local_topk ? grid_dim_x * k : max_samples);

    distances_tmp_dev.resize(target_size, stream);
    if (manage_local_topk) indices_tmp_dev.resize(target_size, stream);

    distances_dev_ptr = distances_tmp_dev.data();
    indices_dev_ptr   = indices_tmp_dev.data();
  } else {
    indices_dev_ptr = neighbors_uint32;
  }

  ivfflat_interleaved_scan<T, typename utils::config<T>::value_t, IdxT, IvfSampleFilterT>(
    index,
    queries,
    coarse_indices_dev.data(),
    n_queries,
    queries_offset,
    index.metric(),
    n_probes,
    k,
    max_samples,
    chunk_index.data(),
    select_min,
    sample_filter,
    indices_dev_ptr,
    distances_dev_ptr,
    grid_dim_x,
    stream);

  RAFT_LOG_TRACE_VEC(distances_dev_ptr, 2 * k);
  if (indices_dev_ptr != nullptr) { RAFT_LOG_TRACE_VEC(indices_dev_ptr, 2 * k); }

  // Merge topk values from different blocks
  if (!manage_local_topk || grid_dim_x > 1) {
    matrix::detail::select_k<AccT, uint32_t>(handle,
                                             distances_tmp_dev.data(),
                                             indices_tmp_dev.data(),
                                             n_queries,
                                             manage_local_topk ? (k * grid_dim_x) : max_samples,
                                             k,
                                             distances,
                                             neighbors_uint32,
                                             select_min,
                                             false,
                                             matrix::SelectAlgo::kAuto,
                                             manage_local_topk ? nullptr : num_samples.data());
  }
  if (!manage_local_topk) {
    // post process distances && neighbor IDs
    ivf::detail::postprocess_distances(
      distances, distances, index.metric(), n_queries, k, 1.0, false, stream);
  }
  ivf::detail::postprocess_neighbors(neighbors,
                                     neighbors_uint32,
                                     index.inds_ptrs().data_handle(),
                                     coarse_indices_dev.data(),
                                     chunk_index.data(),
                                     n_queries,
                                     n_probes,
                                     k,
                                     stream);
}

/** See raft::neighbors::ivf_flat::search docs */
template <typename T,
          typename IdxT,
          typename IvfSampleFilterT = raft::neighbors::filtering::none_ivf_sample_filter>
inline void search(raft::resources const& handle,
                   const search_params& params,
                   const index<T, IdxT>& index,
                   const T* queries,
                   uint32_t n_queries,
                   uint32_t k,
                   IdxT* neighbors,
                   float* distances,
                   rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource(),
                   IvfSampleFilterT sample_filter    = IvfSampleFilterT())
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_flat::search(k = %u, n_queries = %u, dim = %zu)", k, n_queries, index.dim());

  RAFT_EXPECTS(params.n_probes > 0,
               "n_probes (number of clusters to probe in the search) must be positive.");
  auto n_probes          = std::min<uint32_t>(params.n_probes, index.n_lists());
  bool manage_local_topk = is_local_topk_feasible(k);

  uint32_t max_samples = 0;
  if (!manage_local_topk) {
    IdxT ms =
      Pow2<128 / sizeof(float)>::roundUp(std::max<IdxT>(index.accum_sorted_sizes()(n_probes), k));
    RAFT_EXPECTS(ms <= IdxT(std::numeric_limits<uint32_t>::max()),
                 "The maximum sample size is too big.");
    max_samples = ms;
  }

  // a batch size heuristic: try to keep the workspace within the specified size
  constexpr uint64_t kExpectedWsSize = 1024 * 1024 * 1024;
  uint64_t max_ws_size = std::min(resource::get_workspace_free_bytes(handle), kExpectedWsSize);

  uint64_t ws_size_per_query = 4ull * (2 * n_probes + index.n_lists() + index.dim() + 1) +
                               (manage_local_topk ? ((sizeof(IdxT) + 4) * n_probes * k)
                                                  : (4ull * (max_samples + n_probes + 1)));

  const uint32_t max_queries =
    std::min<uint32_t>(n_queries, raft::div_rounding_up_safe(max_ws_size, ws_size_per_query));

  for (uint32_t offset_q = 0; offset_q < n_queries; offset_q += max_queries) {
    uint32_t queries_batch = min(max_queries, n_queries - offset_q);

    search_impl<T, float, IdxT, IvfSampleFilterT>(handle,
                                                  index,
                                                  queries + offset_q * index.dim(),
                                                  queries_batch,
                                                  offset_q,
                                                  k,
                                                  n_probes,
                                                  max_samples,
                                                  raft::distance::is_min_close(index.metric()),
                                                  neighbors + offset_q * k,
                                                  distances + offset_q * k,
                                                  mr,
                                                  sample_filter);
  }
}

}  // namespace raft::neighbors::ivf_flat::detail
