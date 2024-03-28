/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/neighbors/detail/ivf_flat_build.cuh>
#include <raft/neighbors/detail/ivf_flat_interleaved_scan.cuh>
#include <raft/neighbors/detail/refine_common.hpp>
#include <raft/neighbors/sample_filter_types.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>

#include <thrust/sequence.h>

namespace raft::neighbors::detail {

/**
 * See raft::neighbors::refine for docs.
 */
template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine_device(raft::resources const& handle,
                   raft::device_matrix_view<const data_t, matrix_idx, row_major> dataset,
                   raft::device_matrix_view<const data_t, matrix_idx, row_major> queries,
                   raft::device_matrix_view<const idx_t, matrix_idx, row_major> neighbor_candidates,
                   raft::device_matrix_view<idx_t, matrix_idx, row_major> indices,
                   raft::device_matrix_view<distance_t, matrix_idx, row_major> distances,
                   distance::DistanceType metric = distance::DistanceType::L2Unexpanded)
{
  matrix_idx n_candidates = neighbor_candidates.extent(1);
  matrix_idx n_queries    = queries.extent(0);
  matrix_idx dim          = dataset.extent(1);
  uint32_t k              = static_cast<uint32_t>(indices.extent(1));

  // TODO: this restriction could be lifted with some effort
  RAFT_EXPECTS(k <= raft::matrix::detail::select::warpsort::kMaxCapacity,
               "k must be less than topk::kMaxCapacity (%d).",
               raft::matrix::detail::select::warpsort::kMaxCapacity);

  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "neighbors::refine(%zu, %u)", size_t(n_queries), uint32_t(n_candidates));

  refine_check_input(dataset.extents(),
                     queries.extents(),
                     neighbor_candidates.extents(),
                     indices.extents(),
                     distances.extents(),
                     metric);

  // The refinement search can be mapped to an IVF flat search:
  // - We consider that the candidate vectors form a cluster, separately for each query.
  // - In other words, the n_queries * n_candidates vectors form n_queries clusters, each with
  //   n_candidates elements.
  // - We consider that the coarse level search is already performed and assigned a single cluster
  //   to search for each query (the cluster formed from the corresponding candidates).
  // - We run IVF flat search with n_probes=1 to select the best k elements of the candidates.
  rmm::device_uvector<uint32_t> fake_coarse_idx(n_queries, resource::get_cuda_stream(handle));

  thrust::sequence(resource::get_thrust_policy(handle),
                   fake_coarse_idx.data(),
                   fake_coarse_idx.data() + n_queries);

  raft::neighbors::ivf_flat::index<data_t, idx_t> refinement_index(
    handle, metric, n_queries, false, true, dim);

  raft::neighbors::ivf_flat::detail::fill_refinement_index(handle,
                                                           &refinement_index,
                                                           dataset.data_handle(),
                                                           neighbor_candidates.data_handle(),
                                                           n_queries,
                                                           n_candidates);
  uint32_t grid_dim_x = 1;

  // the neighbor ids will be computed in uint32_t as offset
  rmm::device_uvector<uint32_t> neighbors_uint32_buf(0, resource::get_cuda_stream(handle));
  // Offsets per probe for each query [n_queries] as n_probes = 1
  rmm::device_uvector<uint32_t> chunk_index(n_queries, resource::get_cuda_stream(handle));

  // we know that each cluster has exactly n_candidates entries
  thrust::fill(resource::get_thrust_policy(handle),
               chunk_index.data(),
               chunk_index.data() + n_queries,
               uint32_t(n_candidates));

  uint32_t* neighbors_uint32 = nullptr;
  if constexpr (sizeof(idx_t) == sizeof(uint32_t)) {
    neighbors_uint32 = reinterpret_cast<uint32_t*>(indices.data_handle());
  } else {
    neighbors_uint32_buf.resize(std::size_t(n_queries) * std::size_t(k),
                                resource::get_cuda_stream(handle));
    neighbors_uint32 = neighbors_uint32_buf.data();
  }

  raft::neighbors::ivf_flat::detail::ivfflat_interleaved_scan<
    data_t,
    typename raft::spatial::knn::detail::utils::config<data_t>::value_t,
    idx_t>(refinement_index,
           queries.data_handle(),
           fake_coarse_idx.data(),
           static_cast<uint32_t>(n_queries),
           0,
           refinement_index.metric(),
           1,
           k,
           0,
           chunk_index.data(),
           raft::distance::is_min_close(metric),
           raft::neighbors::filtering::none_ivf_sample_filter(),
           neighbors_uint32,
           distances.data_handle(),
           grid_dim_x,
           resource::get_cuda_stream(handle));

  // postprocessing -- neighbors from position to actual id
  ivf::detail::postprocess_neighbors(indices.data_handle(),
                                     neighbors_uint32,
                                     refinement_index.inds_ptrs().data_handle(),
                                     fake_coarse_idx.data(),
                                     chunk_index.data(),
                                     n_queries,
                                     1,
                                     k,
                                     resource::get_cuda_stream(handle));
}

}  // namespace raft::neighbors::detail
