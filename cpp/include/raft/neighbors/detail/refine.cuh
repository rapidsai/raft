/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/neighbors/detail/ivf_flat_build.cuh>
#include <raft/neighbors/detail/ivf_flat_search.cuh>
#include <raft/spatial/knn/detail/ann_utils.cuh>

#include <cstdlib>
#include <omp.h>

#include <thrust/sequence.h>

namespace raft::neighbors::detail {

/** Checks whether the input data extents are compatible. */
template <typename extents_t>
void check_input(extents_t dataset,
                 extents_t queries,
                 extents_t candidates,
                 extents_t indices,
                 extents_t distances,
                 distance::DistanceType metric)
{
  auto n_queries = queries.extent(0);
  auto k         = distances.extent(1);

  RAFT_EXPECTS(k <= raft::matrix::detail::select::warpsort::kMaxCapacity,
               "k must be lest than topk::kMaxCapacity (%d).",
               raft::matrix::detail::select::warpsort::kMaxCapacity);

  RAFT_EXPECTS(indices.extent(0) == n_queries && distances.extent(0) == n_queries &&
                 candidates.extent(0) == n_queries,
               "Number of rows in output indices, distances and candidates matrices must be equal"
               " with the number of rows in search matrix. Expected %d, got %d, %d, and %d",
               static_cast<int>(n_queries),
               static_cast<int>(indices.extent(0)),
               static_cast<int>(distances.extent(0)),
               static_cast<int>(candidates.extent(0)));

  RAFT_EXPECTS(indices.extent(1) == k,
               "Number of columns in output indices and distances matrices must be equal to k");

  RAFT_EXPECTS(queries.extent(1) == dataset.extent(1),
               "Number of columns must be equal for dataset and queries");

  RAFT_EXPECTS(candidates.extent(1) >= k,
               "Number of neighbor candidates must not be smaller than k (%d vs %d)",
               static_cast<int>(candidates.extent(1)),
               static_cast<int>(k));
}

/**
 * See raft::neighbors::refine for docs.
 */
template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine_device(raft::device_resources const& handle,
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

  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "neighbors::refine(%zu, %u)", size_t(n_queries), uint32_t(n_candidates));

  check_input(dataset.extents(),
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
  rmm::device_uvector<uint32_t> fake_coarse_idx(n_queries, handle.get_stream());

  thrust::sequence(
    handle.get_thrust_policy(), fake_coarse_idx.data(), fake_coarse_idx.data() + n_queries);

  raft::neighbors::ivf_flat::index<data_t, idx_t> refinement_index(
    handle, metric, n_queries, false, true, dim);

  raft::neighbors::ivf_flat::detail::fill_refinement_index(handle,
                                                           &refinement_index,
                                                           dataset.data_handle(),
                                                           neighbor_candidates.data_handle(),
                                                           n_queries,
                                                           n_candidates);

  // greppable-id-specializations-ivf-flat-search: The ivfflat_interleaved_scan
  // function is used in both raft::neighbors::ivf_flat::search and
  // raft::neighbors::detail::refine_device. To prevent a duplicate
  // instantiation of this function (which defines ~270 kernels) in the refine
  // specializations, an extern template definition is provided. Please check
  // and adjust the extern template definition and the instantiation when the
  // below function call is edited. Search for
  // `greppable-id-specializations-ivf-flat-search` to find them.
  uint32_t grid_dim_x = 1;
  raft::neighbors::ivf_flat::detail::ivfflat_interleaved_scan<
    data_t,
    typename raft::spatial::knn::detail::utils::config<data_t>::value_t,
    idx_t>(refinement_index,
           queries.data_handle(),
           fake_coarse_idx.data(),
           static_cast<uint32_t>(n_queries),
           refinement_index.metric(),
           1,
           k,
           raft::distance::is_min_close(metric),
           indices.data_handle(),
           distances.data_handle(),
           grid_dim_x,
           handle.get_stream());
}

/** Helper structure for naive CPU implementation of refine. */
typedef struct {
  uint64_t id;
  float distance;
} struct_for_refinement;

inline int _postprocessing_qsort_compare(const void* v1, const void* v2)
{
  // sort in ascending order
  if (((struct_for_refinement*)v1)->distance > ((struct_for_refinement*)v2)->distance) {
    return 1;
  } else if (((struct_for_refinement*)v1)->distance < ((struct_for_refinement*)v2)->distance) {
    return -1;
  } else {
    return 0;
  }
}

/**
 * Naive CPU implementation of refine operation
 *
 * All pointers are expected to be accessible on the host.
 */
template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine_host(raft::host_matrix_view<const data_t, matrix_idx, row_major> dataset,
                 raft::host_matrix_view<const data_t, matrix_idx, row_major> queries,
                 raft::host_matrix_view<const idx_t, matrix_idx, row_major> neighbor_candidates,
                 raft::host_matrix_view<idx_t, matrix_idx, row_major> indices,
                 raft::host_matrix_view<distance_t, matrix_idx, row_major> distances,
                 distance::DistanceType metric = distance::DistanceType::L2Unexpanded)
{
  check_input(dataset.extents(),
              queries.extents(),
              neighbor_candidates.extents(),
              indices.extents(),
              distances.extents(),
              metric);

  switch (metric) {
    case raft::distance::DistanceType::L2Expanded: break;
    case raft::distance::DistanceType::InnerProduct: break;
    default: throw raft::logic_error("Unsopported metric");
  }

  size_t numDataset            = dataset.extent(0);
  size_t numQueries            = queries.extent(0);
  size_t dimDataset            = dataset.extent(1);
  const data_t* dataset_ptr    = dataset.data_handle();
  const data_t* queries_ptr    = queries.data_handle();
  const idx_t* neighbors       = neighbor_candidates.data_handle();
  idx_t topK                   = neighbor_candidates.extent(1);
  idx_t refinedTopK            = indices.extent(1);
  idx_t* refinedNeighbors      = indices.data_handle();
  distance_t* refinedDistances = distances.data_handle();

  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "neighbors::refine_host(%zu, %u)", size_t(numQueries), uint32_t(topK));

#pragma omp parallel
  {
    struct_for_refinement* sfr =
      (struct_for_refinement*)malloc(sizeof(struct_for_refinement) * topK);
    for (size_t i = omp_get_thread_num(); i < numQueries; i += omp_get_num_threads()) {
      // compute distance with original dataset vectors
      const data_t* cur_query = queries_ptr + ((uint64_t)dimDataset * i);
      for (size_t j = 0; j < (size_t)topK; j++) {
        idx_t id                  = neighbors[j + (topK * i)];
        const data_t* cur_dataset = dataset_ptr + ((uint64_t)dimDataset * id);
        float distance            = 0.0;
        for (size_t k = 0; k < (size_t)dimDataset; k++) {
          float val_q = (float)(cur_query[k]);
          float val_d = (float)(cur_dataset[k]);
          if (metric == raft::distance::DistanceType::InnerProduct) {
            distance += -val_q * val_d;  // Negate because we sort in ascending order.
          } else {
            distance += (val_q - val_d) * (val_q - val_d);
          }
        }
        sfr[j].id       = id;
        sfr[j].distance = distance;
      }

      qsort(sfr, topK, sizeof(struct_for_refinement), _postprocessing_qsort_compare);

      for (size_t j = 0; j < (size_t)refinedTopK; j++) {
        refinedNeighbors[j + (refinedTopK * i)] = sfr[j].id;
        if (refinedDistances == NULL) continue;
        if (metric == raft::distance::DistanceType::InnerProduct) {
          refinedDistances[j + (refinedTopK * i)] = -sfr[j].distance;
        } else {
          refinedDistances[j + (refinedTopK * i)] = sfr[j].distance;
        }
      }
    }
    free(sfr);
  }
}

}  // namespace raft::neighbors::detail
