/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "../../cagra_types.hpp"
#include "graph_core.cuh"
#include <chrono>
#include <cstdio>
#include <raft/core/resource/cuda_stream.hpp>
#include <vector>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>

#include <raft/neighbors/detail/refine.cuh>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/refine.cuh>

namespace raft::neighbors::cagra::detail {

template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(raft::resources const& res,
                     mdspan<const DataT, matrix_extent<IdxT>, row_major, accessor> dataset,
                     raft::host_matrix_view<IdxT, IdxT, row_major> knn_graph,
                     std::optional<float> refine_rate                   = std::nullopt,
                     std::optional<ivf_pq::index_params> build_params   = std::nullopt,
                     std::optional<ivf_pq::search_params> search_params = std::nullopt)
{
  RAFT_EXPECTS(!build_params || build_params->metric == distance::DistanceType::L2Expanded,
               "Currently only L2Expanded metric is supported");

  uint32_t node_degree = knn_graph.extent(1);
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("cagra::build_graph(%zu, %zu, %u)",
                                                            size_t(dataset.extent(0)),
                                                            size_t(dataset.extent(1)),
                                                            node_degree);

  if (!build_params) {
    build_params          = ivf_pq::index_params{};
    build_params->n_lists = dataset.extent(0) < 4 * 2500 ? 4 : (uint32_t)(dataset.extent(0) / 2500);
    build_params->pq_dim  = raft::Pow2<8>::roundUp(dataset.extent(1) / 2);
    build_params->pq_bits = 8;
    build_params->kmeans_trainset_fraction = dataset.extent(0) < 10000 ? 1 : 10;
    build_params->kmeans_n_iters           = 25;
    build_params->add_data_on_build        = true;
  }

  // Make model name
  const std::string model_name = [&]() {
    char model_name[1024];
    sprintf(model_name,
            "%s-%lux%lu.cluster_%u.pq_%u.%ubit.itr_%u.metric_%u.pqcenter_%u",
            "IVF-PQ",
            static_cast<size_t>(dataset.extent(0)),
            static_cast<size_t>(dataset.extent(1)),
            build_params->n_lists,
            build_params->pq_dim,
            build_params->pq_bits,
            build_params->kmeans_n_iters,
            build_params->metric,
            static_cast<uint32_t>(build_params->codebook_kind));
    return std::string(model_name);
  }();

  RAFT_LOG_DEBUG("# Building IVF-PQ index %s", model_name.c_str());
  auto index = ivf_pq::build<DataT, int64_t>(
    res, *build_params, dataset.data_handle(), dataset.extent(0), dataset.extent(1));

  //
  // search top (k + 1) neighbors
  //
  if (!search_params) {
    search_params            = ivf_pq::search_params{};
    search_params->n_probes  = std::min<IdxT>(dataset.extent(1) * 2, build_params->n_lists);
    search_params->lut_dtype = CUDA_R_8U;
    search_params->internal_distance_dtype = CUDA_R_32F;
  }
  const auto top_k          = node_degree + 1;
  uint32_t gpu_top_k        = node_degree * refine_rate.value_or(2.0f);
  gpu_top_k                 = std::min<IdxT>(std::max(gpu_top_k, top_k), dataset.extent(0));
  const auto num_queries    = dataset.extent(0);
  const auto max_batch_size = 1024;
  RAFT_LOG_DEBUG(
    "IVF-PQ search node_degree: %d, top_k: %d,  gpu_top_k: %d,  max_batch_size:: %d, n_probes: %u",
    node_degree,
    top_k,
    gpu_top_k,
    max_batch_size,
    search_params->n_probes);

  // TODO(tfeher): shall we use uint32_t?
  auto distances = raft::make_device_matrix<float, int64_t>(res, max_batch_size, gpu_top_k);
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, max_batch_size, gpu_top_k);
  auto refined_distances = raft::make_device_matrix<float, int64_t>(res, max_batch_size, top_k);
  auto refined_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, max_batch_size, top_k);
  auto neighbors_host    = raft::make_host_matrix<int64_t, int64_t>(max_batch_size, gpu_top_k);
  auto queries_host = raft::make_host_matrix<DataT, int64_t>(max_batch_size, dataset.extent(1));
  auto refined_neighbors_host = raft::make_host_matrix<int64_t, int64_t>(max_batch_size, top_k);
  auto refined_distances_host = raft::make_host_matrix<float, int64_t>(max_batch_size, top_k);

  // TODO(tfeher): batched search with multiple GPUs
  std::size_t num_self_included = 0;
  bool first                    = true;
  const auto start_clock        = std::chrono::system_clock::now();

  rmm::mr::device_memory_resource* device_memory = nullptr;
  auto pool_guard = raft::get_pool_memory_resource(device_memory, 1024 * 1024);
  if (pool_guard) { RAFT_LOG_DEBUG("ivf_pq using pool memory resource"); }

  raft::spatial::knn::detail::utils::batch_load_iterator<DataT> vec_batches(
    dataset.data_handle(),
    dataset.extent(0),
    dataset.extent(1),
    max_batch_size,
    resource::get_cuda_stream(res),
    device_memory);

  size_t next_report_offset = 0;
  size_t d_report_offset    = dataset.extent(0) / 100;  // Report progress in 1% steps.

  for (const auto& batch : vec_batches) {
    auto queries_view = raft::make_device_matrix_view<const DataT, uint32_t>(
      batch.data(), batch.size(), batch.row_width());
    auto neighbors_view = make_device_matrix_view<int64_t, uint32_t>(
      neighbors.data_handle(), batch.size(), neighbors.extent(1));
    auto distances_view = make_device_matrix_view<float, uint32_t>(
      distances.data_handle(), batch.size(), distances.extent(1));

    ivf_pq::search(res, *search_params, index, queries_view, neighbors_view, distances_view);

    if constexpr (is_host_mdspan_v<decltype(dataset)>) {
      raft::copy(neighbors_host.data_handle(),
                 neighbors.data_handle(),
                 neighbors_view.size(),
                 resource::get_cuda_stream(res));
      raft::copy(queries_host.data_handle(),
                 batch.data(),
                 queries_view.size(),
                 resource::get_cuda_stream(res));
      auto queries_host_view = make_host_matrix_view<const DataT, int64_t>(
        queries_host.data_handle(), batch.size(), batch.row_width());
      auto neighbors_host_view = make_host_matrix_view<const int64_t, int64_t>(
        neighbors_host.data_handle(), batch.size(), neighbors.extent(1));
      auto refined_neighbors_host_view = make_host_matrix_view<int64_t, int64_t>(
        refined_neighbors_host.data_handle(), batch.size(), top_k);
      auto refined_distances_host_view = make_host_matrix_view<float, int64_t>(
        refined_distances_host.data_handle(), batch.size(), top_k);
      resource::sync_stream(res);

      raft::neighbors::detail::refine_host<int64_t, DataT, float, int64_t>(  // res,
        dataset,
        queries_host_view,
        neighbors_host_view,
        refined_neighbors_host_view,
        refined_distances_host_view,
        build_params->metric);
    } else {
      auto neighbor_candidates_view = make_device_matrix_view<const int64_t, uint64_t>(
        neighbors.data_handle(), batch.size(), gpu_top_k);
      auto refined_neighbors_view = make_device_matrix_view<int64_t, int64_t>(
        refined_neighbors.data_handle(), batch.size(), top_k);
      auto refined_distances_view = make_device_matrix_view<float, int64_t>(
        refined_distances.data_handle(), batch.size(), top_k);

      auto dataset_view = make_device_matrix_view<const DataT, int64_t>(
        dataset.data_handle(), dataset.extent(0), dataset.extent(1));
      raft::neighbors::detail::refine_device<int64_t, DataT, float, int64_t>(
        res,
        dataset_view,
        queries_view,
        neighbor_candidates_view,
        refined_neighbors_view,
        refined_distances_view,
        build_params->metric);
      raft::copy(refined_neighbors_host.data_handle(),
                 refined_neighbors_view.data_handle(),
                 refined_neighbors_view.size(),
                 resource::get_cuda_stream(res));
      resource::sync_stream(res);
    }
    // omit itself & write out
    // TODO(tfeher): do this in parallel with GPU processing of next batch
    for (std::size_t i = 0; i < batch.size(); i++) {
      size_t vec_idx = i + batch.offset();
      for (std::size_t j = 0, num_added = 0; j < top_k && num_added < node_degree; j++) {
        const auto v = refined_neighbors_host(i, j);
        if (static_cast<size_t>(v) == vec_idx) {
          num_self_included++;
          continue;
        }
        knn_graph(vec_idx, num_added) = v;
        num_added++;
      }
    }

    size_t num_queries_done = batch.offset() + batch.size();
    const auto end_clock    = std::chrono::system_clock::now();
    if (batch.offset() > next_report_offset) {
      next_report_offset += d_report_offset;
      const auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() *
        1e-6;
      const auto throughput = num_queries_done / time;

      RAFT_LOG_INFO(
        "# Search %12lu / %12lu (%3.2f %%), %e queries/sec, %.2f minutes ETA, self included = "
        "%3.2f %%    \r",
        num_queries_done,
        dataset.extent(0),
        num_queries_done / static_cast<double>(dataset.extent(0)) * 100,
        throughput,
        (num_queries - num_queries_done) / throughput / 60,
        static_cast<double>(num_self_included) / num_queries_done * 100.);
    }
    first = false;
  }
  if (!first) RAFT_LOG_DEBUG("# Finished building kNN graph");
}

}  // namespace raft::neighbors::cagra::detail
