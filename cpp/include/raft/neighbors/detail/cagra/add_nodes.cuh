/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <raft/stats/histogram.cuh>

#include <rmm/device_buffer.hpp>

#include <cstdint>

// prototype declaration
namespace raft::neighbors::cagra {
template <typename T, typename IdxT>
void search(raft::resources const& res,
            const raft::neighbors::cagra::search_params& params,
            const raft::neighbors::cagra::index<T, IdxT>& idx,
            raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
            raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);
}

namespace raft::neighbors::cagra::detail {
template <class T, class IdxT, class Accessor>
void add_node_core(
  raft::resources const& handle,
  const raft::neighbors::cagra::index<T, IdxT>& idx,
  mdspan<const T, matrix_extent<int64_t>, raft::layout_stride, Accessor> additional_dataset_view,
  raft::host_matrix_view<IdxT, std::int64_t> updated_graph)
{
  using DistanceT                 = float;
  const std::size_t degree        = idx.graph_degree();
  const std::size_t dim           = idx.dim();
  const std::size_t old_size      = idx.dataset().extent(0);
  const std::size_t num_add       = additional_dataset_view.extent(0);
  const std::size_t new_size      = old_size + num_add;
  const std::uint32_t base_degree = degree * 2;

  // Step 0: Calculate the number of incoming edges for each node
  auto dev_num_incoming_edges = raft::make_device_vector<int, std::uint64_t>(handle, new_size);

  RAFT_CUDA_TRY(cudaMemsetAsync(dev_num_incoming_edges.data_handle(),
                                0,
                                sizeof(int) * new_size,
                                raft::resource::get_cuda_stream(handle)));
  raft::stats::histogram<IdxT, std::int64_t>(stats::HistTypeAuto,
                                             dev_num_incoming_edges.data_handle(),
                                             old_size,
                                             idx.graph().data_handle(),
                                             old_size * degree,
                                             1,
                                             raft::resource::get_cuda_stream(handle));

  auto host_num_incoming_edges = raft::make_host_vector<int, std::uint64_t>(new_size);
  raft::copy(host_num_incoming_edges.data_handle(),
             dev_num_incoming_edges.data_handle(),
             new_size,
             raft::resource::get_cuda_stream(handle));

  const std::size_t max_chunk_size = 1024;

  raft::neighbors::cagra::search_params params;
  params.itopk_size = std::max(base_degree * 2lu, 256lu);

  // Memory space for rank-based neighbor list
  auto mr = resource::get_workspace_resource(handle);

  auto neighbor_indices = raft::make_device_mdarray<IdxT, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_chunk_size, base_degree));

  auto neighbor_distances = raft::make_device_mdarray<DistanceT, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_chunk_size, base_degree));

  auto queries = raft::make_device_mdarray<T, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_chunk_size, dim));

  auto host_neighbor_indices =
    raft::make_host_matrix<IdxT, std::int64_t>(max_chunk_size, base_degree);

  auto neighbors_vectors = raft::make_device_mdarray<T, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_chunk_size * base_degree, dim));

  auto two_hop_neighbors_indices = raft::make_device_mdarray<IdxT, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_chunk_size * base_degree, base_degree));

  auto host_two_hop_neighbors_indices =
    raft::make_host_matrix<IdxT, std::int64_t>(max_chunk_size * base_degree, base_degree);

  auto two_hop_neighbors_distances = raft::make_device_mdarray<DistanceT, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_chunk_size * base_degree, base_degree));

  auto host_two_hop_neighbors_distances =
    raft::make_host_matrix<DistanceT, std::int64_t>(max_chunk_size * base_degree, base_degree);

  raft::spatial::knn::detail::utils::batch_load_iterator<T> additional_dataset_batch(
    additional_dataset_view.data_handle(),
    num_add,
    additional_dataset_view.stride(0),
    max_chunk_size,
    raft::resource::get_cuda_stream(handle),
    resource::get_workspace_resource(handle));
  for (const auto& batch : additional_dataset_batch) {
    // Step 1: Obtain K (=base_degree) nearest neighbors of the new vectors by CAGRA search
    // Create queries
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(queries.data_handle(),
                                    sizeof(T) * dim,
                                    batch.data(),
                                    sizeof(T) * additional_dataset_view.stride(0),
                                    sizeof(T) * dim,
                                    batch.size(),
                                    cudaMemcpyDefault,
                                    raft::resource::get_cuda_stream(handle)));

    const auto queries_view = raft::make_device_matrix_view<const T, std::int64_t>(
      queries.data_handle(), batch.size(), dim);

    auto neighbor_indices_view = raft::make_device_matrix_view<IdxT, std::int64_t>(
      neighbor_indices.data_handle(), batch.size(), base_degree);
    auto neighbor_distances_view = raft::make_device_matrix_view<float, std::int64_t>(
      neighbor_distances.data_handle(), batch.size(), base_degree);

    raft::neighbors::cagra::search(
      handle, params, idx, queries_view, neighbor_indices_view, neighbor_distances_view);

    raft::copy(host_neighbor_indices.data_handle(),
               neighbor_indices.data_handle(),
               batch.size() * base_degree,
               raft::resource::get_cuda_stream(handle));

    // Step 2: Obtain K (=base_degree) nearest neighbors of the neighbors of the new vectors by
    // CAGRA search
    for (std::uint32_t i = 0; i < batch.size(); i++) {
      for (std::uint32_t j = 0; j < base_degree; j++) {
        raft::copy(
          neighbors_vectors.data_handle() + (i * base_degree + j) * dim,
          idx.dataset().data_handle() +
            host_neighbor_indices.data_handle()[i * base_degree + j] * idx.dataset().stride(0),
          dim,
          raft::resource::get_cuda_stream(handle));
      }
    }

    raft::neighbors::cagra::search(
      handle, params, idx, queries_view, neighbor_indices_view, neighbor_distances_view);

    raft::copy(host_two_hop_neighbors_indices.data_handle(),
               two_hop_neighbors_indices.data_handle(),
               batch.size() * degree * degree,
               raft::resource::get_cuda_stream(handle));
    raft::resource::sync_stream(handle);

    // Step 3: rank-based reordering
    std::vector<std::pair<IdxT, std::size_t>> detourable_node_count_list(base_degree);
    for (std::size_t vec_i = 0; vec_i < batch.size(); vec_i++) {
      const auto host_two_hop_neighbors_indices_ptr =
        host_two_hop_neighbors_indices.data_handle() + vec_i * base_degree * base_degree;

      // Count detourable edges
      for (std::uint32_t i = 0; i < base_degree; i++) {
        std::uint32_t detourable_node_count = 0;
        const auto a_id                     = host_neighbor_indices(vec_i, i);
        for (std::uint32_t j = i + 1; j < base_degree; j++) {
          const auto b0_id = host_neighbor_indices(vec_i, j);
          for (std::uint32_t k = 0; k <= i; k++) {
            const auto b1_id = host_two_hop_neighbors_indices_ptr[i * base_degree + k];
            if (b0_id == b1_id) {
              detourable_node_count++;
              break;
            }
          }
        }
        detourable_node_count_list[i] = std::make_pair(a_id, detourable_node_count);
      }
      std::sort(detourable_node_count_list.begin(),
                detourable_node_count_list.end(),
                [&](const std::pair<IdxT, std::size_t> a, const std::pair<IdxT, std::size_t> b) {
                  return a.second < b.second;
                });

      for (std::size_t i = 0; i < degree; i++) {
        updated_graph.data_handle()[i + (old_size + batch.offset() + vec_i) * degree] =
          detourable_node_count_list[i].first;
      }
    }

    // Step 4: Add reverse edges
    const std::uint32_t rev_edge_search_range = degree / 2;
    const std::uint32_t num_rev_edges         = degree / 2;
    std::vector<IdxT> rev_edges(num_rev_edges), temp(degree);
    for (std::size_t vec_i = 0; vec_i < batch.size(); vec_i++) {
      // Create a reverse edge list
      const auto target_new_node_id = old_size + batch.offset() + vec_i;
      for (std::size_t i = 0; i < num_rev_edges; i++) {
        const auto target_node_id =
          updated_graph.data_handle()[i + (old_size + batch.offset() + vec_i) * degree];

        auto host_neighbor_indices_ptr = updated_graph.data_handle() + target_node_id * degree;

        IdxT replace_id                        = new_size;
        IdxT replace_id_j                      = 0;
        std::size_t replace_num_incoming_edges = 0;
        for (std::int32_t j = degree - 1; j >= static_cast<std::int32_t>(rev_edge_search_range);
             j--) {
          const auto neighbor_id               = host_neighbor_indices_ptr[j];
          const std::size_t num_incoming_edges = host_num_incoming_edges.data_handle(neighbor_id);
          if (num_incoming_edges > replace_num_incoming_edges) {
            // Check duplication
            bool dup = false;
            for (std::uint32_t k = 0; k < i; k++) {
              if (rev_edges[k] == neighbor_id) {
                dup = true;
                break;
              }
            }
            if (dup) { continue; }

            // Update rev edge candidate
            replace_num_incoming_edges = num_incoming_edges;
            replace_id                 = neighbor_id;
            replace_id_j               = j;
          }
        }
        if (replace_id >= new_size) {
          std::fprintf(stderr, "Invalid rev edge index (%u)\n", replace_id);
          return;
        }
        updated_graph(target_node_id, replace_id_j) = target_new_node_id;
        rev_edges[i]                                = replace_id;
      }
      host_num_incoming_edges(target_new_node_id) = num_rev_edges;

      // Create a neighbor list of a new node by interleaving the kNN neighbor list and reverse edge
      // list
      std::uint32_t interleave_switch = 0, rank_base_i = 0, rev_edges_return_i = 0, num_add = 0;
      const auto rank_based_list_ptr =
        updated_graph.data_handle() + (old_size + batch.offset() + vec_i) * degree;
      const auto rev_edges_return_list_ptr = rev_edges.data();
      while (num_add < degree) {
        const auto node_list_ptr =
          interleave_switch == 0 ? rank_based_list_ptr : rev_edges_return_list_ptr;
        auto& node_list_index          = interleave_switch == 0 ? rank_base_i : rev_edges_return_i;
        const auto max_node_list_index = interleave_switch == 0 ? degree : num_rev_edges;
        for (; node_list_index < max_node_list_index; node_list_index++) {
          const auto candidate = node_list_ptr[node_list_index];
          // Check duplication
          bool dup = false;
          for (std::uint32_t j = 0; j < num_add; j++) {
            if (temp[j] == candidate) {
              dup = true;
              break;
            }
          }
          if (!dup) {
            temp[num_add] = candidate;
            num_add++;
            break;
          }
        }
        interleave_switch = 1 - interleave_switch;
      }
      for (std::uint32_t i = 0; i < degree; i++) {
        updated_graph(target_new_node_id, i) = temp[i];
      }
    }
  }
}
}  // namespace raft::neighbors::cagra::detail
