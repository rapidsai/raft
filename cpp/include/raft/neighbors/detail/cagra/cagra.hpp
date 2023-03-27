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

// TODO(tfeher): remove this and create a corresponding raft_runtime header
namespace raft::neighbors::experimental::cagra::detail {

using DISTANCE_T = float;          // *** DO NOT CHANGE ***
using INDEX_T    = std::uint32_t;  // *** DO NOT CHANGE ***

//
// Optimize a kNN graph.
//
// Keep important edges, remove unnecessary edges, and add important reverse
// edges. Both input and output graphs are unidirectional with a fixed number
// of edges, or degree.
//
void prune_graph(
  const std::string dtype_name,           // Data type of dataset. "float", "int8", or "uint8".
  const std::size_t dataset_size,         // Number of vectors in the dataset.
  const std::size_t dataset_dim,          // Dimensionality of vectors in the dataset.
  const std::size_t input_graph_degree,   // Degree of input graph.
  const std::size_t output_graph_degree,  // Degree of output graph.
  void* dataset_ptr,                      // Host pointer, [dataset_size, dataset_dim]
  INDEX_T* input_graph_ptr,               // Host pointer, [dataset_size, input_graph_degree]
  INDEX_T* output_graph_ptr               // Host pointer, [dataset_size, output_graph_degree]
);

//
// Create a search plan.
//
// Created plan can be used repeatedly as long as the search parameters are not
// changed. The workspace to be used during the search is allocated and retained
// internally when the plan is created.
//
// namespace internal {

void create_plan_dispatch(
  void** plan,                   // Descriptor of search plan created.
  const std::string dtype_name,  // Data type of dataset. "float", "half", "int8", or "uint8".
  const std::size_t
    team_size,  // Number of threads used to calculate a single distance. 4, 8, 16, or 32.
  const std::string search_mode,  // Search algorithm. "single-cta", "multi-cta", or "multi-kernel".
  const std::size_t topk,         // Number of search results for each query.
  const std::size_t
    itopk_size,  // Number of intermediate search results retained during the search.
  const std::size_t num_parents,  // Number of graph nodes to select as the starting point for the
                                  // search in each iteration.
  const std::size_t min_iterations,  // Lower limit of search iterations.
  const std::size_t max_iterations,  // Upper limit of search iterations.
  const std::size_t
    max_queries,  // Maximum number of queries to search at the same time. So called batch size.
  const std::size_t load_bit_length,  // Bit length for reading the dataset vectors. 0, 64 or 128.
                                      // Auto selection when 0.
  const std::size_t
    thread_block_size,  // Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0.
  const std::string
    hashmap_mode,  // Hashmap type. "auto", "hash", or "small-hash". Auto selection when "auto".
  const std::size_t hashmap_min_bitlen,  // Lower limit of hashmap bit length. More than 8.
  const float
    hashmap_max_fill_rate,  // Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
  const std::size_t dataset_size,  // Number of vectors in the dataset.
  const std::size_t dataset_dim,   // Dimensionality of vectors in the dataset.
  const std::size_t graph_degree,  // Degree of graph.
  const void* dev_dataset_ptr,     // Device pointer, [dataset_size, dataset_dim]
  const INDEX_T* dev_graph_ptr     // Device pointer, [dataset_size, graph_degree]
);

//
//
void search_dispatch(
  void* plan,                     // Descriptor of search plan.
  INDEX_T* dev_topk_indices_ptr,  // Device pointer, [num_queries, topk]. Search results (indices).
  DISTANCE_T*
    dev_topk_distances_ptr,    // Device pointer, [num_queries, topk]. Search results (distances).
  const void* dev_query_ptr,   // Device pointer, [num_queries, query_dim]. Query vectors.
  const uint32_t num_queries,  // Number of query vectors.
  const uint32_t
    num_random_samplings,  // Number of iterations of initial random seed node selection. 1 or more.
  const uint64_t rand_xor_mask,       // Bit mask used for initial random seed node selection.
  const INDEX_T* dev_seed_ptr,        // Device pointer, [num_queries, num_seeds]. Usually, nullptr.
  const uint32_t num_seeds,           // Number of specified seed nodes. Usually, 0.
  uint32_t* num_executed_iterations,  // Stats. Number of iterations needed for each query search.
  cudaStream_t cuda_stream            // CUDA stream.
);

//
// Destroy a search plan.
//
// Internally allocated workspaces are freed at this time.
//
void destroy_plan_dispatch(void* plan  // Descriptor of search plan
);
//}  // namespace internal
}  // namespace raft::neighbors::experimental::cagra::detail
