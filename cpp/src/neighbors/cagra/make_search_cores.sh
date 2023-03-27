#!/bin/bash

# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

for max_dataset_dim in 128 256 512 1024 ; do
    for dtype in float half int8_t uint8_t ; do
	for team_size in 4 8 16 32 ; do
	    if [ $max_dataset_dim -gt 128 ] && [ $team_size -lt 8 ]; then
		continue
	    fi
	    if [ $max_dataset_dim -gt 256 ] && [ $team_size -lt 16 ]; then
		continue
	    fi
	    if [ $max_dataset_dim -gt 512 ] && [ $team_size -lt 32 ]; then
		continue
	    fi
	    echo "/*
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
#include \"raft/neighbors/detail/cagra/search_core.cuh\"

namespace raft::neighbors::experimental::cagra::detail {

template void create_plan<${dtype}, ${max_dataset_dim}, ${team_size}>(
    void **plan,
    const std::string search_mode,
    const std::size_t topk,
    const std::size_t itopk_size,
    const std::size_t num_parents,
    const std::size_t min_iterations,
    const std::size_t max_iterations,
    const std::size_t max_queries,
    const std::size_t load_bit_length,
    const std::size_t thread_block_size,
    const std::string hashmap_mode,
    const std::size_t hashmap_min_bitlen,
    const float hashmap_max_fill_rate,
    const std::size_t dataset_size,
    const std::size_t dataset_dim,
    const std::size_t graph_degree,
    const void* dev_dataset_ptr,   // device ptr, [dataset_size, dataset_dim]
    const INDEX_T* dev_graph_ptr   // device ptr, [dataset_size, graph_degree]
    );

template void search<${dtype}, ${max_dataset_dim}, ${team_size}>(
    void *plan,
    INDEX_T* dev_topk_indices_ptr,       // [num_queries, topk]
    DISTANCE_T* dev_topk_distances_ptr,  // [num_queries, topk]
    const void* dev_query_ptr,           // [num_queries, query_dim]
    const uint32_t num_queries,
    const uint32_t num_random_samplings,
    const uint64_t rand_xor_mask,
    const INDEX_T* dev_seed_ptr,   // [num_queries, num_seeds]
    const uint32_t num_seeds,
    uint32_t* num_executed_iterations,
    cudaStream_t cuda_stream
    );

template void destroy_plan<${dtype}, ${max_dataset_dim}, ${team_size}>(
    void *plan
    );
}
" > search_core_${dtype}_dim${max_dataset_dim}_t${team_size}.cu
	done
    done
done
