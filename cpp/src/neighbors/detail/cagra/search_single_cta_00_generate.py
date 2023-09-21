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

header = """
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

/*
 * NOTE: this file is generated by search_single_cta_00_generate.py
 *
 * Make changes there and run in this directory:
 *
 * > python search_single_cta_00_generate.py
 *
 */

#include <raft/neighbors/detail/cagra/search_single_cta_kernel-inl.cuh>
#include <raft/neighbors/sample_filter_types.hpp>

namespace raft::neighbors::cagra::detail::single_cta_search {

#define instantiate_single_cta_select_and_run(                                              \\
  TEAM_SIZE, MAX_DATASET_DIM, DATA_T, INDEX_T, DISTANCE_T, SAMPLE_FILTER_T)                 \\
  template void                                                                             \\
  select_and_run<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, INDEX_T, DISTANCE_T, SAMPLE_FILTER_T>( \\
    raft::device_matrix_view<const DATA_T, int64_t, layout_stride> dataset,                 \\
    raft::device_matrix_view<const INDEX_T, int64_t, row_major> graph,                      \\
    INDEX_T* const topk_indices_ptr,                                                        \\
    DISTANCE_T* const topk_distances_ptr,                                                   \\
    const DATA_T* const queries_ptr,                                                        \\
    const uint32_t num_queries,                                                             \\
    const INDEX_T* dev_seed_ptr,                                                            \\
    uint32_t* const num_executed_iterations,                                                \\
    uint32_t topk,                                                                          \\
    uint32_t num_itopk_candidates,                                                          \\
    uint32_t block_size,                                                                    \\
    uint32_t smem_size,                                                                     \\
    int64_t hash_bitlen,                                                                    \\
    INDEX_T* hashmap_ptr,                                                                   \\
    size_t small_hash_bitlen,                                                               \\
    size_t small_hash_reset_interval,                                                       \\
    uint32_t num_random_samplings,                                                          \\
    uint64_t rand_xor_mask,                                                                 \\
    uint32_t num_seeds,                                                                     \\
    size_t itopk_size,                                                                      \\
    size_t search_width,                                                                    \\
    size_t min_iterations,                                                                  \\
    size_t max_iterations,                                                                  \\
    SAMPLE_FILTER_T sample_filter,                                                          \\
    cudaStream_t stream);

"""

trailer = """
#undef instantiate_single_cta_search_kernel

}  // namespace raft::neighbors::cagra::detail::single_cta_search
"""

mxdim_team = [(128, 8), (256, 16), (512, 32), (1024, 32)]
# block = [(64, 16), (128, 8), (256, 4), (512, 2), (1024, 1)]
# itopk_candidates = [64, 128, 256]
# itopk_size = [64, 128, 256, 512]
# mxelem = [64, 128, 256]

# rblock = [(256, 4), (512, 2), (1024, 1)]
# rcandidates = [32]
# rsize = [256, 512]

search_types = dict(
    float_uint32=("float", "uint32_t", "float"),  # data_t, idx_t, distance_t
    int8_uint32=("int8_t", "uint32_t", "float"),
    uint8_uint32=("uint8_t", "uint32_t", "float"),
    float_uint64=("float", "uint64_t", "float"),
)

# knn
for type_path, (data_t, idx_t, distance_t) in search_types.items():
    for (mxdim, team) in mxdim_team:
        path = f"search_single_cta_{type_path}_dim{mxdim}_t{team}.cu"
        with open(path, "w") as f:
            f.write(header)
            f.write(
                f"instantiate_single_cta_select_and_run(\n  {team}, {mxdim}, {data_t}, {idx_t}, {distance_t}, raft::neighbors::filtering::removed_cagra_filter_with_offset_u32);\n"
            )

            f.write(trailer)
            # For pasting into CMakeLists.txt
            print(f"src/neighbors/detail/cagra/{path}")
