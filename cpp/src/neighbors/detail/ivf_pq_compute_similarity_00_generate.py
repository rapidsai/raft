#!/usr/bin/env python3

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

#include <raft/neighbors/detail/ivf_pq_compute_similarity-inl.cuh>
#include <raft/neighbors/detail/ivf_pq_fp_8bit.cuh>

#define instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(OutT, LutT) \\
    template auto raft::neighbors::ivf_pq::detail::compute_similarity_select<OutT, LutT>( \\
        const cudaDeviceProp& dev_props,                                \\
        bool manage_local_topk,                                         \\
        int locality_hint,                                              \\
        double preferred_shmem_carveout,                                \\
        uint32_t pq_bits,                                               \\
        uint32_t pq_dim,                                                \\
        uint32_t precomp_data_count,                                    \\
        uint32_t n_queries,                                             \\
        uint32_t n_probes,                                              \\
        uint32_t topk) -> raft::neighbors::ivf_pq::detail::selected<OutT, LutT>;

#define COMMA ,
"""

trailer = """
#undef COMMA

#undef instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select
"""

types = dict(
    half_fp8_false=("half", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>"),
    half_fp8_true=("half", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>"),
    half_half=("half", "half"),
    float_half=("float", "half"),
    float_float= ("float", "float"),
    float_fp8_false=("float", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>"),
    float_fp8_true=("float", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>"),
)

for path_key, (OutT, LutT) in types.items():
    path = f"ivf_pq_compute_similarity_{path_key}.cu"
    with open(path, "w") as f:
        f.write(header)
        f.write(f"instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select({OutT}, {LutT});\n")
        f.write(trailer)
    print(f"src/neighbors/detail/{path}")
