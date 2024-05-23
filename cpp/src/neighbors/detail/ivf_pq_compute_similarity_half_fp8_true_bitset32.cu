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

/*
 * NOTE: this file is generated by ivf_pq_compute_similarity_00_generate.py
 * Make changes there and run in this directory:
 * > python ivf_pq_compute_similarity_00_generate.py
 */

#include <raft/neighbors/detail/ivf_pq_compute_similarity_template.cuh>
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA uint32_t>>);
