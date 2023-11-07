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

// XXX: the uint32_t instance is not compiled in libraft.so. So we allow
// instantiating the template here.
//
// TODO: consider removing this test or consider adding an instantiation to the
// library.
#undef RAFT_EXPLICIT_INSTANTIATE_ONLY

#include "../ann_ivf_pq.cuh"

namespace raft::neighbors::ivf_pq {

using f32_f32_u32        = ivf_pq_test<float, float, uint32_t>;
using f32_f32_u32_filter = ivf_pq_filter_test<float, float, uint32_t>;

TEST_BUILD_SEARCH(f32_f32_u32)
TEST_BUILD_SERIALIZE_SEARCH(f32_f32_u32)
INSTANTIATE(f32_f32_u32, defaults() + var_n_probes() + var_k() + special_cases());

TEST_BUILD_SEARCH(f32_f32_u32_filter)
INSTANTIATE(f32_f32_u32_filter, defaults());
}  // namespace raft::neighbors::ivf_pq
