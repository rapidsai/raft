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

#include <raft/neighbors/specializations/detail/ivf_pq_search.cuh>

#include <cuda_fp16.h>

namespace raft::spatial::knn::ivf_pq::detail {

template struct ivfpq_compute_similarity<uint64_t, float, half>::configured<true, true>;

template struct ivfpq_compute_similarity<uint64_t, half, half>::configured<true, true>;

}  // namespace raft::spatial::knn::ivf_pq::detail
