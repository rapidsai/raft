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

#include <raft/neighbors/specializations/detail/ivf_pq_compute_similarity.cuh>

#include <cuda_fp16.h>

namespace raft::neighbors::ivf_pq::detail {

template auto get_compute_similarity_kernel<half, fp_8bit<5, true>, true, true>(uint32_t, uint32_t)
  -> compute_similarity_kernel_t<half, fp_8bit<5, true>>;

}  // namespace raft::neighbors::ivf_pq::detail
