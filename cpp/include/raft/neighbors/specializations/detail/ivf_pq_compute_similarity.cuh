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

#pragma once

#include <raft/neighbors/detail/ivf_pq_search.cuh>

#include <cuda_fp16.h>

namespace raft::neighbors::ivf_pq::detail {

namespace {
using fp8s_t = fp_8bit<5, true>;
using fp8u_t = fp_8bit<5, false>;
}  // namespace

#define RAFT_INST(OutT, LutT)                                                                     \
  extern template auto get_compute_similarity_kernel<OutT, LutT, true, true>(uint32_t, uint32_t)  \
    ->compute_similarity_kernel_t<OutT, LutT>;                                                    \
  extern template auto get_compute_similarity_kernel<OutT, LutT, true, false>(uint32_t, uint32_t) \
    ->compute_similarity_kernel_t<OutT, LutT>;                                                    \
  extern template auto get_compute_similarity_kernel<OutT, LutT, false, true>(uint32_t, uint32_t) \
    ->compute_similarity_kernel_t<OutT, LutT>;

#define RAFT_INST_ALL_OUT_T(LutT) \
  RAFT_INST(float, LutT)          \
  RAFT_INST(half, LutT)

RAFT_INST_ALL_OUT_T(float)
RAFT_INST_ALL_OUT_T(half)
RAFT_INST_ALL_OUT_T(fp8s_t)
RAFT_INST_ALL_OUT_T(fp8u_t)

#undef RAFT_INST
#undef RAFT_INST_ALL_OUT_T

}  // namespace raft::neighbors::ivf_pq::detail
