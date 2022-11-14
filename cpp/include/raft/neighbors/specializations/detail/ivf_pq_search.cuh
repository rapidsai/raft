/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/spatial/knn/detail/ivf_pq_search.cuh>

#include <cuda_fp16.h>

namespace raft::spatial::knn::ivf_pq::detail {

namespace {
using fp8s_t = fp_8bit<5, true>;
using fp8u_t = fp_8bit<5, false>;
}  // namespace

#define RAFT_INST(IdxT, OutT, LutT)                                                           \
  extern template struct ivfpq_compute_similarity<IdxT, OutT, LutT>::configured<true, true>;  \
  extern template struct ivfpq_compute_similarity<IdxT, OutT, LutT>::configured<false, true>; \
  extern template struct ivfpq_compute_similarity<IdxT, OutT, LutT>::configured<true, false>;

#define RAFT_INST_ALL_IDX_T(OutT, LutT) \
  RAFT_INST(uint64_t, OutT, LutT)       \
  RAFT_INST(int64_t, OutT, LutT)        \
  RAFT_INST(uint32_t, OutT, LutT)

#define RAFT_INST_ALL_OUT_T(LutT)  \
  RAFT_INST_ALL_IDX_T(float, LutT) \
  RAFT_INST_ALL_IDX_T(half, LutT)

RAFT_INST_ALL_OUT_T(float)
RAFT_INST_ALL_OUT_T(half)
RAFT_INST_ALL_OUT_T(fp8s_t)
RAFT_INST_ALL_OUT_T(fp8u_t)

#undef RAFT_INST
#undef RAFT_INST_ALL_IDX_T
#undef RAFT_INST_ALL_OUT_T

#define RAFT_INST(T, IdxT)                                   \
  extern template void search<T, IdxT>(const handle_t&,      \
                                       const search_params&, \
                                       const index<IdxT>&,   \
                                       const T*,             \
                                       uint32_t,             \
                                       uint32_t,             \
                                       IdxT*,                \
                                       float*,               \
                                       rmm::mr::device_memory_resource*);

RAFT_INST(float, int64_t);
RAFT_INST(float, uint32_t);
RAFT_INST(float, uint64_t);

#undef RAFT_INST

}  // namespace raft::spatial::knn::ivf_pq::detail
