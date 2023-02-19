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

#include <raft/matrix/detail/select_k.cuh>

#include <cuda_fp16.h>

namespace raft::matrix::detail {

#define RAFT_INST(T, IdxT)                                      \
  extern template void select_k<T, IdxT>(const T*,              \
                                         const IdxT*,           \
                                         size_t,                \
                                         size_t,                \
                                         int,                   \
                                         T*,                    \
                                         IdxT*,                 \
                                         bool,                  \
                                         rmm::cuda_stream_view, \
                                         rmm::mr::device_memory_resource*);

// Commonly used types
RAFT_INST(float, uint64_t);
RAFT_INST(half, uint64_t);

// These instances are used in the ivf_pq::search parameterized by the internal_distance_dtype
RAFT_INST(float, uint32_t);
RAFT_INST(half, uint32_t);

#undef RAFT_INST

}  // namespace raft::matrix::detail
