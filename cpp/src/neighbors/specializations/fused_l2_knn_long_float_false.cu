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

#include <cstdint>
#include <raft/neighbors/specializations.cuh>
#include <raft/spatial/knn/detail/fused_l2_knn.cuh>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

template void fusedL2Knn<long, float, false>(size_t D,
                                             long* out_inds,
                                             float* out_dists,
                                             const float* index,
                                             const float* query,
                                             size_t n_index_rows,
                                             size_t n_query_rows,
                                             int k,
                                             bool rowMajorIndex,
                                             bool rowMajorQuery,
                                             cudaStream_t stream,
                                             raft::distance::DistanceType metric);
};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft
