/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <raft/distance/distance_types.hpp>  // DistanceType
#include <raft/util/raft_explicit.hpp>       // RAFT_EXPLICIT

#include <cstddef>  // size_t
#include <cstdint>  // uint32_t

#if defined(RAFT_EXPLICIT_INSTANTIATE_ONLY)

namespace raft::spatial::knn::detail {

template <typename value_idx, typename value_t, bool usePrevTopKs = false>
void fusedL2Knn(size_t D,
                value_idx* out_inds,
                value_t* out_dists,
                const value_t* index,
                const value_t* query,
                size_t n_index_rows,
                size_t n_query_rows,
                int k,
                bool rowMajorIndex,
                bool rowMajorQuery,
                cudaStream_t stream,
                raft::distance::DistanceType metric,
                const value_t* index_norms = NULL,
                const value_t* query_norms = NULL) RAFT_EXPLICIT;

}  // namespace raft::spatial::knn::detail

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_spatial_knn_detail_fusedL2Knn(Mvalue_idx, Mvalue_t, MusePrevTopKs) \
  extern template void                                                                      \
  raft::spatial::knn::detail::fusedL2Knn<Mvalue_idx, Mvalue_t, MusePrevTopKs>(              \
    size_t D,                                                                               \
    Mvalue_idx * out_inds,                                                                  \
    Mvalue_t * out_dists,                                                                   \
    const Mvalue_t* index,                                                                  \
    const Mvalue_t* query,                                                                  \
    size_t n_index_rows,                                                                    \
    size_t n_query_rows,                                                                    \
    int k,                                                                                  \
    bool rowMajorIndex,                                                                     \
    bool rowMajorQuery,                                                                     \
    cudaStream_t stream,                                                                    \
    raft::distance::DistanceType metric,                                                    \
    const Mvalue_t* index_norms,                                                            \
    const Mvalue_t* query_norms);

instantiate_raft_spatial_knn_detail_fusedL2Knn(int32_t, float, true);
instantiate_raft_spatial_knn_detail_fusedL2Knn(int32_t, float, false);
instantiate_raft_spatial_knn_detail_fusedL2Knn(int64_t, float, true);
instantiate_raft_spatial_knn_detail_fusedL2Knn(int64_t, float, false);

// These are used by brute_force_knn:
instantiate_raft_spatial_knn_detail_fusedL2Knn(uint32_t, float, true);
instantiate_raft_spatial_knn_detail_fusedL2Knn(uint32_t, float, false);

#undef instantiate_raft_spatial_knn_detail_fusedL2Knn
