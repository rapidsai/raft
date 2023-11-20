/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cstddef>                           // size_t
#include <cstdint>                           // int_Xt
#include <raft/distance/distance_types.hpp>  // DistanceType
#include <raft/spatial/knn/detail/fused_l2_knn-inl.cuh>

#define instantiate_raft_spatial_knn_detail_fusedL2Knn(Mvalue_idx, Mvalue_t, MusePrevTopKs)  \
  template void raft::spatial::knn::detail::fusedL2Knn<Mvalue_idx, Mvalue_t, MusePrevTopKs>( \
    size_t D,                                                                                \
    Mvalue_idx * out_inds,                                                                   \
    Mvalue_t * out_dists,                                                                    \
    const Mvalue_t* index,                                                                   \
    const Mvalue_t* query,                                                                   \
    size_t n_index_rows,                                                                     \
    size_t n_query_rows,                                                                     \
    int k,                                                                                   \
    bool rowMajorIndex,                                                                      \
    bool rowMajorQuery,                                                                      \
    cudaStream_t stream,                                                                     \
    raft::distance::DistanceType metric,                                                     \
    const Mvalue_t* index_norms,                                                             \
    const Mvalue_t* query_norms)

instantiate_raft_spatial_knn_detail_fusedL2Knn(int64_t, float, true);
instantiate_raft_spatial_knn_detail_fusedL2Knn(int64_t, float, false);

#undef instantiate_raft_spatial_knn_detail_fusedL2Knn
