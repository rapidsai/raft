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
#pragma once

#include <cstddef>                           // size_t
#include <raft/distance/distance_types.hpp>  // DistanceType
#include <raft/util/raft_explicit.hpp>       // RAFT_EXPLICIT

#if defined(RAFT_EXPLICIT_INSTANTIATE)

namespace raft::spatial::knn::detail {
/**
 * Compute the k-nearest neighbors using L2 expanded/unexpanded distance.

 * @tparam value_idx
 * @tparam value_t
 * @param[out] out_inds output indices array on device (size n_query_rows * k)
 * @param[out] out_dists output dists array on device (size n_query_rows * k)
 * @param[in] index input index array on device (size n_index_rows * D)
 * @param[in] query input query array on device (size n_query_rows * D)
 * @param[in] n_index_rows number of rows in index array
 * @param[in] n_query_rows number of rows in query array
 * @param[in] k number of closest neighbors to return
 * @param[in] rowMajorIndex are the index arrays in row-major layout?
 * @param[in] rowMajorQuery are the query array in row-major layout?
 * @param[in] stream stream to order kernel launch
 */
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
                raft::distance::DistanceType metric) RAFT_EXPLICIT;

}  // namespace raft::spatial::knn::detail

#endif  // RAFT_EXPLICIT_INSTANTIATE

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
    raft::distance::DistanceType metric)

instantiate_raft_spatial_knn_detail_fusedL2Knn(long, float, true);
instantiate_raft_spatial_knn_detail_fusedL2Knn(long, float, false);
instantiate_raft_spatial_knn_detail_fusedL2Knn(int, float, true);
instantiate_raft_spatial_knn_detail_fusedL2Knn(int, float, false);

// These are used by brute_force_knn:
instantiate_raft_spatial_knn_detail_fusedL2Knn(unsigned int, float, true);
instantiate_raft_spatial_knn_detail_fusedL2Knn(unsigned int, float, false);

#undef instantiate_raft_spatial_knn_detail_fusedL2Knn
