/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <raft/spatial/knn/detail/fused_l2_knn.cuh>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

//template void fusedL2Knn<raft::distance::DistanceType::L2SqrtUnexpanded,
//                                std::int64_t, float, true>(
//  std::size_t D, std::int64_t *out_inds, float *out_dists, const float *index,
//  const float *query, std::size_t n_index_rows, std::size_t n_query_rows, int k,
//  bool rowMajorIndex, bool rowMajorQuery, cudaStream_t stream, void *workspace,
//  std::size_t &worksize);
//
//template void fusedL2Knn<raft::distance::DistanceType::L2Unexpanded,
//                                std::int64_t, float, true>(
//  std::size_t D, std::int64_t *out_inds, float *out_dists, const float *index,
//  const float *query, std::size_t n_index_rows, std::size_t n_query_rows, int k,
//  bool rowMajorIndex, bool rowMajorQuery, cudaStream_t stream, void *workspace,
//  std::size_t &worksize);
//
//template void fusedL2Knn<raft::distance::DistanceType::L2SqrtUnexpanded,
//                                int, float, true>(
//  std::size_t D, int *out_inds, float *out_dists, const float *index,
//  const float *query, std::size_t n_index_rows, std::size_t n_query_rows, int k,
//  bool rowMajorIndex, bool rowMajorQuery, cudaStream_t stream, void *workspace,
//  std::size_t &worksize);
//
//template void
//fusedL2Knn<raft::distance::DistanceType::L2Unexpanded, int, float, true>(
//  std::size_t D, int *out_inds, float *out_dists, const float *index,
//  const float *query, std::size_t n_index_rows, std::size_t n_query_rows, int k,
//  bool rowMajorIndex, bool rowMajorQuery, cudaStream_t stream, void *workspace,
//  std::size_t &worksize);
//
//template void fusedL2Knn<raft::distance::DistanceType::L2SqrtUnexpanded,
//                                std::int64_t, float, false>(
//  std::size_t D, std::int64_t *out_inds, float *out_dists, const float *index,
//  const float *query, std::size_t n_index_rows, std::size_t n_query_rows, int k,
//  bool rowMajorIndex, bool rowMajorQuery, cudaStream_t stream, void *workspace,
//  std::size_t &worksize);
//
//template void fusedL2Knn<raft::distance::DistanceType::L2Unexpanded,
//                                std::int64_t, float, false>(
//  std::size_t D, std::int64_t *out_inds, float *out_dists, const float *index,
//  const float *query, std::size_t n_index_rows, std::size_t n_query_rows, int k,
//  bool rowMajorIndex, bool rowMajorQuery, cudaStream_t stream, void *workspace,
//  std::size_t &worksize);
//
//template void fusedL2Knn<raft::distance::DistanceType::L2SqrtUnexpanded,
//                                int, float, false>(
//  std::size_t D, int *out_inds, float *out_dists, const float *index,
//  const float *query, std::size_t n_index_rows, std::size_t n_query_rows, int k,
//  bool rowMajorIndex, bool rowMajorQuery, cudaStream_t stream, void *workspace,
//  std::size_t &worksize);
//
//template void fusedL2Knn<raft::distance::DistanceType::L2Unexpanded, int,
//                                float, false>(
//  std::size_t D, int *out_inds, float *out_dists, const float *index,
//  const float *query, std::size_t n_index_rows, std::size_t n_query_rows, int k,
//  bool rowMajorIndex, bool rowMajorQuery, cudaStream_t stream, void *workspace,
//  std::size_t &worksize);

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft
