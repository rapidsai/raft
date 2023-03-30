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

#include <raft/cluster/specializations.cuh>
#include <raft/distance/specializations.cuh>
#include <raft/matrix/specializations.cuh>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/specializations/detail/ivf_pq_compute_similarity.cuh>

namespace raft::neighbors::ivf_pq {

#ifdef RAFT_DECL_BUILD_EXTEND
#undef RAFT_DECL_BUILD_EXTEND
#endif

#ifdef RAFT_DECL_SEARCH
#undef RAFT_DECL_SEARCH
#endif

// We define overloads for build and extend with void return type. This is used in the Cython
// wrappers, where exception handling is not compatible with return type that has nontrivial
// constructor.
#define RAFT_DECL_BUILD_EXTEND(T, IdxT)                                          \
  extern template auto build(raft::device_resources const&,                      \
                             const raft::neighbors::ivf_pq::index_params&,       \
                             raft::device_matrix_view<const T, IdxT, row_major>) \
    ->raft::neighbors::ivf_pq::index<IdxT>;                                      \
                                                                                 \
  extern template auto extend(                                                   \
    raft::device_resources const&,                                               \
    raft::device_matrix_view<const T, IdxT, row_major>,                          \
    std::optional<raft::device_vector_view<const IdxT, IdxT>>,                   \
    const raft::neighbors::ivf_pq::index<IdxT>&)                                 \
    ->raft::neighbors::ivf_pq::index<IdxT>;                                      \
                                                                                 \
  extern template void extend(                                                   \
    raft::device_resources const&,                                               \
    raft::device_matrix_view<const T, IdxT, row_major>,                          \
    std::optional<raft::device_vector_view<const IdxT, IdxT>>,                   \
    raft::neighbors::ivf_pq::index<IdxT>*);

RAFT_DECL_BUILD_EXTEND(float, int64_t)
RAFT_DECL_BUILD_EXTEND(int8_t, int64_t)
RAFT_DECL_BUILD_EXTEND(uint8_t, int64_t)

#undef RAFT_DECL_BUILD_EXTEND

#define RAFT_DECL_SEARCH(T, IdxT)                                                 \
  extern template void search(raft::device_resources const&,                      \
                              const raft::neighbors::ivf_pq::search_params&,      \
                              const raft::neighbors::ivf_pq::index<IdxT>&,        \
                              raft::device_matrix_view<const T, IdxT, row_major>, \
                              raft::device_matrix_view<IdxT, IdxT, row_major>,    \
                              raft::device_matrix_view<float, IdxT, row_major>);

RAFT_DECL_SEARCH(float, int64_t);
RAFT_DECL_SEARCH(int8_t, int64_t);
RAFT_DECL_SEARCH(uint8_t, int64_t);

#undef RAFT_DECL_SEARCH

}  // namespace raft::neighbors::ivf_pq
