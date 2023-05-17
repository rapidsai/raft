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

#include <raft/neighbors/ivf_pq_types.hpp>

namespace raft::runtime::neighbors::ivf_pq {

// We define overloads for build and extend with void return type. This is used in the Cython
// wrappers, where exception handling is not compatible with return type that has nontrivial
// constructor.
#define RAFT_DECL_BUILD_EXTEND(T, IdxT)                                              \
  [[nodiscard]] raft::neighbors::ivf_pq::index<IdxT> build(                          \
    raft::resources const& handle,                                                   \
    const raft::neighbors::ivf_pq::index_params& params,                             \
    raft::device_matrix_view<const T, IdxT, row_major> dataset);                     \
                                                                                     \
  void build(raft::resources const& handle,                                          \
             const raft::neighbors::ivf_pq::index_params& params,                    \
             raft::device_matrix_view<const T, IdxT, row_major> dataset,             \
             raft::neighbors::ivf_pq::index<IdxT>* idx);                             \
                                                                                     \
  [[nodiscard]] raft::neighbors::ivf_pq::index<IdxT> extend(                         \
    raft::resources const& handle,                                                   \
    raft::device_matrix_view<const T, IdxT, row_major> new_vectors,                  \
    std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,           \
    const raft::neighbors::ivf_pq::index<IdxT>& idx);                                \
                                                                                     \
  void extend(raft::resources const& handle,                                         \
              raft::device_matrix_view<const T, IdxT, row_major> new_vectors,        \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices, \
              raft::neighbors::ivf_pq::index<IdxT>* idx);

RAFT_DECL_BUILD_EXTEND(float, int64_t);
RAFT_DECL_BUILD_EXTEND(int8_t, int64_t);
RAFT_DECL_BUILD_EXTEND(uint8_t, int64_t);

#undef RAFT_DECL_BUILD_EXTEND

#define RAFT_DECL_SEARCH(T, IdxT)                                         \
  void search(raft::resources const& handle,                              \
              const raft::neighbors::ivf_pq::search_params& params,       \
              const raft::neighbors::ivf_pq::index<IdxT>& idx,            \
              raft::device_matrix_view<const T, IdxT, row_major> queries, \
              raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,  \
              raft::device_matrix_view<float, IdxT, row_major> distances);

RAFT_DECL_SEARCH(float, int64_t);
RAFT_DECL_SEARCH(int8_t, int64_t);
RAFT_DECL_SEARCH(uint8_t, int64_t);

#undef RAFT_DECL_SEARCH

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the filename for saving the index
 * @param[in] index IVF-PQ index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const raft::neighbors::ivf_pq::index<int64_t>& index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index IVF-PQ index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 raft::neighbors::ivf_pq::index<int64_t>* index);

}  // namespace raft::runtime::neighbors::ivf_pq
