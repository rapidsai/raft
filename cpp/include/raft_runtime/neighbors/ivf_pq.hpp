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

#define RAFT_INST_SEARCH(T, IdxT)                                        \
  void search(raft::device_resources const&,                             \
              const raft::neighbors::ivf_pq::search_params&,             \
              const raft::neighbors::ivf_pq::index<IdxT>&,               \
              const T*,                                                  \
              uint32_t,                                                  \
              uint32_t,                                                  \
              IdxT*,                                                     \
              float*,                                                    \
              rmm::mr::device_memory_resource*);                         \
                                                                         \
  void search(raft::device_resources const&,                             \
              const raft::neighbors::ivf_pq::search_params&,             \
              const raft::neighbors::ivf_pq::index<IdxT>&,               \
              const raft::device_matrix_view<const T, IdxT, row_major>&, \
              uint32_t,                                                  \
              const raft::device_matrix_view<IdxT, IdxT, row_major>&,    \
              const raft::device_matrix_view<float, IdxT, row_major>&,   \
              rmm::mr::device_memory_resource*);

RAFT_INST_SEARCH(float, uint64_t);
RAFT_INST_SEARCH(int8_t, uint64_t);
RAFT_INST_SEARCH(uint8_t, uint64_t);

#undef RAFT_INST_SEARCH

// We define overloads for build and extend with void return type. This is used in the Cython
// wrappers, where exception handling is not compatible with return type that has nontrivial
// constructor.
#define RAFT_INST_BUILD_EXTEND(T, IdxT)                                                 \
  auto build(raft::device_resources const& handle,                                      \
             const raft::neighbors::ivf_pq::index_params& params,                       \
             const T* dataset,                                                          \
             IdxT n_rows,                                                               \
             uint32_t dim)                                                              \
    ->raft::neighbors::ivf_pq::index<IdxT>;                                             \
                                                                                        \
  auto extend(raft::device_resources const& handle,                                     \
              const raft::neighbors::ivf_pq::index<IdxT>& orig_index,                   \
              const T* new_vectors,                                                     \
              const IdxT* new_indices,                                                  \
              IdxT n_rows)                                                              \
    ->raft::neighbors::ivf_pq::index<IdxT>;                                             \
                                                                                        \
  void build(raft::device_resources const& handle,                                      \
             const raft::neighbors::ivf_pq::index_params& params,                       \
             const T* dataset,                                                          \
             IdxT n_rows,                                                               \
             uint32_t dim,                                                              \
             raft::neighbors::ivf_pq::index<IdxT>* idx);                                \
                                                                                        \
  void extend(raft::device_resources const& handle,                                     \
              raft::neighbors::ivf_pq::index<IdxT>* idx,                                \
              const T* new_vectors,                                                     \
              const IdxT* new_indices,                                                  \
              IdxT n_rows);                                                             \
  auto build(raft::device_resources const& handle,                                      \
             const raft::neighbors::ivf_pq::index_params& params,                       \
             const raft::device_matrix_view<const T, IdxT, row_major>& dataset)         \
    ->raft::neighbors::ivf_pq::index<IdxT>;                                             \
                                                                                        \
  auto extend(raft::device_resources const& handle,                                     \
              const raft::neighbors::ivf_pq::index<IdxT>& orig_index,                   \
              const raft::device_matrix_view<const T, IdxT, row_major>& new_vectors,    \
              const raft::device_matrix_view<const IdxT, IdxT, row_major>& new_indices) \
    ->raft::neighbors::ivf_pq::index<IdxT>;                                             \
                                                                                        \
  void build(raft::device_resources const& handle,                                      \
             const raft::neighbors::ivf_pq::index_params& params,                       \
             const raft::device_matrix_view<const T, IdxT, row_major>& dataset,         \
             raft::neighbors::ivf_pq::index<IdxT>* idx);                                \
                                                                                        \
  void extend(raft::device_resources const& handle,                                     \
              raft::neighbors::ivf_pq::index<IdxT>* idx,                                \
              const raft::device_matrix_view<const T, IdxT, row_major>& new_vectors,    \
              const raft::device_matrix_view<const IdxT, IdxT, row_major>& new_indices);

RAFT_INST_BUILD_EXTEND(float, uint64_t)
RAFT_INST_BUILD_EXTEND(int8_t, uint64_t)
RAFT_INST_BUILD_EXTEND(uint8_t, uint64_t)

#undef RAFT_INST_BUILD_EXTEND

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
void serialize(raft::device_resources const& handle,
               const std::string& filename,
               const raft::neighbors::ivf_pq::index<uint64_t>& index);

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
void deserialize(raft::device_resources const& handle,
                 const std::string& filename,
                 raft::neighbors::ivf_pq::index<uint64_t>* index);

}  // namespace raft::runtime::neighbors::ivf_pq
