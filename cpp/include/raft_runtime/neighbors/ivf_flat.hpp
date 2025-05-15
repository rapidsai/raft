/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/neighbors/ivf_flat_types.hpp>

#include <string>

namespace raft::runtime::neighbors::ivf_flat {

// We define overloads for build and extend with void return type. This is used in the Cython
// wrappers, where exception handling is not compatible with return type that has nontrivial
// constructor.
#define RAFT_INST_BUILD_EXTEND(T, IdxT)                                              \
  auto build(raft::resources const& handle,                                          \
             const raft::neighbors::ivf_flat::index_params& params,                  \
             raft::device_matrix_view<const T, IdxT, row_major> dataset)             \
    -> raft::neighbors::ivf_flat::index<T, IdxT>;                                    \
                                                                                     \
  auto extend(raft::resources const& handle,                                         \
              raft::device_matrix_view<const T, IdxT, row_major> new_vectors,        \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices, \
              const raft::neighbors::ivf_flat::index<T, IdxT>& orig_index)           \
    -> raft::neighbors::ivf_flat::index<T, IdxT>;                                    \
                                                                                     \
  void build(raft::resources const& handle,                                          \
             const raft::neighbors::ivf_flat::index_params& params,                  \
             raft::device_matrix_view<const T, IdxT, row_major> dataset,             \
             raft::neighbors::ivf_flat::index<T, IdxT>& idx);                        \
                                                                                     \
  void extend(raft::resources const& handle,                                         \
              raft::device_matrix_view<const T, IdxT, row_major> new_vectors,        \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices, \
              raft::neighbors::ivf_flat::index<T, IdxT>* idx);                       \
                                                                                     \
  void serialize_file(raft::resources const& handle,                                 \
                      const std::string& filename,                                   \
                      const raft::neighbors::ivf_flat::index<T, IdxT>& index);       \
                                                                                     \
  void deserialize_file(raft::resources const& handle,                               \
                        const std::string& filename,                                 \
                        raft::neighbors::ivf_flat::index<T, IdxT>* index);           \
  void serialize(raft::resources const& handle,                                      \
                 std::string& str,                                                   \
                 const raft::neighbors::ivf_flat::index<T, IdxT>& index);            \
  void deserialize(raft::resources const& handle,                                    \
                   const std::string& str,                                           \
                   raft::neighbors::ivf_flat::index<T, IdxT>*);

RAFT_INST_BUILD_EXTEND(float, int64_t)
RAFT_INST_BUILD_EXTEND(int8_t, int64_t)
RAFT_INST_BUILD_EXTEND(uint8_t, int64_t)

#undef RAFT_INST_BUILD_EXTEND

#define RAFT_INST_SEARCH(T, IdxT)                                 \
  void search(raft::resources const&,                             \
              raft::neighbors::ivf_flat::search_params const&,    \
              raft::neighbors::ivf_flat::index<T, IdxT> const&,   \
              raft::device_matrix_view<const T, IdxT, row_major>, \
              raft::device_matrix_view<IdxT, IdxT, row_major>,    \
              raft::device_matrix_view<float, IdxT, row_major>);

RAFT_INST_SEARCH(float, int64_t);
RAFT_INST_SEARCH(int8_t, int64_t);
RAFT_INST_SEARCH(uint8_t, int64_t);

#undef RAFT_INST_SEARCH

}  // namespace raft::runtime::neighbors::ivf_flat
