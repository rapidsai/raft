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

#include <raft/neighbors/ivf_flat.cuh>
#include <raft_runtime/neighbors/ivf_flat.hpp>

namespace raft::runtime::neighbors::ivf_flat {

#define RAFT_INST_BUILD_EXTEND(T, IdxT)                                                \
  auto build(raft::resources const& handle,                                            \
             const raft::neighbors::ivf_flat::index_params& params,                    \
             raft::device_matrix_view<const T, IdxT, row_major> dataset)               \
    ->raft::neighbors::ivf_flat::index<T, IdxT>                                        \
  {                                                                                    \
    return raft::neighbors::ivf_flat::build<T, IdxT>(handle, params, dataset);         \
  }                                                                                    \
  auto extend(raft::resources const& handle,                                           \
              raft::device_matrix_view<const T, IdxT, row_major> new_vectors,          \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,   \
              const raft::neighbors::ivf_flat::index<T, IdxT>& orig_index)             \
    ->raft::neighbors::ivf_flat::index<T, IdxT>                                        \
  {                                                                                    \
    return raft::neighbors::ivf_flat::extend<T, IdxT>(                                 \
      handle, new_vectors, new_indices, orig_index);                                   \
  }                                                                                    \
                                                                                       \
  void build(raft::resources const& handle,                                            \
             const raft::neighbors::ivf_flat::index_params& params,                    \
             raft::device_matrix_view<const T, IdxT, row_major> dataset,               \
             raft::neighbors::ivf_flat::index<T, IdxT>& idx)                           \
  {                                                                                    \
    idx = build(handle, params, dataset);                                              \
  }                                                                                    \
                                                                                       \
  void extend(raft::resources const& handle,                                           \
              raft::device_matrix_view<const T, IdxT, row_major> new_vectors,          \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,   \
              raft::neighbors::ivf_flat::index<T, IdxT>* idx)                          \
  {                                                                                    \
    raft::neighbors::ivf_flat::extend<T, IdxT>(handle, new_vectors, new_indices, idx); \
  }

RAFT_INST_BUILD_EXTEND(float, int64_t);
RAFT_INST_BUILD_EXTEND(int8_t, int64_t);
RAFT_INST_BUILD_EXTEND(uint8_t, int64_t);

#undef RAFT_INST_BUILD_EXTEND

}  // namespace raft::runtime::neighbors::ivf_flat
