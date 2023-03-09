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

#include <raft/neighbors/specializations/ivf_flat.cuh>
#include <raft_runtime/neighbors/ivf_flat.hpp>

namespace raft::runtime::neighbors::ivf_flat {

#define RAFT_INST_BUILD_EXTEND(T, IdxT)                                                \
  auto build(raft::device_resources const& handle,                                     \
             raft::device_matrix_view<const T, uint64_t, row_major> dataset,           \
             const raft::neighbors::ivf_flat::index_params& params)                    \
    ->raft::neighbors::ivf_flat::index<T, IdxT>                                        \
  {                                                                                    \
    return raft::neighbors::ivf_flat::build<T, IdxT>(handle, dataset, params);         \
  }                                                                                    \
  auto extend(raft::device_resources const& handle,                                    \
              const raft::neighbors::ivf_flat::index<T, IdxT>& orig_index,             \
              raft::device_matrix_view<const T, IdxT, row_major> new_vectors,          \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices)   \
    ->raft::neighbors::ivf_flat::index<T, IdxT>                                        \
  {                                                                                    \
    return raft::neighbors::ivf_flat::extend<T, IdxT>(                                 \
      handle, orig_index, new_vectors, new_indices);                                   \
  }                                                                                    \
                                                                                       \
  void build(raft::device_resources const& handle,                                     \
             raft::device_matrix_view<const T, uint64_t, row_major> dataset,           \
             const raft::neighbors::ivf_flat::index_params& params,                    \
             raft::neighbors::ivf_flat::index<T, IdxT>* idx)                           \
  {                                                                                    \
    *idx = raft::neighbors::ivf_flat::build<T, IdxT>(handle, dataset, params);         \
  }                                                                                    \
                                                                                       \
  void extend(raft::device_resources const& handle,                                    \
              raft::neighbors::ivf_flat::index<T, IdxT>* idx,                          \
              raft::device_matrix_view<const T, IdxT, row_major> new_vectors,          \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices)   \
  {                                                                                    \
    raft::neighbors::ivf_flat::extend<T, IdxT>(handle, idx, new_vectors, new_indices); \
  }

RAFT_INST_BUILD_EXTEND(float, uint64_t);
RAFT_INST_BUILD_EXTEND(int8_t, uint64_t);
RAFT_INST_BUILD_EXTEND(uint8_t, uint64_t);

#undef RAFT_INST_BUILD_EXTEND

// void save(raft::device_resources const& handle,
//           const std::string& filename,
//           const raft::neighbors::ivf_flat::index<T, uint64_t>& index)
// {
//   raft::spatial::knn::ivf_flat::detail::save(handle, filename, index);
// };

// void load(raft::device_resources const& handle,
//           const std::string& filename,
//           raft::neighbors::ivf_flat::index<T, uint64_t>* index)
// {
//   if (!index) { RAFT_FAIL("Invalid index pointer"); }
//   *index = raft::spatial::knn::ivf_flat::detail::load<T, uint64_t>(handle, filename);
// };
}  // namespace raft::runtime::neighbors::ivf_flat
