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

#include <raft/neighbors/ivf_pq.cuh>
#include <raft_runtime/neighbors/ivf_pq.hpp>

namespace raft::runtime::neighbors::ivf_pq {

#define RAFT_INST_BUILD_EXTEND(T, IdxT)                                                      \
  auto build(raft::device_resources const& handle,                                           \
             const raft::neighbors::ivf_pq::index_params& params,                            \
             const T* dataset,                                                               \
             IdxT n_rows,                                                                    \
             uint32_t dim)                                                                   \
    ->raft::neighbors::ivf_pq::index<IdxT>                                                   \
  {                                                                                          \
    return raft::neighbors::ivf_pq::build<T, IdxT>(handle, params, dataset, n_rows, dim);    \
  }                                                                                          \
  auto extend(raft::device_resources const& handle,                                          \
              const raft::neighbors::ivf_pq::index<IdxT>& orig_index,                        \
              const T* new_vectors,                                                          \
              const IdxT* new_indices,                                                       \
              IdxT n_rows)                                                                   \
    ->raft::neighbors::ivf_pq::index<IdxT>                                                   \
  {                                                                                          \
    return raft::neighbors::ivf_pq::extend<T, IdxT>(                                         \
      handle, orig_index, new_vectors, new_indices, n_rows);                                 \
  }                                                                                          \
                                                                                             \
  void build(raft::device_resources const& handle,                                           \
             const raft::neighbors::ivf_pq::index_params& params,                            \
             const T* dataset,                                                               \
             IdxT n_rows,                                                                    \
             uint32_t dim,                                                                   \
             raft::neighbors::ivf_pq::index<IdxT>* idx)                                      \
  {                                                                                          \
    *idx = raft::neighbors::ivf_pq::build<T, IdxT>(handle, params, dataset, n_rows, dim);    \
  }                                                                                          \
  void extend(raft::device_resources const& handle,                                          \
              raft::neighbors::ivf_pq::index<IdxT>* idx,                                     \
              const T* new_vectors,                                                          \
              const IdxT* new_indices,                                                       \
              IdxT n_rows)                                                                   \
  {                                                                                          \
    raft::neighbors::ivf_pq::extend<T, IdxT>(handle, idx, new_vectors, new_indices, n_rows); \
  }

RAFT_INST_BUILD_EXTEND(float, uint64_t);
RAFT_INST_BUILD_EXTEND(int8_t, uint64_t);
RAFT_INST_BUILD_EXTEND(uint8_t, uint64_t);

#undef RAFT_INST_BUILD_EXTEND

void serialize(raft::device_resources const& handle,
               const std::string& filename,
               const raft::neighbors::ivf_pq::index<uint64_t>& index)
{
  raft::spatial::knn::ivf_pq::detail::serialize(handle, filename, index);
};

void deserialize(raft::device_resources const& handle,
                 const std::string& filename,
                 raft::neighbors::ivf_pq::index<uint64_t>* index)
{
  if (!index) { RAFT_FAIL("Invalid index pointer"); }
  *index = raft::spatial::knn::ivf_pq::detail::deserialize<uint64_t>(handle, filename);
};
}  // namespace raft::runtime::neighbors::ivf_pq
