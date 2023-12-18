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

#include <filesystem>
#include <raft/neighbors/hnsw.hpp>
#include <raft/neighbors/hnsw_serialize.cuh>

#include <raft_runtime/neighbors/cagra.hpp>
#include <raft_runtime/neighbors/hnsw.hpp>

namespace raft::neighbors::hnsw {
#define RAFT_INST_HNSW(T)                                                               \
  template <>                                                                           \
  std::unique_ptr<raft::neighbors::hnsw::index<T>> from_cagra(                          \
    raft::resources const& res, raft::neighbors::cagra::index<T, uint32_t> cagra_index) \
  {                                                                                     \
    std::string filepath = "/tmp/cagra_index.bin";                                      \
    raft::runtime::neighbors::cagra::serialize_to_hnswlib(res, filepath, cagra_index);  \
    auto hnsw_index = raft::runtime::neighbors::hnsw::deserialize_file<T>(              \
      res, filepath, cagra_index.dim(), cagra_index.metric());                          \
    std::filesystem::remove(filepath);                                                  \
    return hnsw_index;                                                                  \
  }

RAFT_INST_HNSW(float);
RAFT_INST_HNSW(int8_t);
RAFT_INST_HNSW(uint8_t);
#undef RAFT_INST_HNSW
}  // namespace raft::neighbors::hnsw

namespace raft::runtime::neighbors::hnsw {

#define RAFT_INST_HNSW(T)                                                                   \
  void search(raft::resources const& handle,                                                \
              raft::neighbors::hnsw::search_params const& params,                           \
              const raft::neighbors::hnsw::index<T>& index,                                 \
              raft::host_matrix_view<const T, int64_t, row_major> queries,                  \
              raft::host_matrix_view<uint64_t, int64_t, row_major> neighbors,               \
              raft::host_matrix_view<float, int64_t, row_major> distances)                  \
  {                                                                                         \
    raft::neighbors::hnsw::search<T>(handle, params, index, queries, neighbors, distances); \
  }                                                                                         \
                                                                                            \
  template <>                                                                               \
  std::unique_ptr<raft::neighbors::hnsw::index<T>> deserialize_file(                        \
    raft::resources const& handle,                                                          \
    const std::string& filename,                                                            \
    int dim,                                                                                \
    raft::distance::DistanceType metric)                                                    \
  {                                                                                         \
    return raft::neighbors::hnsw::deserialize<T>(handle, filename, dim, metric);            \
  }

RAFT_INST_HNSW(float);
RAFT_INST_HNSW(int8_t);
RAFT_INST_HNSW(uint8_t);

#undef RAFT_INST_HNSW

}  // namespace raft::runtime::neighbors::hnsw
