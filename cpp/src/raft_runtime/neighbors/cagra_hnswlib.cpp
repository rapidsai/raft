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

#include <raft/neighbors/cagra_hnswlib.hpp>
#include <raft_runtime/neighbors/cagra_hnswlib.hpp>

namespace raft::runtime::neighbors::cagra_hnswlib {

#define RAFT_INST_CAGRA_HNSWLIB(T)                                                       \
  void search(raft::resources const& handle,                                             \
              raft::neighbors::cagra_hnswlib::search_params const& params,               \
              const raft::neighbors::cagra_hnswlib::index<T>& index,                     \
              raft::host_matrix_view<const T, int64_t, row_major> queries,               \
              raft::host_matrix_view<uint64_t, int64_t, row_major> neighbors,            \
              raft::host_matrix_view<float, int64_t, row_major> distances)               \
  {                                                                                      \
    raft::neighbors::cagra_hnswlib::search<T>(                                           \
      handle, params, index, queries, neighbors, distances);                             \
  }                                                                                      \
  void deserialize_file(raft::resources const& handle,                                   \
                        const std::string& filename,                                     \
                        raft::neighbors::cagra_hnswlib::index<T>*& index,                \
                        int dim,                                                         \
                        raft::distance::DistanceType metric)                             \
  {                                                                                      \
    index = new raft::neighbors::cagra_hnswlib::hnswlib_index<T>(filename, dim, metric); \
    RAFT_EXPECTS(index, "Could not set index pointer");                                  \
  }

RAFT_INST_CAGRA_HNSWLIB(float);
RAFT_INST_CAGRA_HNSWLIB(int8_t);
RAFT_INST_CAGRA_HNSWLIB(uint8_t);

#undef RAFT_INST_CAGRA_HNSWLIB

}  // namespace raft::runtime::neighbors::cagra_hnswlib
