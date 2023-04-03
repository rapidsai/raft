/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <raft/core/device_resources.hpp>
#include <raft/neighbors/cagra.cuh>

namespace raft::neighbors::experimental::cagra {

using DISTANCE_T = float;          // *** DO NOT CHANGE ***
using INDEX_T    = std::uint32_t;  // *** DO NOT CHANGE ***

#define RAFT_INST(DATA_T, IdxT, D_MEM_TYPE, G_MEM_TYPE)                                            \
  template void                                                                                    \
  prune<DATA_T,                                                                                    \
        IdxT,                                                                                      \
        host_device_accessor<std::experimental::default_accessor<DATA_T>, D_MEM_TYPE>,             \
        host_device_accessor<std::experimental::default_accessor<DATA_T>, G_MEM_TYPE>>(            \
    raft::device_resources const& res,                                                             \
    mdspan<const DATA_T,                                                                           \
           matrix_extent<IdxT>,                                                                    \
           row_major,                                                                              \
           host_device_accessor<std::experimental::default_accessor<DATA_T>, D_MEM_TYPE>> dataset, \
    mdspan<IdxT,                                                                                   \
           matrix_extent<IdxT>,                                                                    \
           row_major,                                                                              \
           host_device_accessor<std::experimental::default_accessor<DATA_T>, G_MEM_TYPE>>          \
      knn_graph,                                                                                   \
    raft::host_matrix_view<IdxT, IdxT, row_major> new_graph);

RAFT_INST(float, uint32_t, memory_type::host, memory_type::host);
RAFT_INST(float, uint32_t, memory_type::device, memory_type::host);

// RAFT_INST(uint8_t, uint32_t, memory_type::host, memory_type::host);
// RAFT_INST(uint8_t, uint32_t, memory_type::device, memory_type::host);

// RAFT_INST(int8_t, uint32_t, memory_type::host, memory_type::host);
// RAFT_INST(int8_t, uint32_t, memory_type::device, memory_type::host);
#undef RAFT_INST
}  // namespace raft::neighbors::experimental::cagra
