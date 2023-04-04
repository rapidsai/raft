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

#pragma once

#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/detail/cagra/search_multi_cta.cuh>
#include <raft/neighbors/detail/cagra/search_multi_kernel.cuh>
#include <raft/neighbors/detail/cagra/search_single_cta.cuh>

namespace raft::neighbors::experimental::cagra {

// todo(tfeher): add build_knn_graph

#define RAFT_INST(T, IdxT, MEM)                                                        \
  extern template auto                                                                 \
  build<T, IdxT, host_device_accessor<std::experimental::default_accessor<T>, MEM>>(   \
    raft::device_resources const& handle,                                              \
    const index_params& params,                                                        \
    mdspan<const T,                                                                    \
           matrix_extent<IdxT>,                                                        \
           row_major,                                                                  \
           host_device_accessor<std::experimental::default_accessor<T>, MEM>> dataset) \
    ->index<T, IdxT>;

RAFT_INST(float, uint32_t, memory_type::host);
RAFT_INST(float, uint32_t, memory_type::device);

#undef RAFT_INST

#define RAFT_INST(DATA_T, IdxT, D_MEM_TYPE, G_MEM_TYPE)                                            \
  extern template void                                                                             \
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


#undef RAFT_INST

#define RAFT_INST(T, IdxT)                                      \
  extern template void search<T, IdxT>(                         \
    raft::device_resources const& handle,                       \
    const search_params& params,                                \
    const index<T, IdxT>& idx,                                  \
    raft::device_matrix_view<const T, IdxT, row_major> queries, \
    raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,  \
    raft::device_matrix_view<float, IdxT, row_major> distances);

// RAFT_INST(float, uint32_t)
#undef RAFT_INST
}  // namespace raft::neighbors::experimental::cagra
namespace raft::neighbors::experimental::cagra::detail::single_cta_search {
extern template struct search<8, 128, float, uint32_t, float>;
extern template struct search<16, 256, float, uint32_t, float>;
extern template struct search<32, 512, float, uint32_t, float>;
extern template struct search<32, 1024, float, uint32_t, float>;
}  // namespace raft::neighbors::experimental::cagra::detail::single_cta_search

namespace raft::neighbors::experimental::cagra::detail::multi_cta_search {
extern template struct search<8, 128, float, uint32_t, float>;
extern template struct search<16, 256, float, uint32_t, float>;
extern template struct search<32, 512, float, uint32_t, float>;
extern template struct search<32, 1024, float, uint32_t, float>;
}  // namespace raft::neighbors::experimental::cagra::detail::multi_cta_search
namespace raft::neighbors::experimental::cagra::detail::multi_kernel_search {
extern template struct search<8, 128, float, uint32_t, float>;
extern template struct search<16, 256, float, uint32_t, float>;
extern template struct search<32, 512, float, uint32_t, float>;
extern template struct search<32, 1024, float, uint32_t, float>;
}  // namespace raft::neighbors::experimental::cagra::detail::multi_kernel_search
