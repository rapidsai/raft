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

#include <raft/neighbors/specializations.cuh>

namespace raft::neighbors::ivf_flat {

#define RAFT_MAKE_INSTANCE(T, IdxT)                                          \
  template void raft::neighbors::ivf_flat::detail::ivfflat_interleaved_scan< \
    T,                                                                       \
    typename raft::spatial::knn::detail::utils::config<T>::value_t,          \
    IdxT>(const index<T, IdxT>& index,                                       \
          const T* queries,                                                  \
          const uint32_t* coarse_query_results,                              \
          const uint32_t n_queries,                                          \
          const raft::distance::DistanceType metric,                         \
          const uint32_t n_probes,                                           \
          const uint32_t k,                                                  \
          const bool select_min,                                             \
          IdxT* neighbors,                                                   \
          float* distances,                                                  \
          uint32_t& grid_dim_x,                                              \
          rmm::cuda_stream_view stream);                                     \
                                                                             \
  template void search(raft::device_resources const&,                        \
                       raft::neighbors::ivf_flat::search_params const&,      \
                       const raft::neighbors::ivf_flat::index<T, IdxT>&,     \
                       raft::device_matrix_view<const T, IdxT, row_major>,   \
                       raft::device_matrix_view<IdxT, IdxT, row_major>,      \
                       raft::device_matrix_view<float, IdxT, row_major>);

RAFT_MAKE_INSTANCE(int8_t, int64_t);

#undef RAFT_MAKE_INSTANCE

}  // namespace raft::neighbors::ivf_flat
