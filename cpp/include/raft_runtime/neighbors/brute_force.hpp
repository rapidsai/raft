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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace raft::runtime::neighbors::brute_force {

#define RAFT_INST_BFKNN(IDX_T, DATA_T, MATRIX_IDX_T, INDEX_LAYOUT, SEARCH_LAYOUT)        \
  void knn(raft::resources const& handle,                                                \
           raft::device_matrix_view<const DATA_T, MATRIX_IDX_T, INDEX_LAYOUT> index,     \
           raft::device_matrix_view<const DATA_T, MATRIX_IDX_T, SEARCH_LAYOUT> search,   \
           raft::device_matrix_view<IDX_T, MATRIX_IDX_T, row_major> indices,             \
           raft::device_matrix_view<DATA_T, MATRIX_IDX_T, row_major> distances,          \
           distance::DistanceType metric         = distance::DistanceType::L2Unexpanded, \
           std::optional<float> metric_arg       = std::make_optional<float>(2.0f),      \
           std::optional<IDX_T> global_id_offset = std::nullopt);

RAFT_INST_BFKNN(int64_t, float, int64_t, raft::row_major, raft::row_major);

#undef RAFT_INST_BFKNN

}  // namespace raft::runtime::neighbors::brute_force
