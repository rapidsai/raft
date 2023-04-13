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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/matrix/select_k.cuh>
#include <raft/matrix/specializations.cuh>

#include <raft_runtime/matrix/select_k.hpp>

#include <vector>

namespace raft::runtime::matrix {

void select_k(const device_resources& handle,
              raft::device_matrix_view<const float, int64_t, row_major> in_val,
              std::optional<raft::device_matrix_view<const int64_t, int64_t, row_major>> in_idx,
              raft::device_matrix_view<float, int64_t, row_major> out_val,
              raft::device_matrix_view<int64_t, int64_t, row_major> out_idx,
              bool select_min)
{
  raft::matrix::select_k(handle, in_val, in_idx, out_val, out_idx, select_min);
}
}  // namespace raft::runtime::matrix
