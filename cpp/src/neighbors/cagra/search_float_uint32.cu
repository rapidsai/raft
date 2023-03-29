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
#include <raft/neighbors/cagra.cuh>

namespace raft::neighbors::experimental::cagra {

template void search<float, uint32_t>(
  raft::device_resources const& handle,
  const search_params& params,
  const index<float, uint32_t>& idx,
  raft::device_matrix_view<const float, uint32_t, row_major> queries,
  raft::device_matrix_view<uint32_t, uint32_t, row_major> neighbors,
  raft::device_matrix_view<float, uint32_t, row_major> distances);

}  // namespace raft::neighbors::experimental::cagra
