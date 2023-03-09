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

void serialize(raft::device_resources const& handle,
               const std::string& filename,
               const raft::neighbors::ivf_pq::index<uint64_t>& index)
{
  raft::neighbors::ivf_pq::serialize(handle, filename, index);
};

}  // namespace raft::runtime::neighbors::ivf_pq
