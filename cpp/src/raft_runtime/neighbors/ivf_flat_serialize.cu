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

#include <raft/neighbors/ivf_flat_serialize.cuh>
#include <raft/neighbors/ivf_flat_types.cuh>

#include <raft_runtime/neighbors/ivf_flat.hpp>

namespace raft::runtime::neighbors::ivf_flat {

#define RAFT_IVF_FLAT_SERIALIZE_INST(DTYPE)                                            \
  void serialize(raft::device_resources const& handle,                                 \
                 const std::string& filename,                                          \
                 const raft::neighbors::ivf_flat::index<DTYPE, int64_t>& index)        \
  {                                                                                    \
    raft::neighbors::ivf_pq::serialize(handle, filename, index);                       \
  };                                                                                   \
                                                                                       \
  void deserialize(raft::device_resources const& handle,                               \
                   const std::string& filename,                                        \
                   raft::neighbors::ivf_flat::index<DTYPE, int64_t>* index)            \
  {                                                                                    \
    if (!index) { RAFT_FAIL("Invalid index pointer"); }                                \
    *index = raft::neighbors::ivf_flat::deserialize<DTYPE, int64_t>(handle, filename); \
  };

RAFT_IVF_FLAT_SERIALIZE_INST(float);
RAFT_IVF_FLAT_SERIALIZE_INST(int8);
RAFT_IVF_FLAT_SERIALIZE_INST(uint8);

#undef RAFT_IVF_FLAT_SERIALIZE_INST
}  // namespace raft::runtime::neighbors::ivf_flat
