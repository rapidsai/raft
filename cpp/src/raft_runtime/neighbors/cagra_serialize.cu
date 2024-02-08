/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <sstream>
#include <string>

#include <raft/core/device_resources.hpp>
#include <raft/neighbors/cagra_serialize.cuh>
#include <raft/neighbors/cagra_types.hpp>
#include <raft_runtime/neighbors/cagra.hpp>

namespace raft::runtime::neighbors::cagra {

#define RAFT_INST_CAGRA_SERIALIZE(DTYPE)                                                      \
  void serialize_file(raft::resources const& handle,                                          \
                      const std::string& filename,                                            \
                      const raft::neighbors::cagra::index<DTYPE, uint32_t>& index,            \
                      bool include_dataset)                                                   \
  {                                                                                           \
    raft::neighbors::cagra::serialize(handle, filename, index, include_dataset);              \
  };                                                                                          \
                                                                                              \
  void deserialize_file(raft::resources const& handle,                                        \
                        const std::string& filename,                                          \
                        raft::neighbors::cagra::index<DTYPE, uint32_t>* index)                \
  {                                                                                           \
    if (!index) { RAFT_FAIL("Invalid index pointer"); }                                       \
    *index = raft::neighbors::cagra::deserialize<DTYPE, uint32_t>(handle, filename);          \
  };                                                                                          \
  void serialize(raft::resources const& handle,                                               \
                 std::string& str,                                                            \
                 const raft::neighbors::cagra::index<DTYPE, uint32_t>& index,                 \
                 bool include_dataset)                                                        \
  {                                                                                           \
    std::stringstream os;                                                                     \
    raft::neighbors::cagra::serialize(handle, os, index, include_dataset);                    \
    str = os.str();                                                                           \
  }                                                                                           \
                                                                                              \
  void serialize_to_hnswlib_file(raft::resources const& handle,                               \
                                 const std::string& filename,                                 \
                                 const raft::neighbors::cagra::index<DTYPE, uint32_t>& index) \
  {                                                                                           \
    raft::neighbors::cagra::serialize_to_hnswlib(handle, filename, index);                    \
  };                                                                                          \
  void serialize_to_hnswlib(raft::resources const& handle,                                    \
                            std::string& str,                                                 \
                            const raft::neighbors::cagra::index<DTYPE, uint32_t>& index)      \
  {                                                                                           \
    std::stringstream os;                                                                     \
    raft::neighbors::cagra::serialize_to_hnswlib(handle, os, index);                          \
    str = os.str();                                                                           \
  }                                                                                           \
                                                                                              \
  void deserialize(raft::resources const& handle,                                             \
                   const std::string& str,                                                    \
                   raft::neighbors::cagra::index<DTYPE, uint32_t>* index)                     \
  {                                                                                           \
    std::istringstream is(str);                                                               \
    if (!index) { RAFT_FAIL("Invalid index pointer"); }                                       \
    *index = raft::neighbors::cagra::deserialize<DTYPE, uint32_t>(handle, is);                \
  }

RAFT_INST_CAGRA_SERIALIZE(float);
RAFT_INST_CAGRA_SERIALIZE(int8_t);
RAFT_INST_CAGRA_SERIALIZE(uint8_t);

#undef RAFT_INST_CAGRA_SERIALIZE
}  // namespace raft::runtime::neighbors::cagra
