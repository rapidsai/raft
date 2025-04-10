/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <nccl.h>

#include <memory>

namespace raft::resource {

class nccl_comm_resource : public resource {
 public:
  nccl_comm_resource() : nccl_comm_(std::make_unique<ncclComm_t>()) {}
  ~nccl_comm_resource() override {}
  void* get_resource() override { return nccl_comm_.get(); }

 private:
  std::unique_ptr<ncclComm_t> nccl_comm_;
};

/** Factory that knows how to construct a specific raft::resource to populate the res_t. */
class nccl_comm_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::NCCL_COMM; }
  resource* make_resource() override { return new nccl_comm_resource(); }
};

/**
 * @defgroup ncclComm_t NCCL comm resource functions
 * @{
 */

/**
 * Load a NCCL comm from a res (and populate it on the res if needed).
 * @param res raft res object for managing resources
 * @return NCCL comm
 */
inline ncclComm_t& get_nccl_comm(const resources const& res)
{
  if (!res.has_resource_factory(resource_type::NCCL_COMM)) {
    res.add_resource_factory(std::make_shared<nccl_comm_resource_factory>());
  }
  return *res.get_resource<ncclComm_t>(resource_type::NCCL_COMM);
};

/**
 * @}
 */

}  // namespace raft::resource
