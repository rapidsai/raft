/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/comms/nccl_clique.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <memory>

namespace raft::resource {

class nccl_clique_resource : public resource {
 public:
  nccl_clique_resource() : clique_() {}
  ~nccl_clique_resource() noexcept override {}
  void* get_resource() override { return &clique_; }

 private:
  raft::comms::nccl_clique clique_;
};

/** Factory that knows how to construct a specific raft::resource to populate the res_t. */
class nccl_clique_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::NCCL_CLIQUE; }
  resource* make_resource() override { return new nccl_clique_resource(); }
};

/**
 * @defgroup nccl_clique_resource resource functions
 * @{
 */

/**
 * Retrieves a NCCL clique from raft res if it exists, otherwise initializes it and return it.
 *
 * @param[in] res the raft resources object
 * @return NCCL clique
 */
inline raft::comms::nccl_clique get_nccl_clique_handle(resources const& res)
{
  if (!res.has_resource_factory(resource_type::NCCL_CLIQUE)) {
    res.add_resource_factory(std::make_shared<nccl_clique_resource_factory>());
  }
  auto ret = *res.get_resource<raft::comms::nccl_clique>(resource_type::NCCL_CLIQUE);
  return ret;
};

/**
 * @}
 */

}  // namespace raft::resource
