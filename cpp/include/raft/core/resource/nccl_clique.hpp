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

#include <raft/core/nccl_clique.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <memory>

namespace raft::resource {

class nccl_clique_resource : public resource {
 public:
  nccl_clique_resource(std::optional<std::vector<int>>& device_ids, int percent_of_free_memory)
  {
    if (device_ids.has_value()) {
      clique_ = std::make_unique<raft::core::nccl_clique>(*device_ids, percent_of_free_memory);
    } else {
      clique_ = std::make_unique<raft::core::nccl_clique>(percent_of_free_memory);
    }
  }

  ~nccl_clique_resource() override {}
  void* get_resource() override { return clique_.get(); }

 private:
  std::unique_ptr<raft::core::nccl_clique> clique_;
};

/** Factory that knows how to construct a specific raft::resource to populate the res_t. */
class nccl_clique_resource_factory : public resource_factory {
 public:
  nccl_clique_resource_factory(const std::optional<std::vector<int>>& device_ids,
                               int percent_of_free_memory)
    : device_ids(device_ids), percent_of_free_memory(percent_of_free_memory)
  {
  }

  resource_type get_resource_type() override { return resource_type::NCCL_CLIQUE; }
  resource* make_resource() override
  {
    return new nccl_clique_resource(this->device_ids, this->percent_of_free_memory);
  }

  std::optional<std::vector<int>> device_ids;
  int percent_of_free_memory;
};

inline const raft::core::nccl_clique& build_nccl_clique(
  resources const& res,
  const std::optional<std::vector<int>>& device_ids,
  int percent_of_free_memory)
{
  if (!res.has_resource_factory(resource_type::NCCL_CLIQUE)) {
    res.add_resource_factory(
      std::make_shared<nccl_clique_resource_factory>(device_ids, percent_of_free_memory));
  } else {
    RAFT_LOG_WARN("Attempted re-initialize the NCCL clique on a RAFT resource.");
  }
  return *res.get_resource<raft::core::nccl_clique>(resource_type::NCCL_CLIQUE);
}

/**
 * @defgroup nccl_clique_resource resource functions
 * @{
 */

/**
 * Initializes a NCCL clique and sets it into a raft resource instance
 *
 * @param[in] res the raft resources object
 * @param[in] percent_of_free_memory percentage of device memory to pre-allocate as a memory pool on
 * each GPU
 * @return NCCL clique
 */
inline const raft::core::nccl_clique& initialize_nccl_clique(resources const& res,
                                                             int percent_of_free_memory = 80)
{
  return build_nccl_clique(res, std::nullopt, percent_of_free_memory);
};

/**
 * Initializes a NCCL clique and sets it into a raft resource instance
 *
 * @param[in] res the raft resources object
 * @param[in] device_ids selection of GPUs initialize the clique on
 * @param[in] percent_of_free_memory percentage of device memory to pre-allocate as a memory pool on
 * each GPU
 * @return NCCL clique
 */
inline const raft::core::nccl_clique& initialize_nccl_clique(
  resources const& res, std::optional<std::vector<int>> device_ids, int percent_of_free_memory = 80)
{
  return build_nccl_clique(res, device_ids, percent_of_free_memory);
};

/**
 * Retrieves a NCCL clique from raft resource instance, initializes one with default parameters if
 * absent
 *
 * @param[in] res the raft resources object
 * @return NCCL clique
 */
inline const raft::core::nccl_clique& get_nccl_clique(resources const& res)
{
  if (!res.has_resource_factory(resource_type::NCCL_CLIQUE)) {
    raft::resource::initialize_nccl_clique(res);
  }
  return *res.get_resource<raft::core::nccl_clique>(resource_type::NCCL_CLIQUE);
};

/**
 * @}
 */

}  // namespace raft::resource
