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

#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <memory>

namespace raft::resource {

class multi_gpu_resource : public resource {
 public:
  multi_gpu_resource() : world_resources_(std::make_unique<std::vector<raft::resources>>(0)) {}
  ~multi_gpu_resource() override {}
  void* get_resource() override { return world_resources_.get(); }

 private:
  std::unique_ptr<std::vector<raft::resources>> world_resources_;
};

/** Factory that knows how to construct a specific raft::resource to populate the res_t. */
class multi_gpu_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::MULTI_GPU; }
  resource* make_resource() override { return new multi_gpu_resource(); }
};

class root_rank_resource : public resource {
 public:
  root_rank_resource() : root_rank_(0) {}
  void* get_resource() override { return &root_rank_; }

  ~root_rank_resource() override {}

 private:
  int root_rank_;
};

class root_rank_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::ROOT_RANK; }
  resource* make_resource() override { return new root_rank_resource(); }
};

inline int& get_root_rank(resources const& res)
{
  if (!res.has_resource_factory(resource_type::ROOT_RANK)) {
    res.add_resource_factory(std::make_shared<root_rank_resource_factory>());
  }
  return *res.get_resource<int>(resource_type::ROOT_RANK);
};

/**
 * Retrieves a multi gpu resource from raft res if it exists, otherwise initializes it and returns
 * it.
 *
 * @param[in] res the raft resources object
 * @return resource for each device in multi-gpu world
 */
inline std::vector<raft::resources>& get_multi_gpu_resource(resources const& res)
{
  if (!res.has_resource_factory(resource_type::MULTI_GPU)) {
    res.add_resource_factory(std::make_shared<multi_gpu_resource_factory>());
  }
  return *res.get_resource<std::vector<raft::resources>>(resource_type::MULTI_GPU);
};

/**
 * @brief Returns true if res has a multi GPU resource type
 */
inline bool is_multi_gpu(resources const& res)
{
  return res.has_resource_factory(resource_type::MULTI_GPU);
};

/**
 * @brief Get number of gpus in multi-gpu world
 */
inline int get_num_ranks(resources const& res)
{
  return raft::resource::get_multi_gpu_resource(res).size();
}

/**
 * @brief Get specific rank's raft::resources object
 */
inline const raft::resources& get_device_resources_for_rank(resources const& res, int rank)
{
  std::vector<raft::resources>& world_resources = raft::resource::get_multi_gpu_resource(res);
  return world_resources[rank];
}

/**
 * @brief Switch device to rank and return its raft::resources object
 */
inline const raft::resources& set_current_device_to_rank(resources const& res, int rank)
{
  const raft::resources& dev_res = raft::resource::get_device_resources_for_rank(res, rank);
  RAFT_CUDA_TRY(cudaSetDevice(raft::resource::get_device_id(dev_res)));
  return dev_res;
}

/**
 * @brief Switch to root rank and return its raft::resources object
 */
inline const raft::resources& set_current_device_to_root_rank(resources const& res)
{
  int root_rank                  = get_root_rank(res);
  const raft::resources& dev_res = raft::resource::get_device_resources_for_rank(res, root_rank);
  RAFT_CUDA_TRY(cudaSetDevice(raft::resource::get_device_id(dev_res)));
  return dev_res;
}

/**
 * @brief Set the root rank to given rank
 */
inline void set_root_rank(resources const& res, int root_rank)
{
  int& root_rank_ = get_root_rank(res);
  root_rank_      = root_rank;
};

}  // namespace raft::resource
