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

class main_gpu_resource : public device_id_resource {
 public:
  main_gpu_resource() : device_id_resource() {}
  ~main_gpu_resource() override {}
};

class main_gpu_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::MAIN_GPU_ID; }
  resource* make_resource() override { return new main_gpu_resource(); }
};

inline int& get_main_gpu_id(resources const& res)
{
  if (!res.has_resource_factory(resource_type::MAIN_GPU_ID)) {
    res.add_resource_factory(std::make_shared<main_gpu_resource_factory>());
  }
  return *res.get_resource<int>(resource_type::MAIN_GPU_ID);
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
inline int get_world_size(resources const& res)
{
  return raft::resource::get_multi_gpu_resource(res).size();
}

/**
 * @brief Get specific GPU's raft::resources object
 */
inline const raft::resources& get_device_resources_for_gpu_id(resources const& res, int gpu_id)
{
  std::vector<raft::resources>& world_resources = raft::resource::get_multi_gpu_resource(res);
  return world_resources[gpu_id];
}

/**
 * @brief Switch device to given gpu_id and return its raft::resources object
 */
inline const raft::resources& set_current_device_to_gpu_id(resources const& res, int gpu_id)
{
  const raft::resources& dev_res = raft::resource::get_device_resources_for_gpu_id(res, gpu_id);
  RAFT_CUDA_TRY(cudaSetDevice(raft::resource::get_device_id(dev_res)));
  return dev_res;
}

/**
 * @brief Switch to main GPU and return its raft::resources object
 */
inline const raft::resources& set_current_device_to_main_gpu(resources const& res)
{
  int main_gpu_id = get_main_gpu_id(res);
  const raft::resources& dev_res =
    raft::resource::get_device_resources_for_gpu_id(res, main_gpu_id);
  RAFT_CUDA_TRY(cudaSetDevice(raft::resource::get_device_id(dev_res)));
  return dev_res;
}

/**
 * @brief Set the main gpu id to given main_gpu_id
 */
inline void set_main_gpu_id(resources const& res, int main_gpu_id)
{
  int& main_gpu_id_ = get_main_gpu_id(res);
  main_gpu_id_      = main_gpu_id;
};

}  // namespace raft::resource
