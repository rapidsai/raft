/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <raft/core/resource/nccl_comm.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <nccl.h>

#include <memory>

namespace raft::resource {

class nccl_clique_resource : public resource {
 public:
  nccl_clique_resource() : clique_(std::make_unique<std::vector<raft::resources>>(0)) {}
  ~nccl_clique_resource() override {}
  void* get_resource() override { return clique_.get(); }

 private:
  std::unique_ptr<std::vector<raft::resources>> clique_;
};

/** Factory that knows how to construct a specific raft::resource to populate the res_t. */
class nccl_clique_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::NCCL_CLIQUE; }
  resource* make_resource() override { return new nccl_clique_resource(); }
};

class clique_root_rank_resource : public resource {
 public:
  clique_root_rank_resource() : clique_root_rank_(0) {}
  void* get_resource() override { return &clique_root_rank_; }

  ~clique_root_rank_resource() override {}

 private:
  int clique_root_rank_;
};

class clique_root_rank_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::CLIQUE_ROOT_RANK; }
  resource* make_resource() override { return new clique_root_rank_resource(); }
};

inline int& _get_clique_root_rank(resources const& res)
{
  if (!res.has_resource_factory(resource_type::CLIQUE_ROOT_RANK)) {
    res.add_resource_factory(std::make_shared<clique_root_rank_resource_factory>());
  }
  return *res.get_resource<int>(resource_type::CLIQUE_ROOT_RANK);
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
inline std::vector<raft::resources>& get_nccl_clique(resources const& res)
{
  if (!res.has_resource_factory(resource_type::NCCL_CLIQUE)) {
    res.add_resource_factory(std::make_shared<nccl_clique_resource_factory>());
  }
  return *res.get_resource<std::vector<raft::resources>>(resource_type::NCCL_CLIQUE);
};

/**
 * @brief Get number of ranks in clique
 */
inline int get_num_ranks(resources const& res)
{
  return raft::resource::get_nccl_clique(res).size();
}

/**
 * @brief Get rank's raft::resources object
 */
inline const raft::resources& get_device_resources(resources const& res, int rank)
{
  std::vector<raft::resources>& clique_device_resources = raft::resource::get_nccl_clique(res);
  return clique_device_resources[rank];
}

/**
 * @brief Set current device ID to rank and return its raft::resources object
 */
inline const raft::resources& set_current_device_to_rank(resources const& res, int rank)
{
  const raft::resources& dev_res = raft::resource::get_device_resources(res, rank);
  RAFT_CUDA_TRY(cudaSetDevice(raft::resource::get_device_id(dev_res)));
  return dev_res;
}

/**
 * @brief Set current device ID to root rank and return its raft::resources object
 */
inline const raft::resources& set_current_device_to_root_rank(resources const& res)
{
  int root_rank                  = _get_clique_root_rank(res);
  const raft::resources& dev_res = raft::resource::get_device_resources(res, root_rank);
  RAFT_CUDA_TRY(cudaSetDevice(raft::resource::get_device_id(dev_res)));
  return dev_res;
}

/**
 * @brief Get rank's NCCL comm
 */
inline ncclComm_t& get_nccl_comm(resources const& res, int rank)
{
  const raft::resources& dev_res = raft::resource::get_device_resources(res, rank);
  return raft::resource::get_nccl_comm(dev_res);
}

/**
 * @brief Set clique root rank
 */
inline void set_clique_root_rank(resources const& res, int clique_root_rank)
{
  int& clique_root_rank_ = _get_clique_root_rank(res);
  clique_root_rank_      = clique_root_rank;
};

/**
 * @brief Get clique root rank
 */
inline int get_clique_root_rank(resources const& res) { return _get_clique_root_rank(res); };

/**
 * @}
 */

}  // namespace raft::resource
