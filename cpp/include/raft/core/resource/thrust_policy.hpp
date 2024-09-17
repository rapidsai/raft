/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <rmm/exec_policy.hpp>
namespace raft::resource {
class thrust_policy_resource : public resource {
 public:
  thrust_policy_resource(rmm::cuda_stream_view stream_view)
    : thrust_policy_(std::make_unique<rmm::exec_policy_nosync>(stream_view))
  {
  }
  void* get_resource() override { return thrust_policy_.get(); }

  ~thrust_policy_resource() override {}

 private:
  std::unique_ptr<rmm::exec_policy_nosync> thrust_policy_;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class thrust_policy_resource_factory : public resource_factory {
 public:
  thrust_policy_resource_factory(rmm::cuda_stream_view stream_view) : stream_view_(stream_view) {}
  resource_type get_resource_type() override { return resource_type::THRUST_POLICY; }
  resource* make_resource() override { return new thrust_policy_resource(stream_view_); }

 private:
  rmm::cuda_stream_view stream_view_;
};

/**
 * @defgroup resource_thrust_policy Thrust policy resource functions
 * @{
 */

/**
 * Load a thrust policy from a res (and populate it on the res if needed).
 * @param res raft res object for managing resources
 * @return thrust execution policy
 */
inline rmm::exec_policy_nosync& get_thrust_policy(resources const& res)
{
  if (!res.has_resource_factory(resource_type::THRUST_POLICY)) {
    rmm::cuda_stream_view stream = get_cuda_stream(res);
    res.add_resource_factory(std::make_shared<thrust_policy_resource_factory>(stream));
  }
  return *res.get_resource<rmm::exec_policy_nosync>(resource_type::THRUST_POLICY);
};

/**
 * @}
 */

}  // namespace raft::resource
