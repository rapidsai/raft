/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/core/comms.hpp>
#include <raft/core/resource/resource_types.hpp>

namespace raft::core {
class sub_comms_resource_t : public resource_t {
 public:
  sub_comms_resource_t() {}
  void* get_resource() override { return &communicators_; }

  ~sub_comms_resource_t() override {}

 private:
  std::unordered_map<std::string, std::shared_ptr<comms::comms_t>> communicators_;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource_t to populate
 * the handle_t.
 */
class sub_comms_resource_factory_t : public resource_factory_t {
 public:
  resource_type_t resource_type() override { return resource_type_t::SUB_COMMUNICATOR; }
  resource_t* make_resource() override { return new sub_comms_resource_t(); }
};

const comms::comms_t& get_subcomm(const base_handle_t& handle, std::string key)
{
  if (!handle.has_resource_factory(resource_type_t::SUB_COMMUNICATOR)) {
    handle.add_resource_factory(std::make_shared<sub_comms_resource_factory_t>());
  }

  auto sub_comms =
    *handle.get_resource<std::unordered_map<std::string, std::shared_ptr<comms::comms_t>>>(
      resource_type_t::SUB_COMMUNICATOR);
  auto sub_comm = sub_comms.at(key);

  RAFT_EXPECTS(nullptr != sub_comm.get(), "ERROR: Subcommunicator was not initialized");

  return *sub_comm;
}

inline void set_subcomm(base_handle_t& handle,
                        std::string key,
                        std::shared_ptr<comms::comms_t> subcomm)
{
  if (!handle.has_resource_factory(resource_type_t::SUB_COMMUNICATOR)) {
    handle.add_resource_factory(std::make_shared<sub_comms_resource_factory_t>());
  }

  auto sub_comms =
    *handle.get_resource<std::unordered_map<std::string, std::shared_ptr<comms::comms_t>>>(
      resource_type_t::SUB_COMMUNICATOR);
  sub_comms[key] = subcomm;
}
}  // namespace raft::core