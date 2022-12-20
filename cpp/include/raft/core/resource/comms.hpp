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
class comms_resource_t : public resource_t {
 public:
  comms_resource_t(std::shared_ptr<comms::comms_t> comnumicator) : communicator_(comnumicator) {}

  void* get_resource() override { return &communicator_; }

  ~comms_resource_t() override {}

 private:
  std::shared_ptr<comms::comms_t> communicator_;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource_t to populate
 * the handle_t.
 */
class comms_resource_factory_t : public resource_factory_t {
 public:
  comms_resource_factory_t(std::shared_ptr<comms::comms_t> communicator)
    : communicator_(communicator)
  {
  }

  resource_type_t resource_type() override { return resource_type_t::COMMUNICATOR; }

  resource_t* make_resource() override { return new comms_resource_t(communicator_); }

 private:
  std::shared_ptr<comms::comms_t> communicator_;
};

inline bool comms_initialized(base_handle_t const& handle)
{
  return handle.has_resource_factory(resource_type_t::COMMUNICATOR);
}

inline const comms::comms_t& get_comms(const base_handle_t& handle)
{
  RAFT_EXPECTS(comms_initialized(handle), "ERROR: Communicator was not initialized\n");
  return *handle.get_resource<comms::comms_t>(resource_type_t::COMMUNICATOR);
}

inline void set_comms(base_handle_t& handle, std::shared_ptr<comms::comms_t> communicator)
{
  handle.add_resource_factory(std::make_shared<comms_resource_factory_t>(communicator));
}
}  // namespace raft::core
