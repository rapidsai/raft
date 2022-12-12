/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <raft/comms/comms.hpp>
#include <raft/comms/detail/mpi_comms.hpp>

namespace raft {
namespace comms {

using mpi_comms = detail::mpi_comms;

/**
 * @defgroup mpi_comms_factory MPI Comms Factory Functions
 * @{
 */

/**
 * Given a properly initialized MPI_Comm, construct an instance of RAFT's
 * MPI Communicator and inject it into the given RAFT handle instance
 * @param handle raft handle for managing expensive resources
 * @param comm an initialized MPI communicator
 */
inline void initialize_mpi_comms(handle_t* handle, MPI_Comm comm)
{
  auto communicator = std::make_shared<comms_t>(
    std::unique_ptr<comms_iface>(new mpi_comms(comm, false, handle->get_stream())));
  handle->set_comms(communicator);
};

/**
 * @}
 */

};  // namespace comms
};  // end namespace raft
