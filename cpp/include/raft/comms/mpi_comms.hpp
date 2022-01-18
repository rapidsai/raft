/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
>>>>>>> branch-22.02

// FIXME: Remove after consumer rename
#ifndef MPI_TRY_NO_THROW
#define MPI_TRY_NO_THROW(call) RAFT_MPI_TRY_NO_THROW(call)
#endif

namespace raft {
namespace comms {

using mpi_comms = detail::mpi_comms;

inline void initialize_mpi_comms(handle_t* handle, MPI_Comm comm)
{
  auto communicator =
    std::make_shared<comms_t>(std::unique_ptr<comms_iface>(new mpi_comms(comm, false)));
  handle->set_comms(communicator);
};

};  // namespace comms
};  // end namespace raft
