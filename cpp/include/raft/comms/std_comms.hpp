/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/detail/std_comms.hpp>

#include <iostream>
#include <nccl.h>
#include <ucp/api/ucp.h>

namespace raft {
namespace comms {

using std_comms = detail::std_comms;

/**
 * @defgroup std_comms_factory std_comms Factory functions
 * @{
 */

/**
 * Factory function to construct a RAFT NCCL communicator and inject it into a
 * RAFT handle.
 *
 * @param handle raft::resources for injecting the comms
 * @param nccl_comm initialized NCCL communicator to use for collectives
 * @param num_ranks number of ranks in communicator clique
 * @param rank rank of local instance
 *
 * @code{.cpp}
 * #include <raft/comms/std_comms.hpp>
 * #include <raft/core/device_mdarray.hpp>
 *
 * ncclComm_t nccl_comm;
 * raft::raft::resources handle;
 *
 * build_comms_nccl_only(&handle, nccl_comm, 5, 0);
 * ...
 * const auto& comm = resource::get_comms(handle);
 * auto gather_data = raft::make_device_vector<float>(handle, comm.get_size());
 * ...
 * comm.allgather((gather_data.data_handle())[comm.get_rank()],
 *                gather_data.data_handle(),
 *                1,
 *                resource::get_cuda_stream(handle));
 *
 * comm.sync_stream(resource::get_cuda_stream(handle));
 * @endcode
 */
void build_comms_nccl_only(resources* handle, ncclComm_t nccl_comm, int num_ranks, int rank)
{
  cudaStream_t stream = resource::get_cuda_stream(*handle);

  auto communicator = std::make_shared<comms_t>(
    std::unique_ptr<comms_iface>(new raft::comms::std_comms(nccl_comm, num_ranks, rank, stream)));
  resource::set_comms(*handle, communicator);
}

/**
 * Factory function to construct a RAFT NCCL+UCX and inject it into a RAFT
 * handle.
 *
 * @param handle raft::resources for injecting the comms
 * @param nccl_comm initialized NCCL communicator to use for collectives
 * @param ucp_worker of local process
 *        Note: This is purposefully left as void* so that the ucp_worker_h
 *        doesn't need to be exposed through the cython layer
 * @param eps array of ucp_ep_h instances.
 *        Note: This is purposefully left as void* so that
 *        the ucp_ep_h doesn't need to be exposed through the cython layer.
 * @param num_ranks number of ranks in communicator clique
 * @param rank rank of local instance
 *
 * @code{.cpp}
 * #include <raft/comms/std_comms.hpp>
 * #include <raft/core/device_mdarray.hpp>
 *
 * ncclComm_t nccl_comm;
 * raft::raft::resources handle;
 * ucp_worker_h ucp_worker;
 * ucp_ep_h *ucp_endpoints_arr;
 *
 * build_comms_nccl_ucx(&handle, nccl_comm, &ucp_worker, ucp_endpoints_arr, 5, 0);
 * ...
 * const auto& comm = resource::get_comms(handle);
 * auto gather_data = raft::make_device_vector<float>(handle, comm.get_size());
 * ...
 * comm.allgather((gather_data.data_handle())[comm.get_rank()],
 *                gather_data.data_handle(),
 *                1,
 *                resource::get_cuda_stream(handle));
 *
 * comm.sync_stream(resource::get_cuda_stream(handle));
 * @endcode
 */
void build_comms_nccl_ucx(
  resources* handle, ncclComm_t nccl_comm, void* ucp_worker, void* eps, int num_ranks, int rank)
{
  auto eps_sp = std::make_shared<ucp_ep_h*>(new ucp_ep_h[num_ranks]);

  auto size_t_ep_arr = reinterpret_cast<size_t*>(eps);

  for (int i = 0; i < num_ranks; i++) {
    size_t ptr    = size_t_ep_arr[i];
    auto ucp_ep_v = reinterpret_cast<ucp_ep_h*>(*eps_sp);

    if (ptr != 0) {
      auto eps_ptr = reinterpret_cast<ucp_ep_h>(size_t_ep_arr[i]);
      ucp_ep_v[i]  = eps_ptr;
    } else {
      ucp_ep_v[i] = nullptr;
    }
  }

  cudaStream_t stream = resource::get_cuda_stream(*handle);

  auto communicator =
    std::make_shared<comms_t>(std::unique_ptr<comms_iface>(new raft::comms::std_comms(
      nccl_comm, (ucp_worker_h)ucp_worker, eps_sp, num_ranks, rank, stream)));
  resource::set_comms(*handle, communicator);
}

/**
 * @}
 */

inline void nccl_unique_id_from_char(ncclUniqueId* id, char* uniqueId)
{
  memcpy(id->internal, uniqueId, NCCL_UNIQUE_ID_BYTES);
}

inline void get_nccl_unique_id(char* uid)
{
  ncclUniqueId id;
  ncclGetUniqueId(&id);

  memcpy(uid, id.internal, NCCL_UNIQUE_ID_BYTES);
}
};  // namespace comms
};  // end namespace raft
