/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <raft/handle.hpp>
#include <raft/comms/comms.hpp>
#include <raft/mr/device/buffer.hpp>
#include <iostream>

namespace raft {
namespace comms {

void build_comms_nccl_only(handle_t *handle, ncclComm_t comm, int size,
                          int rank) {

  auto *raft_comm = new raft::comms::std_comms(comm, size, rank);
  auto communicator = std::make_shared<comms_t>(
	std::unique_ptr<comms_t>(raft_comm));
  handle->set_comms(communicator);
}

void build_comms_nccl_ucx(handle_t *handle, ncclComm_t comm, void *ucp_worker,
                     void *eps, int size, int rank) {

  std::shared_ptr<ucp_ep_h *> eps_sp =
    std::make_shared<ucp_ep_h *>(new ucp_ep_h[size]);

  size_t *size_t_ep_arr = (size_t *)eps;

  for (int i = 0; i < size; i++) {
    size_t ptr = size_t_ep_arr[i];
    ucp_ep_h *ucp_ep_v = (ucp_ep_h *)*eps_sp;

    if (ptr != 0) {
      ucp_ep_h eps_ptr = (ucp_ep_h)size_t_ep_arr[i];
      ucp_ep_v[i] = eps_ptr;
    } else {
      ucp_ep_v[i] = nullptr;
    }
  }

  auto communicator = std::make_shared<comms_t>(
    std::unique_ptr<comms_t>(
      new raft::comms::std_comms(comm, (ucp_worker_h)ucp_worker, eps_sp, size, rank)));
  handle->set_comms(communicator);
}


bool test_collective_allreduce(const handle_t& handle) {
  const comms_t& communicator = handle.get_comms();

  const int send = 1;

  cudaStream_t stream = handle.get_stream();

  raft::mr::device::buffer<int> temp_d(handle.get_device_allocator(), stream);
  temp_d.resize(1, stream);
  CUDA_CHECK(cudaMemcpyAsync(temp_d.data(), &send, sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  communicator.allreduce(temp_d.data(), temp_d.data(), 1, getDataType(temp_d.data()),
                         comms_t::SUM, stream);
  int temp_h = 0;
  CUDA_CHECK(cudaMemcpyAsync(&temp_h, temp_d.data(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  communicator.barrier();

  std::cout << "Clique size: " << communicator.getSize() << std::endl;
  std::cout << "final_size: " << temp_h << std::endl;

  return temp_h == communicator.getSize();
}

bool test_pointToPoint_simple_send_recv(const handle_t& h,
                                        int numTrials) {
  const comms_t& communicator = h.get_comms();
  const int rank = communicator.getRank();

  bool ret = true;
  for (int i = 0; i < numTrials; i++) {
    std::vector<int> received_data((communicator.getSize() - 1), -1);

    std::vector<comms_t::request_t> requests;
    requests.resize(2 * (communicator.getSize() - 1));
    int request_idx = 0;
    //post receives
    for (int r = 0; r < communicator.getSize(); ++r) {
      if (r != rank) {
        communicator.irecv(received_data.data() + request_idx, 1, r, 0,
                           requests.data() + request_idx);
        ++request_idx;
      }
    }

    for (int r = 0; r < communicator.getSize(); ++r) {
      if (r != rank) {
        communicator.isend(&rank, 1, r, 0, requests.data() + request_idx);
        ++request_idx;
      }
    }

    communicator.waitall(requests.size(), requests.data());
    communicator.barrier();

    if (communicator.getRank() == 0) {
      std::cout << "=========================" << std::endl;
      std::cout << "Trial " << i << std::endl;
    }

    for (int printrank = 0; printrank < communicator.getSize(); ++printrank) {
      if (communicator.getRank() == printrank) {
        std::cout << "Rank " << communicator.getRank() << " received: [";
        for (int i = 0; i < received_data.size(); i++) {
          auto rec = received_data[i];
          std::cout << rec;
          if (rec == -1) ret = false;
          communicator.barrier();
          if (i < received_data.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
      }

      communicator.barrier();
    }

    if (communicator.getRank() == 0)
      std::cout << "=========================" << std::endl;
  }

  return ret;
}

};  // namespace comms
};  // end namespace raft
