/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <iostream>
#include <raft/comms/comms.hpp>
#include <raft/handle.hpp>
#include <raft/mr/device/buffer.hpp>

namespace raft {
namespace comms {

/**
 * Function to construct comms_t and inject it on a handle_t. This
 * is used for convenience in the Python layer.
 *
 * @param handle raft::handle_t for injecting the comms
 * @param nccl_comm initialized NCCL communicator to use for collectives
 * @param num_ranks number of ranks in communicator clique
 * @param rank rank of local instance
 */
void build_comms_nccl_only(handle_t *handle, ncclComm_t nccl_comm,
                           int num_ranks, int rank) {
  auto d_alloc = handle->get_device_allocator();
  raft::comms::comms_iface *raft_comm =
    new raft::comms::std_comms(nccl_comm, num_ranks, rank, d_alloc);
  std::cout << "Comms: " << raft_comm->getSize() << std::endl;

  auto communicator =
    std::make_shared<comms_t>(std::unique_ptr<comms_iface>(raft_comm));
  handle->set_comms(communicator);
}

/**
 * Function to construct comms_t and inject it on a handle_t. This
 * is used for convenience in the Python layer.
 *
 * @param handle raft::handle_t for injecting the comms
 * @param nccl_comm initialized NCCL communicator to use for collectives
 * @param ucp_worker of local process
 *        Note: This is purposefully left as void* so that the ucp_worker_h
 *        doesn't need to be exposed through the cython layer
 * @param eps array of ucp_ep_h instances.
 *        Note: This is purposefully left as void* so that
 *        the ucp_ep_h doesn't need to be exposed through the cython layer.
 * @param num_ranks number of ranks in communicator clique
 * @param rank rank of local instance
 */
void build_comms_nccl_ucx(handle_t *handle, ncclComm_t nccl_comm,
                          void *ucp_worker, void *eps, int num_ranks,
                          int rank) {
  auto eps_sp = std::make_shared<ucp_ep_h *>(new ucp_ep_h[num_ranks]);

  auto size_t_ep_arr = reinterpret_cast<size_t *>(eps);

  for (int i = 0; i < num_ranks; i++) {
    size_t ptr = size_t_ep_arr[i];
    auto ucp_ep_v = reinterpret_cast<ucp_ep_h *>(*eps_sp);

    if (ptr != 0) {
      auto eps_ptr = reinterpret_cast<ucp_ep_h>(size_t_ep_arr[i]);
      ucp_ep_v[i] = eps_ptr;
    } else {
      ucp_ep_v[i] = nullptr;
    }
  }

  auto d_alloc = handle->get_device_allocator();
  auto *raft_comm = new raft::comms::std_comms(
    nccl_comm, (ucp_worker_h)ucp_worker, eps_sp, num_ranks, rank, d_alloc);
  auto communicator =
    std::make_shared<comms_t>(std::unique_ptr<comms_iface>(raft_comm));
  handle->set_comms(communicator);
}

/**
 * A simple sanity check that NCCL is able to perform a collective operation
 *
 * @param the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 */
bool test_collective_allreduce(const handle_t &handle, int root) {
  const comms_t &communicator = handle.get_comms();

  const int send = 1;

  cudaStream_t stream = handle.get_stream();

  raft::mr::device::buffer<int> temp_d(handle.get_device_allocator(), stream);
  temp_d.resize(1, stream);
  CUDA_CHECK(
    cudaMemcpyAsync(temp_d.data(), &send, 1, cudaMemcpyHostToDevice, stream));
  communicator.allreduce(temp_d.data(), temp_d.data(), 1, op_t::SUM, stream);
  int temp_h = 0;
  CUDA_CHECK(
    cudaMemcpyAsync(&temp_h, temp_d.data(), 1, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  communicator.barrier();

  std::cout << "Clique size: " << communicator.getSize() << std::endl;
  std::cout << "final_size: " << temp_h << std::endl;

  return temp_h == communicator.getSize();
}

/**
 * A simple sanity check that NCCL is able to perform a collective operation
 *
 * @param the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 */
bool test_collective_broadcast(const handle_t &handle, int root) {
  const comms_t &communicator = handle.get_comms();

  const int send = root;

  cudaStream_t stream = handle.get_stream();

  raft::mr::device::buffer<int> temp_d(handle.get_device_allocator(), stream);
  temp_d.resize(1, stream);

  if (communicator.getRank() == root)
    CUDA_CHECK(cudaMemcpyAsync(temp_d.data(), &send, sizeof(int),
                               cudaMemcpyHostToDevice, stream));

  communicator.bcast(temp_d.data(), 1, root, stream);
  communicator.syncStream(stream);
  int temp_h = -1;  // Verify more than one byte is being sent
  CUDA_CHECK(cudaMemcpyAsync(&temp_h, temp_d.data(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  communicator.barrier();

  std::cout << "Clique size: " << communicator.getSize() << std::endl;
  std::cout << "final_size: " << temp_h << std::endl;

  return temp_h == root;
}

bool test_collective_reduce(const handle_t &handle, int root) {
  const comms_t &communicator = handle.get_comms();

  const int send = root;

  cudaStream_t stream = handle.get_stream();

  raft::mr::device::buffer<int> temp_d(handle.get_device_allocator(), stream);
  temp_d.resize(1, stream);

  CUDA_CHECK(cudaMemcpyAsync(temp_d.data(), &send, sizeof(int),
                             cudaMemcpyHostToDevice, stream));

  communicator.reduce(temp_d.data(), temp_d.data(), 1, op_t::SUM, root, stream);
  communicator.syncStream(stream);
  int temp_h = -1;  // Verify more than one byte is being sent
  CUDA_CHECK(cudaMemcpyAsync(&temp_h, temp_d.data(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  communicator.barrier();

  std::cout << "Clique size: " << communicator.getSize() << std::endl;
  std::cout << "final_size: " << temp_h << std::endl;

  if (communicator.getRank() == root)
    return temp_h == root * communicator.getSize();
  else
    return true;
}

bool test_collective_allgather(const handle_t &handle, int root) {
  const comms_t &communicator = handle.get_comms();

  const int send = root;

  cudaStream_t stream = handle.get_stream();

  raft::mr::device::buffer<int> temp_d(handle.get_device_allocator(), stream);
  temp_d.resize(1, stream);

  raft::mr::device::buffer<int> recv_d(handle.get_device_allocator(), stream,
                                       communicator.getSize());

  CUDA_CHECK(cudaMemcpyAsync(temp_d.data(), &send, sizeof(int),
                             cudaMemcpyHostToDevice, stream));

  communicator.allgather(temp_d.data(), recv_d.data(), 1, stream);
  communicator.syncStream(stream);
  int
    temp_h[communicator.getSize()];  // Verify more than one byte is being sent
  CUDA_CHECK(cudaMemcpyAsync(&temp_h, temp_d.data(),
                             sizeof(int) * communicator.getSize(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  communicator.barrier();

  std::cout << "Clique size: " << communicator.getSize() << std::endl;
  std::cout << "final_size: " << temp_h << std::endl;

  for (int i = 0; i < communicator.getSize(); i++)
    if (temp_h[i] != i) return false;
  return true;
}

bool test_collective_reducescatter(const handle_t &handle, int root) {
  const comms_t &communicator = handle.get_comms();

  const int send = 1;

  cudaStream_t stream = handle.get_stream();

  raft::mr::device::buffer<int> temp_d(handle.get_device_allocator(), stream,
                                       1);
  raft::mr::device::buffer<int> recv_d(handle.get_device_allocator(), stream,
                                       1);

  CUDA_CHECK(cudaMemcpyAsync(temp_d.data(), &send, sizeof(int),
                             cudaMemcpyHostToDevice, stream));

  communicator.reducescatter(temp_d.data(), recv_d.data(), 1, op_t::SUM,
                             stream);
  communicator.syncStream(stream);
  int temp_h = -1;  // Verify more than one byte is being sent
  CUDA_CHECK(cudaMemcpyAsync(&temp_h, temp_d.data(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  communicator.barrier();

  std::cout << "Clique size: " << communicator.getSize() << std::endl;
  std::cout << "final_size: " << temp_h << std::endl;

  return temp_h = communicator.getSize();
}

/**
 * A simple sanity check that UCX is able to send messages between all ranks
 *
 * @param the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 * @param number of iterations of all-to-all messaging to perform
 */
bool test_pointToPoint_simple_send_recv(const handle_t &h, int numTrials) {
  const comms_t &communicator = h.get_comms();
  const int rank = communicator.getRank();

  bool ret = true;
  for (int i = 0; i < numTrials; i++) {
    std::vector<int> received_data((communicator.getSize() - 1), -1);

    std::vector<request_t> requests;
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
