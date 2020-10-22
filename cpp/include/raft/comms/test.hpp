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
 * A simple sanity check that NCCL is able to perform a collective operation
 *
 * @param the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 */
bool test_collective_allreduce(const handle_t &handle, int root) {
  comms_t const &communicator = handle.get_comms();

  int const send = 1;

  cudaStream_t stream = handle.get_stream();

  raft::mr::device::buffer<int> temp_d(handle.get_device_allocator(), stream);
  temp_d.resize(1, stream);
  CUDA_CHECK(
    cudaMemcpyAsync(temp_d.data(), &send, 1, cudaMemcpyHostToDevice, stream));

  communicator.allreduce(temp_d.data(), temp_d.data(), 1, OpT::kSum, stream);

  int temp_h = 0;
  CUDA_CHECK(
    cudaMemcpyAsync(&temp_h, temp_d.data(), 1, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  communicator.barrier();

  std::cout << "Clique size: " << communicator.get_size() << std::endl;
  std::cout << "final_size: " << temp_h << std::endl;

  return temp_h == communicator.get_size();
}

/**
 * A simple sanity check that NCCL is able to perform a collective operation
 *
 * @param the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 */
bool test_collective_broadcast(const handle_t &handle, int root) {
  comms_t const &communicator = handle.get_comms();

  int const send = root;

  cudaStream_t stream = handle.get_stream();

  raft::mr::device::buffer<int> temp_d(handle.get_device_allocator(), stream);
  temp_d.resize(1, stream);

  if (communicator.get_rank() == root)
    CUDA_CHECK(cudaMemcpyAsync(temp_d.data(), &send, sizeof(int),
                               cudaMemcpyHostToDevice, stream));

  communicator.bcast(temp_d.data(), 1, root, stream);
  communicator.sync_stream(stream);
  int temp_h = -1;  // Verify more than one byte is being sent
  CUDA_CHECK(cudaMemcpyAsync(&temp_h, temp_d.data(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  communicator.barrier();

  std::cout << "Clique size: " << communicator.get_size() << std::endl;
  std::cout << "final_size: " << temp_h << std::endl;

  return temp_h == root;
}

bool test_collective_reduce(const handle_t &handle, int root) {
  comms_t const &communicator = handle.get_comms();

  int const send = root;

  cudaStream_t stream = handle.get_stream();

  raft::mr::device::buffer<int> temp_d(handle.get_device_allocator(), stream);
  temp_d.resize(1, stream);

  CUDA_CHECK(cudaMemcpyAsync(temp_d.data(), &send, sizeof(int),
                             cudaMemcpyHostToDevice, stream));

  communicator.reduce(temp_d.data(), temp_d.data(), 1, OpT::kSum, root, stream);
  communicator.sync_stream(stream);
  int temp_h = -1;  // Verify more than one byte is being sent
  CUDA_CHECK(cudaMemcpyAsync(&temp_h, temp_d.data(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  communicator.barrier();

  std::cout << "Clique size: " << communicator.get_size() << std::endl;
  std::cout << "final_size: " << temp_h << std::endl;

  if (communicator.get_rank() == root)
    return temp_h == root * communicator.get_size();
  else
    return true;
}

bool test_collective_allgather(const handle_t &handle, int root) {
  comms_t const &communicator = handle.get_comms();

  int const send = communicator.get_rank();

  cudaStream_t stream = handle.get_stream();

  raft::mr::device::buffer<int> temp_d(handle.get_device_allocator(), stream);
  temp_d.resize(1, stream);

  raft::mr::device::buffer<int> recv_d(handle.get_device_allocator(), stream,
                                       communicator.get_size());

  CUDA_CHECK(cudaMemcpyAsync(temp_d.data(), &send, sizeof(int),
                             cudaMemcpyHostToDevice, stream));

  communicator.allgather(temp_d.data(), recv_d.data(), 1, stream);
  communicator.sync_stream(stream);
  int
    temp_h[communicator.get_size()];  // Verify more than one byte is being sent
  CUDA_CHECK(cudaMemcpyAsync(&temp_h, recv_d.data(),
                             sizeof(int) * communicator.get_size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  communicator.barrier();

  std::cout << "Clique size: " << communicator.get_size() << std::endl;
  std::cout << "final_size: " << temp_h << std::endl;

  for (int i = 0; i < communicator.get_size(); i++) {
    if (temp_h[i] != i) return false;
  }
  return true;
}

bool test_collective_reducescatter(const handle_t &handle, int root) {
  comms_t const &communicator = handle.get_comms();

  int const send = 1;

  cudaStream_t stream = handle.get_stream();

  raft::mr::device::buffer<int> temp_d(handle.get_device_allocator(), stream,
                                       1);
  raft::mr::device::buffer<int> recv_d(handle.get_device_allocator(), stream,
                                       1);

  CUDA_CHECK(cudaMemcpyAsync(temp_d.data(), &send, sizeof(int),
                             cudaMemcpyHostToDevice, stream));

  communicator.reducescatter(temp_d.data(), recv_d.data(), 1, OpT::kSum,
                             stream);
  communicator.sync_stream(stream);
  int temp_h = -1;  // Verify more than one byte is being sent
  CUDA_CHECK(cudaMemcpyAsync(&temp_h, temp_d.data(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  communicator.barrier();

  std::cout << "Clique size: " << communicator.get_size() << std::endl;
  std::cout << "final_size: " << temp_h << std::endl;

  return temp_h == communicator.get_size();
}

/**
 * A simple sanity check that UCX is able to send messages between all ranks
 *
 * @param the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 * @param number of iterations of all-to-all messaging to perform
 */
bool test_pointToPoint_simple_send_recv(const handle_t &h, int numTrials) {
  comms_t const &communicator = h.get_comms();
  int const rank = communicator.get_rank();

  bool ret = true;
  for (int i = 0; i < numTrials; i++) {
    std::vector<int> received_data((communicator.get_size() - 1), -1);

    std::vector<request_t> requests;
    requests.resize(2 * (communicator.get_size() - 1));
    int request_idx = 0;
    //post receives
    for (int r = 0; r < communicator.get_size(); ++r) {
      if (r != rank) {
        communicator.irecv(received_data.data() + request_idx, 1, r, 0,
                           requests.data() + request_idx);
        ++request_idx;
      }
    }

    for (int r = 0; r < communicator.get_size(); ++r) {
      if (r != rank) {
        communicator.isend(&rank, 1, r, 0, requests.data() + request_idx);
        ++request_idx;
      }
    }

    communicator.waitall(requests);
    communicator.barrier();

    if (communicator.get_rank() == 0) {
      std::cout << "=========================" << std::endl;
      std::cout << "Trial " << i << std::endl;
    }

    for (int printrank = 0; printrank < communicator.get_size(); ++printrank) {
      if (communicator.get_rank() == printrank) {
        std::cout << "Rank " << communicator.get_rank() << " received: [";
        for (size_t i = 0; i < received_data.size(); i++) {
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

    if (communicator.get_rank() == 0)
      std::cout << "=========================" << std::endl;
  }

  return ret;
}

/**
 * A simple test that the comms can be split into 2 separate subcommunicators
 *
 * @param the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 * @param n_colors number of different colors to test
 */
bool test_commsplit(const handle_t &h, int n_colors) {
  comms_t const &communicator = h.get_comms();
  int const rank = communicator.get_rank();
  int const size = communicator.get_size();

  if (n_colors > size) n_colors = size;

  // first we need to assign to a color, then assign the rank within the color
  int color = rank % n_colors;
  int key = rank / n_colors;

  handle_t new_handle(1);
  auto shared_comm =
    std::make_shared<comms_t>(communicator.comm_split(color, key));
  new_handle.set_comms(shared_comm);

  return test_collective_allreduce(new_handle, 0);
}

}  // namespace comms
};  // namespace raft
