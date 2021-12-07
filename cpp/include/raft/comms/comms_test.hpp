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

#include <raft/comms/comms.hpp>
#include <raft/comms/detail/test.hpp>

#include <raft/handle.hpp>

namespace raft {
namespace comms {

/**
 * @brief A simple sanity check that NCCL is able to perform a collective operation
 *
 * @param[in] handle the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 *  @param[in] root the root rank id
 */
bool test_collective_allreduce(const handle_t& handle, int root)
{
  return detail::test_collective_allreduce(handle, root);
}

/**
 * @brief A simple sanity check that NCCL is able to perform a collective operation
 *
 * @param[in] handle the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 *  @param[in] root the root rank id
 */
bool test_collective_broadcast(const handle_t& handle, int root)
{
  return detail::test_collective_broadcast(handle, root);
}

/**
 * @brief A simple sanity check that NCCL is able to perform a collective reduce
 *
 * @param[in] handle the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 *  @param[in] root the root rank id
 */
bool test_collective_reduce(const handle_t& handle, int root)
{
  return detail::test_collective_reduce(handle, root);
}

/**
 * @brief A simple sanity check that NCCL is able to perform a collective allgather
 *
 * @param[in] handle the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 *  @param[in] root the root rank id
 */
bool test_collective_allgather(const handle_t& handle, int root)
{
  return detail::test_collective_allgather(handle, root);
}

/**
 * @brief A simple sanity check that NCCL is able to perform a collective gather
 *
 * @param[in] handle the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 *  @param[in] root the root rank id
 */
bool test_collective_gather(const handle_t& handle, int root)
{
  return detail::test_collective_gather(handle, root);
}

/**
 * @brief A simple sanity check that NCCL is able to perform a collective gatherv
 *
 * @param[in] handle the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 *  @param[in] root the root rank id
 */
bool test_collective_gatherv(const handle_t& handle, int root)
{
  return detail::test_collective_gatherv(handle, root);
}

/**
 * @brief A simple sanity check that NCCL is able to perform a collective reducescatter
 *
 * @param[in] handle the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 *  @param[in] root the root rank id
 */
bool test_collective_reducescatter(const handle_t& handle, int root)
{
  return detail::test_collective_reducescatter(handle, root);
}

/**
 * A simple sanity check that UCX is able to send messages between all ranks
 *
 * @param[in] h the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 * @param[in] numTrials number of iterations of all-to-all messaging to perform
 */
bool test_pointToPoint_simple_send_recv(const handle_t& h, int numTrials)
{
  return detail::test_pointToPoint_simple_send_recv(h, numTrials);
}

/**
 * A simple sanity check that device is able to send OR receive.
 *
 * @param h the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 * @param numTrials number of iterations of send or receive messaging to perform
 */
bool test_pointToPoint_device_send_or_recv(const handle_t& h, int numTrials)
{
  return detail::test_pointToPoint_device_send_or_recv(h, numTrials);
}

/**
 * A simple sanity check that device is able to send and receive at the same time.
 *
 * @param h the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 * @param numTrials number of iterations of send or receive messaging to perform
 */
bool test_pointToPoint_device_sendrecv(const handle_t& h, int numTrials)
{
  return detail::test_pointToPoint_device_sendrecv(h, numTrials);
}

/**
 * A simple sanity check that device is able to perform multiple concurrent sends and receives.
 *
 * @param h the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 * @param numTrials number of iterations of send or receive messaging to perform
 */
bool test_pointToPoint_device_multicast_sendrecv(const handle_t& h, int numTrials)
{
  return detail::test_pointToPoint_device_multicast_sendrecv(h, numTrials);
}

/**
 * A simple test that the comms can be split into 2 separate subcommunicators
 *
 * @param h the raft handle to use. This is expected to already have an
 *        initialized comms instance.
 * @param n_colors number of different colors to test
 */
bool test_commsplit(const handle_t& h, int n_colors) { return test_commsplit(h, n_colors); }

}  // namespace comms
};  // namespace raft
