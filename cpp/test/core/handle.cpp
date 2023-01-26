/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cstddef>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <unordered_map>

namespace raft {

using namespace comms;
class mock_comms : public comms_iface {
 public:
  mock_comms(int n) : n_ranks(n) {}
  ~mock_comms() {}

  int get_size() const override { return n_ranks; }

  int get_rank() const override { return 0; }

  std::unique_ptr<comms_iface> comm_split(int color, int key) const
  {
    return std::unique_ptr<comms_iface>(new mock_comms(0));
  }

  void barrier() const {}

  void get_request_id(request_t* req) const {}

  void isend(const void* buf, size_t size, int dest, int tag, request_t* request) const {}

  void irecv(void* buf, size_t size, int source, int tag, request_t* request) const {}

  void waitall(int count, request_t array_of_requests[]) const {}

  void allreduce(const void* sendbuff,
                 void* recvbuff,
                 size_t count,
                 datatype_t datatype,
                 op_t op,
                 cudaStream_t stream) const
  {
  }

  void bcast(void* buff, size_t count, datatype_t datatype, int root, cudaStream_t stream) const {}

  void bcast(const void* sendbuff,
             void* recvbuff,
             size_t count,
             datatype_t datatype,
             int root,
             cudaStream_t stream) const
  {
  }

  void reduce(const void* sendbuff,
              void* recvbuff,
              size_t count,
              datatype_t datatype,
              op_t op,
              int root,
              cudaStream_t stream) const
  {
  }

  void allgather(const void* sendbuff,
                 void* recvbuff,
                 size_t sendcount,
                 datatype_t datatype,
                 cudaStream_t stream) const
  {
  }

  void allgatherv(const void* sendbuf,
                  void* recvbuf,
                  const size_t* recvcounts,
                  const size_t* displs,
                  datatype_t datatype,
                  cudaStream_t stream) const
  {
  }

  void gather(const void* sendbuff,
              void* recvbuff,
              size_t sendcount,
              datatype_t datatype,
              int root,
              cudaStream_t stream) const
  {
  }

  void gatherv(const void* sendbuff,
               void* recvbuff,
               size_t sendcount,
               const size_t* recvcounts,
               const size_t* displs,
               datatype_t datatype,
               int root,
               cudaStream_t stream) const
  {
  }

  void reducescatter(const void* sendbuff,
                     void* recvbuff,
                     size_t recvcount,
                     datatype_t datatype,
                     op_t op,
                     cudaStream_t stream) const
  {
  }

  status_t sync_stream(cudaStream_t stream) const { return status_t::SUCCESS; }

  // if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
  void device_send(const void* buf, size_t size, int dest, cudaStream_t stream) const {}

  // if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
  void device_recv(void* buf, size_t size, int source, cudaStream_t stream) const {}

  void device_sendrecv(const void* sendbuf,
                       size_t sendsize,
                       int dest,
                       void* recvbuf,
                       size_t recvsize,
                       int source,
                       cudaStream_t stream) const
  {
  }

  void device_multicast_sendrecv(const void* sendbuf,
                                 std::vector<size_t> const& sendsizes,
                                 std::vector<size_t> const& sendoffsets,
                                 std::vector<int> const& dests,
                                 void* recvbuf,
                                 std::vector<size_t> const& recvsizes,
                                 std::vector<size_t> const& recvoffsets,
                                 std::vector<int> const& sources,
                                 cudaStream_t stream) const
  {
  }

  void group_start() const {}

  void group_end() const {}

 private:
  int n_ranks;
};

void assert_handles_equal(raft::handle_t& handle_one, raft::handle_t& handle_two)
{
  // Assert shallow copied state
  ASSERT_EQ(handle_one.get_stream().value(), handle_two.get_stream().value());
  ASSERT_EQ(handle_one.get_stream_pool_size(), handle_two.get_stream_pool_size());

  // Sanity check to make sure non-corresponding streams are not equal
  ASSERT_NE(handle_one.get_stream_pool().get_stream(0).value(),
            handle_two.get_stream_pool().get_stream(1).value());

  for (size_t i = 0; i < handle_one.get_stream_pool_size(); ++i) {
    ASSERT_EQ(handle_one.get_stream_pool().get_stream(i).value(),
              handle_two.get_stream_pool().get_stream(i).value());
  }
}

TEST(Raft, HandleDefault)
{
  handle_t h;
  ASSERT_EQ(0, h.get_device());
  ASSERT_EQ(rmm::cuda_stream_per_thread, h.get_stream());
  ASSERT_NE(nullptr, h.get_cublas_handle());
  ASSERT_NE(nullptr, h.get_cusolver_dn_handle());
  ASSERT_NE(nullptr, h.get_cusolver_sp_handle());
  ASSERT_NE(nullptr, h.get_cusparse_handle());
}

TEST(Raft, Handle)
{
  // test stream pool creation
  constexpr std::size_t n_streams = 4;
  auto stream_pool                = std::make_shared<rmm::cuda_stream_pool>(n_streams);
  handle_t h(rmm::cuda_stream_default, stream_pool);
  ASSERT_EQ(n_streams, h.get_stream_pool_size());

  // test non default stream handle
  cudaStream_t stream;
  RAFT_CUDA_TRY(cudaStreamCreate(&stream));
  rmm::cuda_stream_view stream_view(stream);
  handle_t handle(stream_view);
  ASSERT_EQ(stream_view, handle.get_stream());
  handle.sync_stream(stream);
  RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}

TEST(Raft, DefaultConstructor)
{
  handle_t handle;

  // Make sure waiting on the default stream pool
  // does not fail.
  handle.wait_stream_pool_on_stream();
  handle.sync_stream_pool();

  auto s1 = handle.get_next_usable_stream();
  auto s2 = handle.get_stream();
  auto s3 = handle.get_next_usable_stream(5);

  ASSERT_EQ(s1, s2);
  ASSERT_EQ(s2, s3);
  ASSERT_EQ(0, handle.get_stream_pool_size());
}

TEST(Raft, GetHandleFromPool)
{
  constexpr std::size_t n_streams = 4;
  auto stream_pool                = std::make_shared<rmm::cuda_stream_pool>(n_streams);
  handle_t parent(rmm::cuda_stream_default, stream_pool);

  for (std::size_t i = 0; i < n_streams; i++) {
    auto worker_stream = parent.get_stream_from_stream_pool(i);
    handle_t child(worker_stream);
    ASSERT_EQ(parent.get_stream_from_stream_pool(i), child.get_stream());
  }

  parent.wait_stream_pool_on_stream();
}

TEST(Raft, Comms)
{
  handle_t handle;
  auto comm1 = std::make_shared<comms_t>(std::unique_ptr<comms_iface>(new mock_comms(2)));
  handle.set_comms(comm1);

  ASSERT_EQ(handle.get_comms().get_size(), 2);
}

TEST(Raft, SubComms)
{
  handle_t handle;
  auto comm1 = std::make_shared<comms_t>(std::unique_ptr<comms_iface>(new mock_comms(1)));
  handle.set_subcomm("key1", comm1);

  auto comm2 = std::make_shared<comms_t>(std::unique_ptr<comms_iface>(new mock_comms(2)));
  handle.set_subcomm("key2", comm2);

  ASSERT_EQ(handle.get_subcomm("key1").get_size(), 1);
  ASSERT_EQ(handle.get_subcomm("key2").get_size(), 2);
}

TEST(Raft, WorkspaceResource)
{
  handle_t handle;

  ASSERT_TRUE(dynamic_cast<const rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>*>(
                handle.get_workspace_resource()) == nullptr);
  ASSERT_EQ(rmm::mr::get_current_device_resource(), handle.get_workspace_resource());

  auto pool_mr = new rmm::mr::pool_memory_resource(rmm::mr::get_current_device_resource());
  std::shared_ptr<rmm::cuda_stream_pool> pool = {nullptr};
  handle_t handle2(rmm::cuda_stream_per_thread, pool, pool_mr);

  ASSERT_TRUE(dynamic_cast<const rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>*>(
                handle2.get_workspace_resource()) != nullptr);
  ASSERT_EQ(pool_mr, handle2.get_workspace_resource());

  delete pool_mr;
}

TEST(Raft, WorkspaceResourceCopy)
{
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(10);

  handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

  auto pool_mr = new rmm::mr::pool_memory_resource(rmm::mr::get_current_device_resource());

  handle_t copied_handle(handle, pool_mr);

  assert_handles_equal(handle, copied_handle);

  // Assert the workspace_resources are what we expect
  ASSERT_TRUE(dynamic_cast<const rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>*>(
                handle.get_workspace_resource()) == nullptr);

  ASSERT_TRUE(dynamic_cast<const rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>*>(
                copied_handle.get_workspace_resource()) != nullptr);
}

TEST(Raft, HandleCopy)
{
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(10);

  handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  handle_t copied_handle(handle);

  assert_handles_equal(handle, copied_handle);
}

TEST(Raft, HandleAssign)
{
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(10);

  handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  handle_t copied_handle = handle;

  assert_handles_equal(handle, copied_handle);
}

}  // namespace raft
