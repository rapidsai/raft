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

#include <raft/cudart_utils.h>
#include <memory>

namespace raft {
namespace comms {

typedef unsigned int request_t;
enum class datatype_t {
  CHAR,
  UINT8,
  INT32,
  UINT32,
  INT64,
  UINT64,
  FLOAT32,
  FLOAT64
};
enum class op_t { SUM, PROD, MIN, MAX };

/**
 * The resulting status of distributed stream synchronization
 */
enum class status_t {
  SUCCESS,  // Synchronization successful
  ERROR,    // An error occured querying sync status
  ABORT     // A failure occurred in sync, queued operations aborted
};

template <typename value_t>
constexpr datatype_t get_type();

template <>
constexpr datatype_t get_type<char>() {
  return datatype_t::CHAR;
}

template <>
constexpr datatype_t get_type<uint8_t>() {
  return datatype_t::UINT8;
}

template <>
constexpr datatype_t get_type<int>() {
  return datatype_t::INT32;
}

template <>
constexpr datatype_t get_type<uint32_t>() {
  return datatype_t::UINT32;
}

template <>
constexpr datatype_t get_type<int64_t>() {
  return datatype_t::INT64;
}

template <>
constexpr datatype_t get_type<uint64_t>() {
  return datatype_t::UINT64;
}

template <>
constexpr datatype_t get_type<float>() {
  return datatype_t::FLOAT32;
}

template <>
constexpr datatype_t get_type<double>() {
  return datatype_t::FLOAT64;
}

class comms_iface {
 public:
  virtual int get_size() const = 0;
  virtual int get_rank() const = 0;

  virtual std::unique_ptr<comms_iface> comm_split(int color, int key) const = 0;
  virtual void barrier() const = 0;

  virtual status_t sync_stream(cudaStream_t stream) const = 0;

  virtual void isend(const void* buf, size_t size, int dest, int tag,
                     request_t* request) const = 0;

  virtual void irecv(void* buf, size_t size, int source, int tag,
                     request_t* request) const = 0;

  virtual void waitall(int count, request_t array_of_requests[]) const = 0;

  virtual void allreduce(const void* sendbuff, void* recvbuff, size_t count,
                         datatype_t datatype, op_t op,
                         cudaStream_t stream) const = 0;

  virtual void bcast(void* buff, size_t count, datatype_t datatype, int root,
                     cudaStream_t stream) const = 0;

  virtual void reduce(const void* sendbuff, void* recvbuff, size_t count,
                      datatype_t datatype, op_t op, int root,
                      cudaStream_t stream) const = 0;

  virtual void allgather(const void* sendbuff, void* recvbuff, size_t sendcount,
                         datatype_t datatype, cudaStream_t stream) const = 0;

  virtual void allgatherv(const void* sendbuf, void* recvbuf,
                          const size_t recvcounts[], const size_t displs[],
                          datatype_t datatype, cudaStream_t stream) const = 0;

  virtual void reducescatter(const void* sendbuff, void* recvbuff,
                             size_t recvcount, datatype_t datatype, op_t op,
                             cudaStream_t stream) const = 0;
};

class comms_t {
 public:
  comms_t(std::unique_ptr<comms_iface> impl) : impl_(impl.release()) {
    ASSERT(nullptr != impl_.get(), "ERROR: Invalid comms_iface used!");
  }

  /**
   * Virtual Destructor to enable polymorphism
   */
  virtual ~comms_t() {}

  /**
   * Returns the size of the communicator clique
   */

  int get_size() const { return impl_->get_size(); }

  /**
   * Returns the local rank
   */
  int get_rank() const { return impl_->get_rank(); }

  /**
   * Splits the current communicator clique into sub-cliques matching
   * the given color and key
   *
   * @param color ranks w/ the same color are placed in the same communicator
   * @param key controls rank assignment
   */
  std::unique_ptr<comms_iface> comm_split(int color, int key) const {
    return impl_->comm_split(color, key);
  }

  /**
   * Performs a collective barrier synchronization
   */
  void barrier() const { impl_->barrier(); }

  /**
   * Some collective communications implementations (eg. NCCL) might use asynchronous
   * collectives that are explicitly synchronized. It's important to always synchronize
   * using this method to allow failures to propagate, rather than `cudaStreamSynchronize()`,
   * to prevent the potential for deadlocks.
   *
   * @param stream the cuda stream to sync collective operations on
   */
  status_t sync_stream(cudaStream_t stream) const {
    return impl_->sync_stream(stream);
  }

  /**
   * Performs an asynchronous point-to-point send
   * @tparam value_t the type of data to send
   * @param buf pointer to array of data to send
   * @param size number of elements in buf
   * @param dest destination rank
   * @param tag a tag to use for the receiver to filter
   * @param request pointer to hold returned request_t object.
   * 		This will be used in `waitall()` to synchronize until the message is delivered (or fails).
   */
  template <typename value_t>
  void isend(const value_t* buf, size_t size, int dest, int tag,
             request_t* request) const {
    impl_->isend(static_cast<const void*>(buf), size * sizeof(value_t), dest,
                 tag, request);
  }

  /**
   * Performs an asynchronous point-to-point receive
   * @tparam value_t the type of data to be received
   * @param buf pointer to (initialized) array that will hold received data
   * @param size number of elements in buf
   * @param source source rank
   * @param tag a tag to use for message filtering
   * @param request pointer to hold returned request_t object.
   * 		This will be used in `waitall()` to synchronize until the message is delivered (or fails).
   */
  template <typename value_t>
  void irecv(value_t* buf, size_t size, int source, int tag,
             request_t* request) const {
    impl_->irecv(static_cast<void*>(buf), size * sizeof(value_t), source, tag,
                 request);
  }

  /**
   * Synchronize on an array of request_t objects returned from isend/irecv
   * @param count number of requests to synchronize on
   * @param array_of_requests an array of request_t objects returned from isend/irecv
   */
  void waitall(int count, request_t array_of_requests[]) const {
    impl_->waitall(count, array_of_requests);
  }

  /**
   * Perform an allreduce collective
   * @tparam value_t datatype of underlying buffers
   * @param sendbuff data to reduce
   * @param recvbuff buffer to hold the reduced result
   * @param count number of elements in sendbuff
   * @param op reduction operation to perform
   * @param stream CUDA stream to synchronize operation
   */
  template <typename value_t>
  void allreduce(const value_t* sendbuff, value_t* recvbuff, size_t count,
                 op_t op, cudaStream_t stream) const {
    impl_->allreduce(static_cast<const void*>(sendbuff),
                     static_cast<void*>(recvbuff), count, get_type<value_t>(),
                     op, stream);
  }

  /**
   * Broadcast data from one rank to the rest
   * @tparam value_t datatype of underlying buffers
   * @param buff buffer to send
   * @param count number of elements if buff
   * @param root the rank initiating the broadcast
   * @param stream CUDA stream to synchronize operation
   */
  template <typename value_t>
  void bcast(value_t* buff, size_t count, int root, cudaStream_t stream) const {
    impl_->bcast(static_cast<void*>(buff), count, get_type<value_t>(), root,
                 stream);
  }

  /**
   * Reduce data from many ranks down to a single rank
   * @tparam value_t datatype of underlying buffers
   * @param sendbuff buffer containing data to reduce
   * @param recvbuff buffer containing reduced data (only needs to be initialized on root)
   * @param count number of elements in sendbuff
   * @param op reduction operation to perform
   * @param root rank to store the results
   * @param stream CUDA stream to synchronize operation
   */
  template <typename value_t>
  void reduce(const value_t* sendbuff, value_t* recvbuff, size_t count, op_t op,
              int root, cudaStream_t stream) const {
    impl_->reduce(static_cast<const void*>(sendbuff),
                  static_cast<void*>(recvbuff), count, get_type<value_t>(), op,
                  root, stream);
  }

  /**
   * Gathers data from each rank onto all ranks
   * @tparam value_t datatype of underlying buffers
   * @param sendbuff buffer containing data to gather
   * @param recvbuff buffer containing gathered data from all ranks
   * @param sendcount number of elements in send buffer
   * @param stream CUDA stream to synchronize operation
   */
  template <typename value_t>
  void allgather(const value_t* sendbuff, value_t* recvbuff, size_t sendcount,
                 cudaStream_t stream) const {
    impl_->allgather(static_cast<const void*>(sendbuff),
                     static_cast<void*>(recvbuff), sendcount,
                     get_type<value_t>(), stream);
  }

  /**
   * Gathers data from all ranks and delivers to combined data to all ranks
   * @param value_t datatype of underlying buffers
   * @param sendbuff buffer containing data to send
   * @param recvbuff buffer containing data to receive
   * @param recvcounts array (of length num_ranks size) containing the number of elements
   * 		that are to be received from each rank
   * @param displs array (of length num_ranks size) to specify the displacement (relative to recvbuf)
   * 		at which to place the incoming data from each rank
   * @param stream CUDA stream to synchronize operation
   */
  template <typename value_t>
  void allgatherv(const value_t* sendbuf, value_t* recvbuf,
                  const size_t recvcounts[], const size_t displs[],
                  cudaStream_t stream) const {
    impl_->allgatherv(static_cast<const void*>(sendbuf),
                      static_cast<void*>(recvbuf), recvcounts, displs,
                      get_type<value_t>(), stream);
  }

  /**
   * Reduces data from all ranks then scatters the result across ranks
   * @tparam value_t datatype of underlying buffers
   * @param sendbuff buffer containing data to send (size recvcount * num_ranks)
   * @param recvbuff buffer containing received data
   * @param op reduction operation to perform
   * @param stream CUDA stream to synchronize operation
   */
  template <typename value_t>
  void reducescatter(const value_t* sendbuff, value_t* recvbuff,
                     size_t recvcount, op_t op, cudaStream_t stream) const {
    impl_->reducescatter(static_cast<const void*>(sendbuff),
                         static_cast<void*>(recvbuff), recvcount,
                         get_type<value_t>(), op, stream);
  }

 private:
  std::unique_ptr<comms_iface> impl_;
};

}  // namespace comms
}  // namespace raft
