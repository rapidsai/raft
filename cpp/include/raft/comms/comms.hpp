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

#pragma once

#include <memory>

namespace raft {
namespace comms {

typedef unsigned int request_t;
enum datatype_t { CHAR, UINT8, INT, UINT, INT64, UINT64, FLOAT, DOUBLE };
enum op_t { SUM, PROD, MIN, MAX };

/**
 * The resulting status of distributed stream synchronization
 */
enum status_t {
  commStatusSuccess,  // Synchronization successful
  commStatusError,    // An error occured querying sync status
  commStatusAbort     // A failure occurred in sync, queued operations aborted
};

class comms_iface {
 public:
  virtual ~comms_iface();

  virtual int getSize() const = 0;
  virtual int getRank() const = 0;

  virtual std::unique_ptr<comms_iface> commSplit(int color, int key) const = 0;

  virtual void barrier() const = 0;

  virtual status_t syncStream(cudaStream_t stream) const = 0;

  virtual void isend(const void* buf, int size, int dest, int tag,
                     request_t* request) const = 0;

  virtual void irecv(void* buf, int size, int source, int tag,
                     request_t* request) const = 0;

  virtual void waitall(int count, request_t array_of_requests[]) const = 0;

  virtual void allreduce(const void* sendbuff, void* recvbuff, int count,
                         datatype_t datatype, op_t op,
                         cudaStream_t stream) const = 0;

  virtual void bcast(void* buff, int count, datatype_t datatype, int root,
                     cudaStream_t stream) const = 0;

  virtual void reduce(const void* sendbuff, void* recvbuff, int count,
                      datatype_t datatype, op_t op, int root,
                      cudaStream_t stream) const = 0;

  virtual void allgather(const void* sendbuff, void* recvbuff, int sendcount,
                         datatype_t datatype, cudaStream_t stream) const = 0;

  virtual void allgatherv(const void* sendbuf, void* recvbuf,
                          const int recvcounts[], const int displs[],
                          datatype_t datatype, cudaStream_t stream) const = 0;

  virtual void reducescatter(const void* sendbuff, void* recvbuff,
                             int recvcount, datatype_t datatype, op_t op,
                             cudaStream_t stream) const = 0;
};

class comms_t : public comms_iface {
 public:
  comms_t(std::unique_ptr<comms_iface> impl) : _impl(impl.release()) {
    ASSERT(nullptr != _impl.get(), "ERROR: Invalid comms_iface used!");
  }

  int getSize() const { return _impl->getSize(); }

  int getRank() const { return _impl->getRank(); }

  std::unique_ptr<comms_iface> commSplit(int color, int key) const {
    return _impl->commSplit(color, key);
  }

  void barrier() const { _impl->barrier(); }

  status_t syncStream(cudaStream_t stream) const {
    return _impl->syncStream(stream);
  }

  void isend(const void* buf, int size, int dest, int tag,
             request_t* request) const {
    _impl->isend(buf, size, dest, tag, request);
  }

  void irecv(void* buf, int size, int source, int tag,
             request_t* request) const {
    _impl->irecv(buf, size, source, tag, request);
  }

  void waitall(int count, request_t array_of_requests[]) const {
    _impl->waitall(count, array_of_requests);
  }

  void allreduce(const void* sendbuff, void* recvbuff, int count,
                 datatype_t datatype, op_t op, cudaStream_t stream) const {
    _impl->allreduce(sendbuff, recvbuff, count, datatype, op, stream);
  }

  void bcast(void* buff, int count, datatype_t datatype, int root,
             cudaStream_t stream) const {
    _impl->bcast(buff, count, datatype, root, stream);
  }

  void reduce(const void* sendbuff, void* recvbuff, int count,
              datatype_t datatype, op_t op, int root,
              cudaStream_t stream) const {
    _impl->reduce(sendbuff, recvbuff, count, datatype, op, root, stream);
  }

  void allgather(const void* sendbuff, void* recvbuff, int sendcount,
                 datatype_t datatype, cudaStream_t stream) const {
    _impl->allgather(sendbuff, recvbuff, sendcount, datatype, stream);
  }

  void allgatherv(const void* sendbuf, void* recvbuf, const int recvcounts[],
                  const int displs[], datatype_t datatype,
                  cudaStream_t stream) const {
    _impl->allgatherv(sendbuf, recvbuf, recvcounts, displs, datatype, stream);
  }

  void reducescatter(const void* sendbuff, void* recvbuff, int recvcount,
                     datatype_t datatype, op_t op, cudaStream_t stream) const {
    _impl->reducescatter(sendbuff, recvbuff, recvcount, datatype, op, stream);
  }

 private:
  std::unique_ptr<comms_iface> _impl;
};

comms_iface::~comms_iface() {}

}  // namespace comms
}  // namespace raft
