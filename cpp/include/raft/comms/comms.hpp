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

class comms_t {
 public:

  typedef unsigned int request_t;
  enum datatype_t { CHAR, UINT8, INT, UINT, INT64, UINT64, FLOAT, DOUBLE };
  enum op_t { SUM, PROD, MIN, MAX };

  /**
   * The resulting status of distributed stream synchronization
   */
  enum status_t {
	commStatusSuccess,  // Synchronization successful
	commStatusError,    // An error occured querying sync status
	commStatusAbort
  };  // A failure occurred in sync, queued operations aborted

  virtual ~comms_t();

  virtual int getSize() const = 0;
  virtual int getRank() const = 0;

  virtual std::unique_ptr<comms_t> commSplit(int color, int key) const = 0;

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


template <typename T>
comms_t::datatype_t getDataType(T a);

template <>
comms_t::datatype_t getDataType<char>(char a) {
  return comms_t::CHAR;
}

template <>
comms_t::datatype_t getDataType<uint8_t>(uint8_t a) {
  return comms_t::UINT8;
}

template <>
comms_t::datatype_t getDataType<int>(int a) {
  return comms_t::INT;
}

template <>
comms_t::datatype_t getDataType<uint32_t>(uint32_t a) {
  return comms_t::UINT;
}

template <>
comms_t::datatype_t getDataType<int64_t>(int64_t a) {
  return comms_t::INT64;
}

template <>
comms_t::datatype_t getDataType<uint64_t>(uint64_t a) {
  return comms_t::UINT64;
}

template <>
comms_t::datatype_t getDataType<float>(float a) {
  return comms_t::FLOAT;
}

template <>
comms_t::datatype_t getDataType<double>(double a) {
  return comms_t::DOUBLE;
}
} /// namespace comms
}  // namespace raft
