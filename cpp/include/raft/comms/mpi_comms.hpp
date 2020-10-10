/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cstdio>
#include <memory>

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <mpi.h>
#include <nccl.h>

#include <raft/cudart_utils.h>
#include <raft/comms/comms.hpp>
#include <raft/comms/util.hpp>
#include <raft/error.hpp>
#include <raft/handle.hpp>

#define MPI_TRY(call)                                                          \
  do {                                                                         \
    int status = call;                                                         \
    if (MPI_SUCCESS != status) {                                               \
      int mpi_error_string_lenght = 0;                                         \
      char mpi_error_string[MPI_MAX_ERROR_STRING];                             \
      MPI_Error_string(status, mpi_error_string, &mpi_error_string_lenght);    \
      RAFT_EXPECTS(MPI_SUCCESS == status, "ERROR: MPI call='%s'. Reason:%s\n", \
                   #call, mpi_error_string);                                   \
    }                                                                          \
  } while (0)

#define MPI_TRY_NO_THROW(call)                                              \
  do {                                                                      \
    int status = call;                                                      \
    if (MPI_SUCCESS != status) {                                            \
      int mpi_error_string_lenght = 0;                                      \
      char mpi_error_string[MPI_MAX_ERROR_STRING];                          \
      MPI_Error_string(status, mpi_error_string, &mpi_error_string_lenght); \
      printf("MPI call='%s' at file=%s line=%d failed with %s ", #call,     \
             __FILE__, __LINE__, mpi_error_string);                         \
    }                                                                       \
  } while (0)

namespace raft {
namespace comms {

constexpr MPI_Datatype get_mpi_datatype(const datatype_t datatype) {
  switch (datatype) {
    case datatype_t::CHAR:
      return MPI_CHAR;
    case datatype_t::UINT8:
      return MPI_UNSIGNED_CHAR;
    case datatype_t::INT32:
      return MPI_INT;
    case datatype_t::UINT32:
      return MPI_UNSIGNED;
    case datatype_t::INT64:
      return MPI_LONG_LONG;
    case datatype_t::UINT64:
      return MPI_UNSIGNED_LONG_LONG;
    case datatype_t::FLOAT32:
      return MPI_FLOAT;
    case datatype_t::FLOAT64:
      return MPI_DOUBLE;
    default:
      // Execution should never reach here. This takes care of compiler warning.
      return MPI_DOUBLE;
  }
}

constexpr MPI_Op get_mpi_op(const op_t op) {
  switch (op) {
    case op_t::SUM:
      return MPI_SUM;
    case op_t::PROD:
      return MPI_PROD;
    case op_t::MIN:
      return MPI_MIN;
    case op_t::MAX:
      return MPI_MAX;
    default:
      // Execution should never reach here. This takes care of compiler warning.
      return MPI_MAX;
  }
}

class mpi_comms : public comms_iface {
 public:
  mpi_comms(MPI_Comm comm, const bool owns_mpi_comm)
    : owns_mpi_comm_(owns_mpi_comm),
      mpi_comm_(comm),
      size_(0),
      rank_(1),
      next_request_id_(0) {
    int mpi_is_initialized = 0;
    MPI_TRY(MPI_Initialized(&mpi_is_initialized));
    RAFT_EXPECTS(mpi_is_initialized, "ERROR: MPI is not initialized!");
    MPI_TRY(MPI_Comm_size(mpi_comm_, &size_));
    MPI_TRY(MPI_Comm_rank(mpi_comm_, &rank_));
    //get NCCL unique ID at rank 0 and broadcast it to all others
    ncclUniqueId id;
    if (0 == rank_) NCCL_TRY(ncclGetUniqueId(&id));
    MPI_TRY(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, mpi_comm_));

    //initializing NCCL
    NCCL_TRY(ncclCommInitRank(&nccl_comm_, size_, id, rank_));
  }

  virtual ~mpi_comms() {
    //finalizing NCCL
    NCCL_TRY_NO_THROW(ncclCommDestroy(nccl_comm_));
    if (owns_mpi_comm_) {
      MPI_TRY_NO_THROW(MPI_Comm_free(&mpi_comm_));
    }
  }

  int get_size() const { return size_; }

  int get_rank() const { return rank_; }

  // FIXME: a temporary hack, should be removed
  ncclComm_t get_nccl_comm() const { return nccl_comm_; }

  std::unique_ptr<comms_iface> comm_split(int color, int key) const {
    MPI_Comm new_comm;
    MPI_TRY(MPI_Comm_split(mpi_comm_, color, key, &new_comm));
    return std::unique_ptr<comms_iface>(new mpi_comms(new_comm, true));
  }

  void barrier() const { MPI_TRY(MPI_Barrier(mpi_comm_)); }

  void isend(const void* buf, size_t size, int dest, int tag,
             request_t* request) const {
    MPI_Request mpi_req;
    request_t req_id;
    if (free_requests_.empty()) {
      req_id = next_request_id_++;
    } else {
      auto it = free_requests_.begin();
      req_id = *it;
      free_requests_.erase(it);
    }
    MPI_TRY(MPI_Isend(buf, size, MPI_BYTE, dest, tag, mpi_comm_, &mpi_req));
    requests_in_flight_.insert(std::make_pair(req_id, mpi_req));
    *request = req_id;
  }

  void irecv(void* buf, size_t size, int source, int tag,
             request_t* request) const {
    MPI_Request mpi_req;
    request_t req_id;
    if (free_requests_.empty()) {
      req_id = next_request_id_++;
    } else {
      auto it = free_requests_.begin();
      req_id = *it;
      free_requests_.erase(it);
    }

    MPI_TRY(MPI_Irecv(buf, size, MPI_BYTE, source, tag, mpi_comm_, &mpi_req));
    requests_in_flight_.insert(std::make_pair(req_id, mpi_req));
    *request = req_id;
  }

  void waitall(int count, request_t array_of_requests[]) const {
    std::vector<MPI_Request> requests;
    requests.reserve(count);
    for (int i = 0; i < count; ++i) {
      auto req_it = requests_in_flight_.find(array_of_requests[i]);
      RAFT_EXPECTS(requests_in_flight_.end() != req_it,
                   "ERROR: waitall on invalid request: %d",
                   array_of_requests[i]);
      requests.push_back(req_it->second);
      free_requests_.insert(req_it->first);
      requests_in_flight_.erase(req_it);
    }
    MPI_TRY(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
  }

  void allreduce(const void* sendbuff, void* recvbuff, size_t count,
                 datatype_t datatype, op_t op, cudaStream_t stream) const {
    NCCL_TRY(ncclAllReduce(sendbuff, recvbuff, count,
                           get_nccl_datatype(datatype), get_nccl_op(op),
                           nccl_comm_, stream));
  }

  void bcast(void* buff, size_t count, datatype_t datatype, int root,
             cudaStream_t stream) const {
    NCCL_TRY(ncclBroadcast(buff, buff, count, get_nccl_datatype(datatype), root,
                           nccl_comm_, stream));
  }

  void reduce(const void* sendbuff, void* recvbuff, size_t count,
              datatype_t datatype, op_t op, int root,
              cudaStream_t stream) const {
    NCCL_TRY(ncclReduce(sendbuff, recvbuff, count, get_nccl_datatype(datatype),
                        get_nccl_op(op), root, nccl_comm_, stream));
  }

  void allgather(const void* sendbuff, void* recvbuff, size_t sendcount,
                 datatype_t datatype, cudaStream_t stream) const {
    NCCL_TRY(ncclAllGather(sendbuff, recvbuff, sendcount,
                           get_nccl_datatype(datatype), nccl_comm_, stream));
  }

  void allgatherv(const void* sendbuf, void* recvbuf, const size_t* recvcounts,
                  const size_t* displs, datatype_t datatype,
                  cudaStream_t stream) const {
    //From: "An Empirical Evaluation of Allgatherv on Multi-GPU Systems" - https://arxiv.org/pdf/1812.05964.pdf
    //Listing 1 on page 4.
    for (int root = 0; root < size_; ++root) {
      NCCL_TRY(ncclBroadcast(sendbuf,
                             static_cast<char*>(recvbuf) +
                               displs[root] * get_datatype_size(datatype),
                             recvcounts[root], get_nccl_datatype(datatype),
                             root, nccl_comm_, stream));
    }
  }

  void reducescatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                     datatype_t datatype, op_t op, cudaStream_t stream) const {
    NCCL_TRY(ncclReduceScatter(sendbuff, recvbuff, recvcount,
                               get_nccl_datatype(datatype), get_nccl_op(op),
                               nccl_comm_, stream));
  }

  status_t sync_stream(cudaStream_t stream) const {
    cudaError_t cudaErr;
    ncclResult_t ncclErr, ncclAsyncErr;
    while (1) {
      cudaErr = cudaStreamQuery(stream);
      if (cudaErr == cudaSuccess) return status_t::SUCCESS;

      if (cudaErr != cudaErrorNotReady) {
        // An error occurred querying the status of the stream
        return status_t::ERROR;
      }

      ncclErr = ncclCommGetAsyncError(nccl_comm_, &ncclAsyncErr);
      if (ncclErr != ncclSuccess) {
        // An error occurred retrieving the asynchronous error
        return status_t::ERROR;
      }

      if (ncclAsyncErr != ncclSuccess) {
        // An asynchronous error happened. Stop the operation and destroy
        // the communicator
        ncclErr = ncclCommAbort(nccl_comm_);
        if (ncclErr != ncclSuccess)
          // Caller may abort with an exception or try to re-create a new communicator.
          return status_t::ABORT;
      }

      // Let other threads (including NCCL threads) use the CPU.
      pthread_yield();
    }
  };

 private:
  bool owns_mpi_comm_;
  MPI_Comm mpi_comm_;

  ncclComm_t nccl_comm_;
  int size_;
  int rank_;
  mutable request_t next_request_id_;
  mutable std::unordered_map<request_t, MPI_Request> requests_in_flight_;
  mutable std::unordered_set<request_t> free_requests_;
};

inline void initialize_mpi_comms(handle_t* handle, MPI_Comm comm) {
  auto communicator = std::make_shared<comms_t>(
    std::unique_ptr<comms_iface>(new mpi_comms(comm, true)));
  handle->set_comms(communicator);
};

};  // end namespace comms
};  // end namespace raft
