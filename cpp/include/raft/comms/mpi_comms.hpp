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

#include "comms.hpp"

#include <cstdio>
#include <memory>

#include <raft/handle.hpp>
#include <raft/cudart_utils.h>

#define MPI_CHECK(call)                                                     \
  do {                                                                      \
    int status = call;                                                      \
    if (MPI_SUCCESS != status) {                                            \
      int mpi_error_string_lenght = 0;                                      \
      char mpi_error_string[MPI_MAX_ERROR_STRING];                          \
      MPI_Error_string(status, mpi_error_string, &mpi_error_string_lenght); \
      ASSERT(MPI_SUCCESS == status, "ERROR: MPI call='%s'. Reason:%s\n",    \
             #call, mpi_error_string);                                      \
    }                                                                       \
  } while (0)

#define MPI_CHECK_NO_THROW(call)                                            \
  do {                                                                      \
    int status = call;                                                      \
    if (MPI_SUCCESS != status) {                                            \
      int mpi_error_string_lenght = 0;                                      \
      char mpi_error_string[MPI_MAX_ERROR_STRING];                          \
      MPI_Error_string(status, mpi_error_string, &mpi_error_string_lenght); \
      CUML_LOG_ERROR("MPI call='%s' at file=%s line=%d failed with %s ",    \
                     #call, __FILE__, __LINE__, mpi_error_string);          \
    }                                                                       \
  } while (0)

namespace raft {
namespace comms {

static MPI_Datatype get_mpi_datatype(
  const datatype_t datatype) {
  switch (datatype) {
    case datatype_t::CHAR:
      return MPI_CHAR;
    case datatype_t::UINT8:
      return MPI_UNSIGNED_CHAR;
    case datatype_t::INT:
      return MPI_INT;
    case datatype_t::UINT:
      return MPI_UNSIGNED;
    case datatype_t::INT64:
      return MPI_LONG_LONG;
    case datatype_t::UINT64:
      return MPI_UNSIGNED_LONG_LONG;
    case datatype_t::FLOAT:
      return MPI_FLOAT;
    case datatype_t::DOUBLE:
      return MPI_DOUBLE;
    default:
      // Execution should never reach here. This takes care of compiler warning.
      return MPI_DOUBLE;
  }
}

static MPI_Op get_mpi_op(const op_t op) {
  switch (op) {
    case datatype_t::SUM:
      return MPI_SUM;
    case datatype_t::PROD:
      return MPI_PROD;
    case datatype_t::MIN:
      return MPI_MIN;
    case datatype_t::MAX:
      return MPI_MAX;
    default:
      // Execution should never reach here. This takes care of compiler warning.
      return MPI_MAX;
  }
}

void initialize_mpi_comms(handle_t& handle, MPI_Comm comm) {
  auto communicator = std::make_shared<comms_t>(
    std::unique_ptr<comms_iface>(
      new mpi_comms(comm)));
  handle.set_comms(communicator);
}

class mpi_comms : public comms_iface {
	mpi_comms(MPI_Comm comm, const bool owns_mpi_comm)
	  : _owns_mpi_comm(owns_mpi_comm),
	    _mpi_comm(comm),
	    _size(0),
	    _rank(1),
	    _next_request_id(0) {
	  int mpi_is_initialized = 0;
	  MPI_CHECK(MPI_Initialized(&mpi_is_initialized));
	  ASSERT(mpi_is_initialized, "ERROR: MPI is not initialized!");
	  MPI_CHECK(MPI_Comm_size(_mpi_comm, &_size));
	  MPI_CHECK(MPI_Comm_rank(_mpi_comm, &_rank));
	#ifdef HAVE_NCCL
	  //get NCCL unique ID at rank 0 and broadcast it to all others
	  ncclUniqueId id;
	  if (0 == _rank) NCCL_CHECK(ncclGetUniqueId(&id));
	  MPI_CHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, _mpi_comm));

	  //initializing NCCL
	  NCCL_CHECK(ncclCommInitRank(&_nccl_comm, _size, id, _rank));
	#endif
	}

	virtual ~mpi_comms() {
	#ifdef HAVE_NCCL
	  //finalizing NCCL
	  NCCL_CHECK_NO_THROW(ncclCommDestroy(_nccl_comm));
	#endif
	  if (_owns_mpi_comm) {
	    MPI_CHECK_NO_THROW(MPI_Comm_free(&_mpi_comm));
	  }
	}

	int get_size() const { return _size; }

	int get_rank() const { return _rank; }

	std::unique_ptr<comms_iface>
	comm_split(int color, int key) const {
	  MPI_Comm new_comm;
	  MPI_CHECK(MPI_Comm_split(_mpi_comm, color, key, &new_comm));
	  return std::unique_ptr<comms_iface>(
	    new mpi_comms(new_comm, true));
	}

	void barrier() const {
	  MPI_CHECK(MPI_Barrier(_mpi_comm));
	}

	void isend(const void* buf, size_t size, int dest,
	                                     int tag, request_t* request) const {
	  MPI_Request mpi_req;
	  request_t req_id;
	  if (_free_requests.empty()) {
	    req_id = _next_request_id++;
	  } else {
	    auto it = _free_requests.begin();
	    req_id = *it;
	    _free_requests.erase(it);
	  }
	  MPI_CHECK(MPI_Isend(buf, size, MPI_BYTE, dest, tag, _mpi_comm, &mpi_req));
	  _requests_in_flight.insert(std::make_pair(req_id, mpi_req));
	  *request = req_id;
	}

	void irecv(void* buf, size_t size, int source, int tag,
	                                     request_t* request) const {
	  if (source == CUML_ANY_SOURCE) source = MPI_ANY_SOURCE;

	  MPI_Request mpi_req;
	  request_t req_id;
	  if (_free_requests.empty()) {
	    req_id = _next_request_id++;
	  } else {
	    auto it = _free_requests.begin();
	    req_id = *it;
	    _free_requests.erase(it);
	  }

	  MPI_CHECK(MPI_Irecv(buf, size, MPI_BYTE, source, tag, _mpi_comm, &mpi_req));
	  _requests_in_flight.insert(std::make_pair(req_id, mpi_req));
	  *request = req_id;
	}

	void waitall(int count,
	                                       request_t array_of_requests[]) const {
	  std::vector<MPI_Request> requests;
	  requests.reserve(count);
	  for (int i = 0; i < count; ++i) {
	    auto req_it = _requests_in_flight.find(array_of_requests[i]);
	    ASSERT(_requests_in_flight.end() != req_it,
	           "ERROR: waitall on invalid request: %d", array_of_requests[i]);
	    requests.push_back(req_it->second);
	    _free_requests.insert(req_it->first);
	    _requests_in_flight.erase(req_it);
	  }
	  MPI_CHECK(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
	}

	void allreduce(const void* sendbuff, void* recvbuff,
	                                         size_t count, datatype_t datatype,
	                                         op_t op, cudaStream_t stream) const {
	#ifdef HAVE_NCCL
	  NCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, count, getNCCLDatatype(datatype),
	                           getNCCLOp(op), _nccl_comm, stream));
	#else
	  CUDA_CHECK(cudaStreamSynchronize(stream));
	  MPI_CHECK(MPI_Allreduce(sendbuff, recvbuff, count, get_mpi_datatype(datatype),
	                          get_mpi_op(op), _mpi_comm));
	#endif
	}

	void bcast(void* buff, size_t count, datatype_t datatype,
	                                     int root, cudaStream_t stream) const {
	#ifdef HAVE_NCCL
	  NCCL_CHECK(ncclBroadcast(buff, buff, count, getNCCLDatatype(datatype), root,
	                           _nccl_comm, stream));
	#else
	  CUDA_CHECK(cudaStreamSynchronize(stream));
	  MPI_CHECK(MPI_Bcast(buff, count, get_mpi_datatype(datatype), root, _mpi_comm));
	#endif
	}

	void reduce(const void* sendbuff, void* recvbuff,
	                                      size_t count, datatype_t datatype, op_t op,
	                                      int root, cudaStream_t stream) const {
	#ifdef HAVE_NCCL
	  NCCL_CHECK(ncclReduce(sendbuff, recvbuff, count, getNCCLDatatype(datatype),
	                        getNCCLOp(op), root, _nccl_comm, stream));
	#else
	  CUDA_CHECK(cudaStreamSynchronize(stream));
	  MPI_CHECK(MPI_Reduce(sendbuff, recvbuff, count, get_mpi_datatype(datatype),
	                       get_mpi_op(op), root, _mpi_comm));
	#endif
	}

	void allgather(const void* sendbuff, void* recvbuff,
	                                         size_t sendcount, datatype_t datatype,
	                                         cudaStream_t stream) const {
	#ifdef HAVE_NCCL
	  NCCL_CHECK(ncclAllGather(sendbuff, recvbuff, sendcount,
	                           getNCCLDatatype(datatype), _nccl_comm, stream));
	#else
	  CUDA_CHECK(cudaStreamSynchronize(stream));
	  MPI_CHECK(MPI_Allgather(sendbuff, sendcount, get_mpi_datatype(datatype),
	                          recvbuff, sendcount, get_mpi_datatype(datatype),
	                          _mpi_comm));
	#endif
	}

	void allgatherv(const void* sendbuf, void* recvbuf,
	                                          const size_t recvcounts[],
	                                          const int displs[],
	                                          datatype_t datatype,
	                                          cudaStream_t stream) const {
	#ifdef HAVE_NCCL
	  //From: "An Empirical Evaluation of Allgatherv on Multi-GPU Systems" - https://arxiv.org/pdf/1812.05964.pdf
	  //Listing 1 on page 4.
	  for (int root = 0; root < _size; ++root) {
	    NCCL_CHECK(ncclBroadcast(
	      sendbuf,
	      static_cast<char*>(recvbuf) + displs[root] * get_datatype_size(datatype),
	      recvcounts[root], getNCCLDatatype(datatype), root, _nccl_comm, stream));
	  }
	#else
	  CUDA_CHECK(cudaStreamSynchronize(stream));
	  MPI_CHECK(MPI_Allgatherv(sendbuf, recvcounts[_rank], get_mpi_datatype(datatype),
	                           recvbuf, recvcounts, displs,
	                           get_mpi_datatype(datatype), _mpi_comm));
	#endif
	}

	void reducescatter(const void* sendbuff,
	                                             void* recvbuff, size_t recvcount,
	                                             datatype_t datatype, op_t op,
	                                             cudaStream_t stream) const {
	#ifdef HAVE_NCCL
	  NCCL_CHECK(ncclReduceScatter(sendbuff, recvbuff, recvcount,
	                               getNCCLDatatype(datatype), getNCCLOp(op),
	                               _nccl_comm, stream));
	#else
	  CUDA_CHECK(cudaStreamSynchronize(stream));
	  std::vector<int> recvcounts(_size, recvcount);
	  MPI_CHECK(MPI_Reduce_scatter(sendbuff, recvbuff, recvcounts.data(),
	                               get_mpi_datatype(datatype), get_mpi_op(op),
	                               _mpi_comm));
	#endif
	}

	status_t sync_stream(
	  cudaStream_t stream) const {
	#ifdef HAVE_NCCL
	  cudaError_t cudaErr;
	  ncclResult_t ncclErr, ncclAsyncErr;
	  while (1) {
	    cudaErr = cudaStreamQuery(stream);
	    if (cudaErr == cudaSuccess) return status_t::commStatusSuccess;

	    if (cudaErr != cudaErrorNotReady) {
	      // An error occurred querying the status of the stream
	      return status_t::commStatusError;
	    }

	    ncclErr = ncclCommGetAsyncError(_nccl_comm, &ncclAsyncErr);
	    if (ncclErr != ncclSuccess) {
	      // An error occurred retrieving the asynchronous error
	      return status_t::commStatusError;
	    }

	    if (ncclAsyncErr != ncclSuccess) {
	      // An asynchronous error happened. Stop the operation and destroy
	      // the communicator
	      ncclErr = ncclCommAbort(_nccl_comm);
	      if (ncclErr != ncclSuccess)
	        // Caller may abort with an exception or try to re-create a new communicator.
	        return status_t::commStatusAbort;
	    }

	    // Let other threads (including NCCL threads) use the CPU.
	    pthread_yield();
	  }
	#else
	  CUDA_CHECK(cudaStreamSynchronize(stream));
	  return status_t::commStatusSuccess;
	#endif
	};

};
}  // end namespace comms
}  // end namespace raft
