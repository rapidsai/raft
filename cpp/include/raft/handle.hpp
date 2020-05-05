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

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <memory>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>

///@todo: enable once we have migrated cuml-comms layer too
//#include <common/cuml_comms_int.hpp>

#include "allocator.hpp"
#include "cudart_utils.h"
#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/cusolver_wrappers.h>
#include <raft/sparse/cusparse_wrappers.h>

namespace raft {

/**
 * @brief Main handle object that stores all necessary context used for calling
 *        necessary cuda kernels and/or libraries
 */
class handle_t {
 private:
  static constexpr int NumDefaultWorkerStreams = 0;

 public:
  /**
   * @brief Construct a handle with the specified number of worker streams
   *
   * @param[in] n_streams number worker streams to be created
   */
  handle_t(int n_streams = NumDefaultWorkerStreams)
    : _dev_id([]() -> int {
      int cur_dev = -1;
      CUDA_CHECK(cudaGetDevice(&cur_dev));
      return cur_dev;
    }()),
    _num_streams(n_streams),
    _cublasInitialized(false),
    _cusolverDnInitialized(false),
    _cusolverSpInitialized(false),
    _cusparseInitialized(false),
    _deviceAllocator(std::make_shared<defaultDeviceAllocator>()),
    _hostAllocator(std::make_shared<defaultHostAllocator>()),
    _userStream(NULL),
    _devicePropInitialized(false) {
    createResources();
  }

  /** Destroys all held-up resources */
  ~handle_t() { destroyResources(); }

  int getDevice() const { return _dev_id; }

  void setStream(cudaStream_t stream) { _userStream = stream; }
  cudaStream_t getStream() const { return _userStream; }

  void setDeviceAllocator(std::shared_ptr<deviceAllocator> allocator) {
    _deviceAllocator = allocator;
  }
  std::shared_ptr<deviceAllocator> getDeviceAllocator() const {
    return _deviceAllocator;
  }

  void setHostAllocator(std::shared_ptr<hostAllocator> allocator) {
    _hostAllocator = allocator;
  }
  std::shared_ptr<hostAllocator> getHostAllocator() const {
    return _hostAllocator;
  }

  cublasHandle_t getCublasHandle() const {
    if (!_cublasInitialized) {
      CUBLAS_CHECK(cublasCreate(&_cublas_handle));
      _cublasInitialized = true;
    }
    return _cublas_handle;
  }

  cusolverDnHandle_t getcusolverDnHandle() const {
    if (!_cusolverDnInitialized) {
      CUSOLVER_CHECK(cusolverDnCreate(&_cusolverDn_handle));
      _cusolverDnInitialized = true;
    }
    return _cusolverDn_handle;
  }

  cusolverSpHandle_t getcusolverSpHandle() const {
    if (!_cusolverSpInitialized) {
      CUSOLVER_CHECK(cusolverSpCreate(&_cusolverSp_handle));
      _cusolverSpInitialized = true;
    }
    return _cusolverSp_handle;
  }

  cusparseHandle_t getcusparseHandle() const {
    if (!_cusparseInitialized) {
      CUSPARSE_CHECK(cusparseCreate(&_cusparse_handle));
      _cusparseInitialized = true;
    }
    return _cusparse_handle;
  }

  cudaStream_t getInternalStream(int sid) const { return _streams[sid]; }
  int getNumInternalStreams() const { return _num_streams; }
  std::vector<cudaStream_t> getInternalStreams() const {
    std::vector<cudaStream_t> int_streams_vec(_num_streams);
    for (auto s : _streams) {
      int_streams_vec.push_back(s);
    }
    return int_streams_vec;
  }

  void waitOnUserStream() const {
    CUDA_CHECK(cudaEventRecord(_event, _userStream));
    for (auto s : _streams) {
      CUDA_CHECK(cudaStreamWaitEvent(s, _event, 0));
    }
  }

  void waitOnInternalStreams() const {
    for (auto s : _streams) {
      CUDA_CHECK(cudaEventRecord(_event, s));
      CUDA_CHECK(cudaStreamWaitEvent(_userStream, _event, 0));
    }
  }

  ///@todo: enable this once we have cuml-comms migrated
  // void setCommunicator(
  //   std::shared_ptr<MLCommon::cumlCommunicator> communicator);
  // const MLCommon::cumlCommunicator& getCommunicator() const;
  // bool commsInitialized() const;

  const cudaDeviceProp& getDeviceProperties() const {
    if (!_devicePropInitialized) {
      CUDA_CHECK(cudaGetDeviceProperties(&_prop, _dev_id));
      _devicePropInitialized = true;
    }
    return _prop;
  }

 private:
  const int _dev_id;
  const int _num_streams;
  std::vector<cudaStream_t> _streams;
  mutable cublasHandle_t _cublas_handle;
  mutable bool _cublasInitialized;
  mutable cusolverDnHandle_t _cusolverDn_handle;
  mutable bool _cusolverDnInitialized;
  mutable cusolverSpHandle_t _cusolverSp_handle;
  mutable bool _cusolverSpInitialized;
  mutable cusparseHandle_t _cusparse_handle;
  mutable bool _cusparseInitialized;
  std::shared_ptr<deviceAllocator> _deviceAllocator;
  std::shared_ptr<hostAllocator> _hostAllocator;
  cudaStream_t _userStream;
  cudaEvent_t _event;
  mutable cudaDeviceProp _prop;
  mutable bool _devicePropInitialized;

  ///@todo: enable this once we have migrated cuml-comms
  //std::shared_ptr<MLCommon::cumlCommunicator> _communicator;

  void createResources() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    _streams.push_back(stream);
    for (int i = 1; i < _num_streams; ++i) {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      _streams.push_back(stream);
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&_event, cudaEventDisableTiming));
  }

  void destroyResources() {
    ///@todo: enable *_NO_THROW variants once we have enabled logging
    if (_cusparseInitialized) {
      //CUSPARSE_CHECK_NO_THROW(cusparseDestroy(_cusparse_handle));
      CUSPARSE_CHECK(cusparseDestroy(_cusparse_handle));
    }
    if (_cusolverDnInitialized) {
      //CUSOLVER_CHECK_NO_THROW(cusolverDnDestroy(_cusolverDn_handle));
    }
    if (_cusolverSpInitialized) {
      //CUSOLVER_CHECK_NO_THROW(cusolverSpDestroy(_cusolverSp_handle));
      CUSOLVER_CHECK(cusolverSpDestroy(_cusolverSp_handle));
    }
    if (_cublasInitialized) {
      //CUBLAS_CHECK_NO_THROW(cublasDestroy(_cublas_handle));
      CUBLAS_CHECK(cublasDestroy(_cublas_handle));
    }
    while (!_streams.empty()) {
      //CUDA_CHECK_NO_THROW(cudaStreamDestroy(_streams.back()));
      CUDA_CHECK(cudaStreamDestroy(_streams.back()));
      _streams.pop_back();
    }
    //CUDA_CHECK_NO_THROW(cudaEventDestroy(_event));
    CUDA_CHECK(cudaEventDestroy(_event));
  }
};  // class handle_t

/**
 * @brief RAII approach to synchronizing across all streams in the handle
 */
class streamSyncer {
 public:
  streamSyncer(const handle_t& handle) : _handle(handle) {
    _handle.waitOnUserStream();
  }
  ~streamSyncer() { _handle.waitOnInternalStreams(); }

  streamSyncer(const streamSyncer& other) = delete;
  streamSyncer& operator=(const streamSyncer& other) = delete;

 private:
  const handle_t& _handle;
};  // class streamSyncer

}  // end namespace ML
