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

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>

///@todo: enable once we have migrated cuml-comms layer too
//#include <common/cuml_comms_int.hpp>

#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/cusolver_wrappers.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/comms/comms.hpp>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/host/allocator.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include "cudart_utils.h"

namespace raft {

/**
 * @brief Main handle object that stores all necessary context used for calling
 *        necessary cuda kernels and/or libraries
 */
class handle_t {
 private:
  static constexpr int kNumDefaultWorkerStreams = 0;

 public:
  // delete copy/move constructors and assignment operators as
  // copying and moving underlying resources is unsafe
  handle_t(const handle_t&) = delete;
  handle_t& operator=(const handle_t&) = delete;
  handle_t(handle_t&&) = delete;
  handle_t& operator=(handle_t&&) = delete;

  /**
   * @brief Construct a handle with a stream view and stream pool
   *
   * @param[in] stream the default stream (which has the default per-thread stream if unspecified)
   * @param[in] stream_pool the stream pool used (which has default pool of size 0 if unspecified)
   */
  handle_t(rmm::cuda_stream_view stream = rmm::cuda_stream_per_thread,
           const rmm::cuda_stream_pool& stream_pool = rmm::cuda_stream_pool{0})
    : dev_id_([]() -> int {
        int cur_dev = -1;
        CUDA_CHECK(cudaGetDevice(&cur_dev));
        return cur_dev;
      }()),
      device_allocator_(std::make_shared<mr::device::default_allocator>()),
      host_allocator_(std::make_shared<mr::host::default_allocator>()),
      stream_view_(stream),
      stream_pool_(stream_pool) {
    create_resources();
  }

  /** Destroys all held-up resources */
  virtual ~handle_t() { destroy_resources(); }

  int get_device() const { return dev_id_; }

  /**
   * @brief returns main stream on the handle
   */
  const rmm::cuda_stream_view& get_stream() const { return stream_view_; }

  /**
   * @brief returns stream pool on the handle, could be 0 sized
   */
  const rmm::cuda_stream_pool& get_stream_pool() const { return stream_pool_; }

  void set_device_allocator(std::shared_ptr<mr::device::allocator> allocator) {
    device_allocator_ = allocator;
  }
  std::shared_ptr<mr::device::allocator> get_device_allocator() const {
    return device_allocator_;
  }

  void set_host_allocator(std::shared_ptr<mr::host::allocator> allocator) {
    host_allocator_ = allocator;
  }
  std::shared_ptr<mr::host::allocator> get_host_allocator() const {
    return host_allocator_;
  }

  cublasHandle_t get_cublas_handle() const {
    std::lock_guard<std::mutex> _(mutex_);
    if (!cublas_initialized_) {
      CUBLAS_CHECK(cublasCreate(&cublas_handle_));
      CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_view_));
      cublas_initialized_ = true;
    }
    return cublas_handle_;
  }

  cusolverDnHandle_t get_cusolver_dn_handle() const {
    std::lock_guard<std::mutex> _(mutex_);
    if (!cusolver_dn_initialized_) {
      CUSOLVER_CHECK(cusolverDnCreate(&cusolver_dn_handle_));
      CUSOLVER_CHECK(cusolverDnSetStream(cusolver_dn_handle_, stream_view_));
      cusolver_dn_initialized_ = true;
    }
    return cusolver_dn_handle_;
  }

  cusolverSpHandle_t get_cusolver_sp_handle() const {
    std::lock_guard<std::mutex> _(mutex_);
    if (!cusolver_sp_initialized_) {
      CUSOLVER_CHECK(cusolverSpCreate(&cusolver_sp_handle_));
      CUSOLVER_CHECK(cusolverSpSetStream(cusolver_sp_handle_, stream_view_));
      cusolver_sp_initialized_ = true;
    }
    return cusolver_sp_handle_;
  }

  cusparseHandle_t get_cusparse_handle() const {
    std::lock_guard<std::mutex> _(mutex_);
    if (!cusparse_initialized_) {
      CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));
      CUSPARSE_CHECK(cusparseSetStream(cusparse_handle_, stream_view_));
      cusparse_initialized_ = true;
    }
    return cusparse_handle_;
  }

  /**
   * @brief synchronize main stream on the handle
   */
  void sync_stream() const { stream_view_.synchronize(); }

  /**
   * @brief synchronize the stream pool on the handle
   */
  void sync_stream_pool() const {
    for (std::size_t i = 0; i < stream_pool_.get_pool_size(); i++) {
      stream_pool_.get_stream(i).synchronize();
    }
  }

  /**
   * @brief synchronize subset of stream pool
   * 
   * @param[in] stream_indices the indices of the streams in the stream pool to synchronize
   */
  void sync_stream_pool(const std::vector<std::size_t> stream_indices) const {
    for (const auto& stream_index : stream_indices) {
      stream_pool_.get_stream(stream_index).synchronize();
    }
  }

  /**
   * @brief ask stream pool to wait on last event in main stream
   */
  void wait_stream_pool_on_stream() const {
    CUDA_CHECK(cudaEventRecord(event_, stream_view_));
    for (std::size_t i = 0; i < stream_pool_.get_pool_size(); i++) {
      CUDA_CHECK(cudaStreamWaitEvent(stream_pool_.get_stream(i), event_, 0));
    }
  }

  void set_comms(std::shared_ptr<comms::comms_t> communicator) {
    communicator_ = communicator;
  }

  const comms::comms_t& get_comms() const {
    RAFT_EXPECTS(this->comms_initialized(),
                 "ERROR: Communicator was not initialized\n");
    return *communicator_;
  }

  void set_subcomm(std::string key, std::shared_ptr<comms::comms_t> subcomm) {
    subcomms_[key] = subcomm;
  }

  const comms::comms_t& get_subcomm(std::string key) const {
    RAFT_EXPECTS(subcomms_.find(key) != subcomms_.end(),
                 "%s was not found in subcommunicators.", key.c_str());

    auto subcomm = subcomms_.at(key);

    RAFT_EXPECTS(nullptr != subcomm.get(),
                 "ERROR: Subcommunicator was not initialized");

    return *subcomm;
  }

  bool comms_initialized() const { return (nullptr != communicator_.get()); }

  const cudaDeviceProp& get_device_properties() const {
    std::lock_guard<std::mutex> _(mutex_);
    if (!device_prop_initialized_) {
      CUDA_CHECK(cudaGetDeviceProperties(&prop_, dev_id_));
      device_prop_initialized_ = true;
    }
    return prop_;
  }

 private:
  std::shared_ptr<comms::comms_t> communicator_;
  std::unordered_map<std::string, std::shared_ptr<comms::comms_t>> subcomms_;

  const int dev_id_;
  mutable cublasHandle_t cublas_handle_;
  mutable bool cublas_initialized_{false};
  mutable cusolverDnHandle_t cusolver_dn_handle_;
  mutable bool cusolver_dn_initialized_{false};
  mutable cusolverSpHandle_t cusolver_sp_handle_;
  mutable bool cusolver_sp_initialized_{false};
  mutable cusparseHandle_t cusparse_handle_;
  mutable bool cusparse_initialized_{false};
  std::shared_ptr<mr::device::allocator> device_allocator_;
  std::shared_ptr<mr::host::allocator> host_allocator_;
  rmm::cuda_stream_view stream_view_;
  const rmm::cuda_stream_pool& stream_pool_;
  cudaEvent_t event_;
  mutable cudaDeviceProp prop_;
  mutable bool device_prop_initialized_{false};
  mutable std::mutex mutex_;

  void create_resources() {
    CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  void destroy_resources() {
    ///@todo: enable *_NO_THROW variants once we have enabled logging
    if (cusparse_initialized_) {
      //CUSPARSE_CHECK_NO_THROW(cusparseDestroy(cusparse_handle_));
      CUSPARSE_CHECK(cusparseDestroy(cusparse_handle_));
    }
    if (cusolver_dn_initialized_) {
      //CUSOLVER_CHECK_NO_THROW(cusolverDnDestroy(cusolver_dn_handle_));
      CUSOLVER_CHECK(cusolverDnDestroy(cusolver_dn_handle_));
    }
    if (cusolver_sp_initialized_) {
      //CUSOLVER_CHECK_NO_THROW(cusolverSpDestroy(cusolver_sp_handle_));
      CUSOLVER_CHECK(cusolverSpDestroy(cusolver_sp_handle_));
    }
    if (cublas_initialized_) {
      //CUBLAS_CHECK_NO_THROW(cublasDestroy(cublas_handle_));
      CUBLAS_CHECK(cublasDestroy(cublas_handle_));
    }
    //CUDA_CHECK_NO_THROW(cudaEventDestroy(event_));
    CUDA_CHECK(cudaEventDestroy(event_));
  }
};  // class handle_t

/**
 * @brief RAII approach to synchronizing across all streams in the handle
 */
class stream_syncer {
 public:
  explicit stream_syncer(const handle_t& handle) : handle_(handle) {
    handle_.sync_stream();
  }
  ~stream_syncer() {
    handle_.wait_stream_pool_on_stream();
    handle_.sync_stream_pool();
  }

  stream_syncer(const stream_syncer& other) = delete;
  stream_syncer& operator=(const stream_syncer& other) = delete;

 private:
  const handle_t& handle_;
};  // class stream_syncer

}  // namespace raft
