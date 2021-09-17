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
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/exec_policy.hpp>
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
  /**
   * @brief Construct a handle with the specified number of worker streams
   *
   * @param[in] n_streams number worker streams to be created
   */
  explicit handle_t(int n_streams = kNumDefaultWorkerStreams)
    : dev_id_([]() -> int {
        int cur_dev = -1;
        CUDA_CHECK(cudaGetDevice(&cur_dev));
        return cur_dev;
      }()),
      streams_{[n_streams]() {
        if (n_streams == 0) {
          return std::nullptr_t;
        } else {
          return std::make_unique<rmm::cuda_stream_pool>(n_streams);
        }
      }()} {
    create_resources();
    thrust_policy_ = std::make_unique<rmm::exec_policy>(user_stream_);
  }

  /**
   * @brief Construct a light handle copy from another 
   * user stream, cuda handles, comms and worker pool are not copied
   * The user_stream of the returned handle is set to the specified stream 
   * of the other handle worker pool 
   * @param[in] stream_id stream id in `other` worker streams 
   * to be set as user stream in the constructed handle
   * @param[in] n_streams number worker streams to be created
   */
  handle_t(const handle_t& other, int stream_id,
           int n_streams = kNumDefaultWorkerStreams)
    : dev_id_(other.get_device()), streams_(n_streams) {
    RAFT_EXPECTS(
      other.get_num_internal_streams() > 0,
      "ERROR: the main handle must have at least one worker stream\n");
    prop_ = other.get_device_properties();
    device_prop_initialized_ = true;
    create_resources();
    set_stream(other.get_internal_stream(stream_id));
    thrust_policy_ = std::make_unique<rmm::exec_policy>(user_stream_);
  }

  /** Destroys all held-up resources */
  virtual ~handle_t() { destroy_resources(); }

  int get_device() const { return dev_id_; }

  void set_stream(cudaStream_t stream) { user_stream_ = stream; }
  cudaStream_t get_stream() const { return user_stream_; }
  rmm::cuda_stream_view get_stream_view() const {
    return rmm::cuda_stream_view(user_stream_);
  }

  cublasHandle_t get_cublas_handle() const {
    std::lock_guard<std::mutex> _(mutex_);
    if (!cublas_initialized_) {
      CUBLAS_CHECK(cublasCreate(&cublas_handle_));
      cublas_initialized_ = true;
    }
    return cublas_handle_;
  }

  cusolverDnHandle_t get_cusolver_dn_handle() const {
    std::lock_guard<std::mutex> _(mutex_);
    if (!cusolver_dn_initialized_) {
      CUSOLVER_CHECK(cusolverDnCreate(&cusolver_dn_handle_));
      cusolver_dn_initialized_ = true;
    }
    return cusolver_dn_handle_;
  }

  cusolverSpHandle_t get_cusolver_sp_handle() const {
    std::lock_guard<std::mutex> _(mutex_);
    if (!cusolver_sp_initialized_) {
      CUSOLVER_CHECK(cusolverSpCreate(&cusolver_sp_handle_));
      cusolver_sp_initialized_ = true;
    }
    return cusolver_sp_handle_;
  }

  cusparseHandle_t get_cusparse_handle() const {
    std::lock_guard<std::mutex> _(mutex_);
    if (!cusparse_initialized_) {
      CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));
      cusparse_initialized_ = true;
    }
    return cusparse_handle_;
  }

  rmm::exec_policy& get_thrust_policy() const { return *thrust_policy_; }

  // legacy compatibility for cuML
  cudaStream_t get_internal_stream(int sid) const {
    RAFT_EXPECTS(
      streams_.get() != nullptr,
      "ERROR: rmm::cuda_stream_pool was not initialized with a non-zero value");
    return streams_->get_stream(sid).value();
  }
  // new accessor return rmm::cuda_stream_view
  rmm::cuda_stream_view get_internal_stream_view(int sid) const {
    RAFT_EXPECTS(
      streams_.get() != nullptr,
      "ERROR: rmm::cuda_stream_pool was not initialized with a non-zero value");
    return streams_->get_stream(sid);
  }

  int get_num_internal_streams() const { return streams_.get_pool_size(); }
  std::vector<cudaStream_t> get_internal_streams() const {
    std::vector<cudaStream_t> int_streams_vec;
    for (int i = 0; i < get_num_internal_streams(); i++) {
      int_streams_vec.push_back(get_internal_stream(i));
    }
    return int_streams_vec;
  }

  void wait_on_user_stream() const {
    CUDA_CHECK(cudaEventRecord(event_, user_stream_));
    for (int i = 0; i < get_num_internal_streams(); i++) {
      CUDA_CHECK(cudaStreamWaitEvent(get_internal_stream(i), event_, 0));
    }
  }

  void wait_on_internal_streams() const {
    for (int i = 0; i < get_num_internal_streams(); i++) {
      CUDA_CHECK(cudaEventRecord(event_, get_internal_stream(i)));
      CUDA_CHECK(cudaStreamWaitEvent(user_stream_, event_, 0));
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
  std::unique_ptr<rmm::cuda_stream_pool> streams_;
  mutable cublasHandle_t cublas_handle_;
  mutable bool cublas_initialized_{false};
  mutable cusolverDnHandle_t cusolver_dn_handle_;
  mutable bool cusolver_dn_initialized_{false};
  mutable cusolverSpHandle_t cusolver_sp_handle_;
  mutable bool cusolver_sp_initialized_{false};
  mutable cusparseHandle_t cusparse_handle_;
  mutable bool cusparse_initialized_{false};
  std::unique_ptr<rmm::exec_policy> thrust_policy_{nullptr};
  cudaStream_t user_stream_{nullptr};
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
    handle_.wait_on_user_stream();
  }
  ~stream_syncer() { handle_.wait_on_internal_streams(); }

  stream_syncer(const stream_syncer& other) = delete;
  stream_syncer& operator=(const stream_syncer& other) = delete;

 private:
  const handle_t& handle_;
};  // class stream_syncer

}  // namespace raft
