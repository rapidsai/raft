/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#ifndef __RAFT_RT_HANDLE
#define __RAFT_RT_HANDLE

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

#include <raft/util/cudart_utils.hpp>

#include <raft/core/comms.hpp>
#include <raft/core/cublas_macros.hpp>
#include <raft/core/cusolver_macros.hpp>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/interruptible.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/exec_policy.hpp>

namespace raft {

/**
 * @brief Main handle object that stores all necessary context used for calling
 *        necessary cuda kernels and/or libraries
 */
class handle_t {
 public:
  // delete copy/move constructors and assignment operators as
  // copying and moving underlying resources is unsafe
  handle_t(const handle_t&) = delete;
  handle_t& operator=(const handle_t&) = delete;
  handle_t(handle_t&&)                 = delete;
  handle_t& operator=(handle_t&&) = delete;

  /**
   * @brief Construct a handle with a stream view and stream pool
   *
   * @param[in] stream_view the default stream (which has the default per-thread stream if
   * unspecified)
   * @param[in] stream_pool the stream pool used (which has default of nullptr if unspecified)
   */
  handle_t(rmm::cuda_stream_view stream_view                  = rmm::cuda_stream_per_thread,
           std::shared_ptr<rmm::cuda_stream_pool> stream_pool = {nullptr})
    : dev_id_([]() -> int {
        int cur_dev = -1;
        RAFT_CUDA_TRY(cudaGetDevice(&cur_dev));
        return cur_dev;
      }()),
      stream_view_{stream_view},
      stream_pool_{stream_pool}
  {
    create_resources();
  }

  /** Destroys all held-up resources */
  virtual ~handle_t() { destroy_resources(); }

  int get_device() const { return dev_id_; }

  cublasHandle_t get_cublas_handle() const
  {
    std::lock_guard<std::mutex> _(mutex_);
    if (!cublas_initialized_) {
      RAFT_CUBLAS_TRY_NO_THROW(cublasCreate(&cublas_handle_));
      RAFT_CUBLAS_TRY_NO_THROW(cublasSetStream(cublas_handle_, stream_view_));
      cublas_initialized_ = true;
    }
    return cublas_handle_;
  }

  cusolverDnHandle_t get_cusolver_dn_handle() const
  {
    std::lock_guard<std::mutex> _(mutex_);
    if (!cusolver_dn_initialized_) {
      RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnCreate(&cusolver_dn_handle_));
      RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnSetStream(cusolver_dn_handle_, stream_view_));
      cusolver_dn_initialized_ = true;
    }
    return cusolver_dn_handle_;
  }

  cusolverSpHandle_t get_cusolver_sp_handle() const
  {
    std::lock_guard<std::mutex> _(mutex_);
    if (!cusolver_sp_initialized_) {
      RAFT_CUSOLVER_TRY_NO_THROW(cusolverSpCreate(&cusolver_sp_handle_));
      RAFT_CUSOLVER_TRY_NO_THROW(cusolverSpSetStream(cusolver_sp_handle_, stream_view_));
      cusolver_sp_initialized_ = true;
    }
    return cusolver_sp_handle_;
  }

  cusparseHandle_t get_cusparse_handle() const
  {
    std::lock_guard<std::mutex> _(mutex_);
    if (!cusparse_initialized_) {
      RAFT_CUSPARSE_TRY_NO_THROW(cusparseCreate(&cusparse_handle_));
      RAFT_CUSPARSE_TRY_NO_THROW(cusparseSetStream(cusparse_handle_, stream_view_));
      cusparse_initialized_ = true;
    }
    return cusparse_handle_;
  }

  rmm::exec_policy& get_thrust_policy() const { return *thrust_policy_; }

  /**
   * @brief synchronize a stream on the handle
   */
  void sync_stream(rmm::cuda_stream_view stream) const { interruptible::synchronize(stream); }

  /**
   * @brief synchronize main stream on the handle
   */
  void sync_stream() const { sync_stream(stream_view_); }

  /**
   * @brief returns main stream on the handle
   */
  rmm::cuda_stream_view get_stream() const { return stream_view_; }

  /**
   * @brief returns whether stream pool was initialized on the handle
   */

  bool is_stream_pool_initialized() const { return stream_pool_.get() != nullptr; }

  /**
   * @brief returns stream pool on the handle
   */
  const rmm::cuda_stream_pool& get_stream_pool() const
  {
    RAFT_EXPECTS(stream_pool_, "ERROR: rmm::cuda_stream_pool was not initialized");
    return *stream_pool_;
  }

  std::size_t get_stream_pool_size() const
  {
    return is_stream_pool_initialized() ? stream_pool_->get_pool_size() : 0;
  }

  /**
   * @brief return stream from pool
   */
  rmm::cuda_stream_view get_stream_from_stream_pool() const
  {
    RAFT_EXPECTS(stream_pool_, "ERROR: rmm::cuda_stream_pool was not initialized");
    return stream_pool_->get_stream();
  }

  /**
   * @brief return stream from pool at index
   */
  rmm::cuda_stream_view get_stream_from_stream_pool(std::size_t stream_idx) const
  {
    RAFT_EXPECTS(stream_pool_, "ERROR: rmm::cuda_stream_pool was not initialized");
    return stream_pool_->get_stream(stream_idx);
  }

  /**
   * @brief return stream from pool if size > 0, else main stream on handle
   */
  rmm::cuda_stream_view get_next_usable_stream() const
  {
    return is_stream_pool_initialized() ? get_stream_from_stream_pool() : stream_view_;
  }

  /**
   * @brief return stream from pool at index if size > 0, else main stream on handle
   *
   * @param[in] stream_idx the required index of the stream in the stream pool if available
   */
  rmm::cuda_stream_view get_next_usable_stream(std::size_t stream_idx) const
  {
    return is_stream_pool_initialized() ? get_stream_from_stream_pool(stream_idx) : stream_view_;
  }

  /**
   * @brief synchronize the stream pool on the handle
   */
  void sync_stream_pool() const
  {
    for (std::size_t i = 0; i < get_stream_pool_size(); i++) {
      sync_stream(stream_pool_->get_stream(i));
    }
  }

  /**
   * @brief synchronize subset of stream pool
   *
   * @param[in] stream_indices the indices of the streams in the stream pool to synchronize
   */
  void sync_stream_pool(const std::vector<std::size_t> stream_indices) const
  {
    RAFT_EXPECTS(stream_pool_, "ERROR: rmm::cuda_stream_pool was not initialized");
    for (const auto& stream_index : stream_indices) {
      sync_stream(stream_pool_->get_stream(stream_index));
    }
  }

  /**
   * @brief ask stream pool to wait on last event in main stream
   */
  void wait_stream_pool_on_stream() const
  {
    RAFT_CUDA_TRY(cudaEventRecord(event_, stream_view_));
    for (std::size_t i = 0; i < get_stream_pool_size(); i++) {
      RAFT_CUDA_TRY(cudaStreamWaitEvent(stream_pool_->get_stream(i), event_, 0));
    }
  }

  void set_comms(std::shared_ptr<comms::comms_t> communicator) { communicator_ = communicator; }

  const comms::comms_t& get_comms() const
  {
    RAFT_EXPECTS(this->comms_initialized(), "ERROR: Communicator was not initialized\n");
    return *communicator_;
  }

  void set_subcomm(std::string key, std::shared_ptr<comms::comms_t> subcomm)
  {
    subcomms_[key] = subcomm;
  }

  const comms::comms_t& get_subcomm(std::string key) const
  {
    RAFT_EXPECTS(
      subcomms_.find(key) != subcomms_.end(), "%s was not found in subcommunicators.", key.c_str());

    auto subcomm = subcomms_.at(key);

    RAFT_EXPECTS(nullptr != subcomm.get(), "ERROR: Subcommunicator was not initialized");

    return *subcomm;
  }

  bool comms_initialized() const { return (nullptr != communicator_.get()); }

  const cudaDeviceProp& get_device_properties() const
  {
    std::lock_guard<std::mutex> _(mutex_);
    if (!device_prop_initialized_) {
      RAFT_CUDA_TRY_NO_THROW(cudaGetDeviceProperties(&prop_, dev_id_));
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
  std::unique_ptr<rmm::exec_policy> thrust_policy_{nullptr};
  rmm::cuda_stream_view stream_view_{rmm::cuda_stream_per_thread};
  std::shared_ptr<rmm::cuda_stream_pool> stream_pool_{nullptr};
  cudaEvent_t event_;
  mutable cudaDeviceProp prop_;
  mutable bool device_prop_initialized_{false};
  mutable std::mutex mutex_;

  void create_resources()
  {
    thrust_policy_ = std::make_unique<rmm::exec_policy>(stream_view_);

    RAFT_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  void destroy_resources()
  {
    if (cusparse_initialized_) { RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroy(cusparse_handle_)); }
    if (cusolver_dn_initialized_) {
      RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnDestroy(cusolver_dn_handle_));
    }
    if (cusolver_sp_initialized_) {
      RAFT_CUSOLVER_TRY_NO_THROW(cusolverSpDestroy(cusolver_sp_handle_));
    }
    if (cublas_initialized_) { RAFT_CUBLAS_TRY_NO_THROW(cublasDestroy(cublas_handle_)); }
    RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(event_));
  }
};  // class handle_t

/**
 * @brief RAII approach to synchronizing across all streams in the handle
 */
class stream_syncer {
 public:
  explicit stream_syncer(const handle_t& handle) : handle_(handle) { handle_.sync_stream(); }
  ~stream_syncer()
  {
    handle_.wait_stream_pool_on_stream();
    handle_.sync_stream_pool();
  }

  stream_syncer(const stream_syncer& other) = delete;
  stream_syncer& operator=(const stream_syncer& other) = delete;

 private:
  const handle_t& handle_;
};  // class stream_syncer

}  // namespace raft

#endif