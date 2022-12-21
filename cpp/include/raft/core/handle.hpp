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

#include <raft/core/cudart_utils.hpp>

#include <raft/core/comms.hpp>
#include <raft/core/cublas_macros.hpp>
#include <raft/core/cusolver_macros.hpp>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/interruptible.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/exec_policy.hpp>

#include <raft/core/resource/base_handle.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_event.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resource/cusolver_sp_handle.hpp>
#include <raft/core/resource/cusparse_handle.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resource/sub_comms.hpp>
#include <raft/core/resource/thrust_policy.hpp>

namespace raft {

/**
 * @brief Main handle object that stores all necessary context used for calling
 *        necessary cuda kernels and/or libraries
 */
class device_handle_t : public core::base_handle_t {
 public:
  // delete copy/move constructors and assignment operators as
  // copying and moving underlying resources is unsafe
  device_handle_t(const device_handle_t&) = delete;
  device_handle_t& operator=(const device_handle_t&) = delete;
  device_handle_t(device_handle_t&&)                 = delete;
  device_handle_t& operator=(device_handle_t&&) = delete;

  /**
   * @brief Construct a handle with a stream view and stream pool
   *
   * @param[in] stream_view the default stream (which has the default per-thread stream if
   * unspecified)
   * @param[in] stream_pool the stream pool used (which has default of nullptr if unspecified)
   */

  device_handle_t(rmm::cuda_stream_view stream_view                  = rmm::cuda_stream_per_thread,
                  std::shared_ptr<rmm::cuda_stream_pool> stream_pool = {nullptr})
    : core::base_handle_t{}
  {
    core::base_handle_t::add_resource_factory(
      std::make_shared<core::device_id_resource_factory_t>());
    core::base_handle_t::add_resource_factory(
      std::make_shared<core::cuda_stream_resource_factory_t>(stream_view));
    core::base_handle_t::add_resource_factory(
      std::make_shared<core::cuda_stream_pool_resource_factory_t>(stream_pool));
  }

  /** Destroys all held-up resources */
  virtual ~device_handle_t() {}

  int get_device() const { return core::get_device_id(*this); }

  cublasHandle_t get_cublas_handle() const { return core::get_cublas_handle(*this); }

  cusolverDnHandle_t get_cusolver_dn_handle() const { return core::get_cusolver_dn_handle(*this); }

  cusolverSpHandle_t get_cusolver_sp_handle() const { return core::get_cusolver_sp_handle(*this); }

  cusparseHandle_t get_cusparse_handle() const { return core::get_cusparse_handle(*this); }

  rmm::exec_policy& get_thrust_policy() const { return core::get_thrust_policy(*this); }

  /**
   * @brief synchronize a stream on the handle
   */
  void sync_stream(rmm::cuda_stream_view stream) const { core::sync_stream(*this, stream); }

  /**
   * @brief synchronize main stream on the handle
   */
  void sync_stream() const { core::sync_stream(*this); }

  /**
   * @brief returns main stream on the handle
   */
  rmm::cuda_stream_view get_stream() const { return core::get_cuda_stream(*this); }

  /**
   * @brief returns whether stream pool was initialized on the handle
   */

  bool is_stream_pool_initialized() const { return core::is_stream_pool_initialized(*this); }

  /**
   * @brief returns stream pool on the handle
   */
  const rmm::cuda_stream_pool& get_stream_pool() const { return core::get_cuda_stream_pool(*this); }

  std::size_t get_stream_pool_size() const { return core::get_stream_pool_size(*this); }

  /**
   * @brief return stream from pool
   */
  rmm::cuda_stream_view get_stream_from_stream_pool() const
  {
    return core::get_stream_from_stream_pool(*this);
  }

  /**
   * @brief return stream from pool at index
   */
  rmm::cuda_stream_view get_stream_from_stream_pool(std::size_t stream_idx) const
  {
    return core::get_stream_from_stream_pool(*this, stream_idx);
  }

  /**
   * @brief return stream from pool if size > 0, else main stream on handle
   */
  rmm::cuda_stream_view get_next_usable_stream() const
  {
    return core::get_next_usable_stream(*this);
  }

  /**
   * @brief return stream from pool at index if size > 0, else main stream on handle
   *
   * @param[in] stream_idx the required index of the stream in the stream pool if available
   */
  rmm::cuda_stream_view get_next_usable_stream(std::size_t stream_idx) const
  {
    return core::get_next_usable_stream(*this, stream_idx);
  }

  /**
   * @brief synchronize the stream pool on the handle
   */
  void sync_stream_pool() const { return core::sync_stream_pool(*this); }

  /**
   * @brief synchronize subset of stream pool
   *
   * @param[in] stream_indices the indices of the streams in the stream pool to synchronize
   */
  void sync_stream_pool(const std::vector<std::size_t> stream_indices) const
  {
    return core::sync_stream_pool(*this, stream_indices);
  }

  /**
   * @brief ask stream pool to wait on last event in main stream
   */
  void wait_stream_pool_on_stream() const { return core::wait_stream_pool_on_stream(*this); }

  void set_comms(std::shared_ptr<comms::comms_t> communicator)
  {
    core::set_comms(*this, communicator);
  }

  const comms::comms_t& get_comms() const { return core::get_comms(*this); }

  void set_subcomm(std::string key, std::shared_ptr<comms::comms_t> subcomm)
  {
    core::set_subcomm(*this, key, subcomm);
  }

  const comms::comms_t& get_subcomm(std::string key) const { return core::get_subcomm(*this, key); }

  bool comms_initialized() const { return core::comms_initialized(*this); }

  const cudaDeviceProp& get_device_properties() const { return core::get_device_properties(*this); }
};  // class device_handle_t

using handle_t = device_handle_t;

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