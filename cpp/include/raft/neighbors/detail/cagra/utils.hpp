/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cfloat>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <type_traits>

namespace raft::neighbors::cagra::detail {
namespace utils {
template <class DATA_T>
inline cudaDataType_t get_cuda_data_type();
template <>
inline cudaDataType_t get_cuda_data_type<float>()
{
  return CUDA_R_32F;
}
template <>
inline cudaDataType_t get_cuda_data_type<half>()
{
  return CUDA_R_16F;
}
template <>
inline cudaDataType_t get_cuda_data_type<int8_t>()
{
  return CUDA_R_8I;
}
template <>
inline cudaDataType_t get_cuda_data_type<uint8_t>()
{
  return CUDA_R_8U;
}
template <>
inline cudaDataType_t get_cuda_data_type<uint32_t>()
{
  return CUDA_R_32U;
}
template <>
inline cudaDataType_t get_cuda_data_type<uint64_t>()
{
  return CUDA_R_64U;
}

template <class T>
constexpr unsigned size_of();
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::int8_t>()
{
  return 1;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::uint8_t>()
{
  return 1;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::uint16_t>()
{
  return 2;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::uint32_t>()
{
  return 4;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::uint64_t>()
{
  return 8;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<uint4>()
{
  return 16;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<ulonglong4>()
{
  return 32;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<float>()
{
  return 4;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<half>()
{
  return 2;
}

// max values for data types
template <class BS_T, class FP_T>
union fp_conv {
  BS_T bs;
  FP_T fp;
};
template <class T>
_RAFT_HOST_DEVICE inline T get_max_value();
template <>
_RAFT_HOST_DEVICE inline float get_max_value<float>()
{
  return FLT_MAX;
};
template <>
_RAFT_HOST_DEVICE inline half get_max_value<half>()
{
  return fp_conv<std::uint16_t, half>{.bs = 0x7aff}.fp;
};
template <>
_RAFT_HOST_DEVICE inline std::uint32_t get_max_value<std::uint32_t>()
{
  return 0xffffffffu;
};
template <>
_RAFT_HOST_DEVICE inline std::uint64_t get_max_value<std::uint64_t>()
{
  return 0xfffffffffffffffflu;
};

template <int A, int B, class = void>
struct constexpr_max {
  static const int value = A;
};

template <int A, int B>
struct constexpr_max<A, B, std::enable_if_t<(B > A), bool>> {
  static const int value = B;
};

template <class IdxT>
struct gen_index_msb_1_mask {
  static constexpr IdxT value = static_cast<IdxT>(1) << (utils::size_of<IdxT>() * 8 - 1);
};
}  // namespace utils

/**
 * Utility to sync memory from a host_matrix_view to a device_matrix_view
 *
 * In certain situations (UVM/HMM/ATS) host memory might be directly accessible on the
 * device, and no extra allocations need to be performed. This class checks
 * if the host_matrix_view is already accessible on the device, and only creates device
 * memory and copies over if necessary. In memory limited situations this is preferable
 * to having both a host and device copy
 * TODO: once the mdbuffer changes here https://github.com/wphicks/raft/blob/fea-mdbuffer
 * have been merged, we should remove this class and switch over to using mdbuffer for this
 */
template <typename T, typename IdxT>
class device_matrix_view_from_host {
 public:
  device_matrix_view_from_host(raft::resources const& res, host_matrix_view<T, IdxT> host_view)
    : host_view_(host_view)
  {
    cudaPointerAttributes attr;
    RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, host_view.data_handle()));
    device_ptr = reinterpret_cast<T*>(attr.devicePointer);
    if (device_ptr == NULL) {
      // allocate memory and copy over
      device_mem_.emplace(
        raft::make_device_matrix<T, IdxT>(res, host_view.extent(0), host_view.extent(1)));
      raft::copy(device_mem_->data_handle(),
                 host_view.data_handle(),
                 host_view.extent(0) * host_view.extent(1),
                 resource::get_cuda_stream(res));
      device_ptr = device_mem_->data_handle();
    }
  }

  device_matrix_view<T, IdxT> view()
  {
    return make_device_matrix_view<T, IdxT>(device_ptr, host_view_.extent(0), host_view_.extent(1));
  }

  T* data_handle() { return device_ptr; }

  bool allocated_memory() const { return device_mem_.has_value(); }

 private:
  std::optional<device_matrix<T, IdxT>> device_mem_;
  host_matrix_view<T, IdxT> host_view_;
  T* device_ptr;
};

/**
 * Utility to sync memory from a device_matrix_view to a host_matrix_view
 *
 * In certain situations (UVM/HMM/ATS) device memory might be directly accessible on the
 * host, and no extra allocations need to be performed. This class checks
 * if the device_matrix_view is already accessible on the host, and only creates host
 * memory and copies over if necessary. In memory limited situations this is preferable
 * to having both a host and device copy
 * TODO: once the mdbuffer changes here https://github.com/wphicks/raft/blob/fea-mdbuffer
 * have been merged, we should remove this class and switch over to using mdbuffer for this
 */
template <typename T, typename IdxT>
class host_matrix_view_from_device {
 public:
  host_matrix_view_from_device(raft::resources const& res, device_matrix_view<T, IdxT> device_view)
    : device_view_(device_view)
  {
    cudaPointerAttributes attr;
    RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, device_view.data_handle()));
    host_ptr = reinterpret_cast<T*>(attr.hostPointer);
    if (host_ptr == NULL) {
      // allocate memory and copy over
      host_mem_.emplace(
        raft::make_host_matrix<T, IdxT>(device_view.extent(0), device_view.extent(1)));
      raft::copy(host_mem_->data_handle(),
                 device_view.data_handle(),
                 device_view.extent(0) * device_view.extent(1),
                 resource::get_cuda_stream(res));
      host_ptr = host_mem_->data_handle();
    }
  }

  host_matrix_view<T, IdxT> view()
  {
    return make_host_matrix_view<T, IdxT>(host_ptr, device_view_.extent(0), device_view_.extent(1));
  }

  T* data_handle() { return host_ptr; }

  bool allocated_memory() const { return host_mem_.has_value(); }

 private:
  std::optional<host_matrix<T, IdxT>> host_mem_;
  device_matrix_view<T, IdxT> device_view_;
  T* host_ptr;
};
}  // namespace raft::neighbors::cagra::detail
