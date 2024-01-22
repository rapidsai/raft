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

}  // namespace raft::neighbors::cagra::detail
