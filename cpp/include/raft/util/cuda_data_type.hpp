/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuda_fp16.hpp>

#include <library_types.h>

#include <cstdint>

namespace raft {

template <typename T>
constexpr auto get_cuda_data_type() -> cudaDataType_t;

template <>
inline constexpr auto get_cuda_data_type<int8_t>() -> cudaDataType_t
{
  return CUDA_R_8I;
}
template <>
inline constexpr auto get_cuda_data_type<uint8_t>() -> cudaDataType_t
{
  return CUDA_R_8U;
}
template <>
inline constexpr auto get_cuda_data_type<int16_t>() -> cudaDataType_t
{
  return CUDA_R_16I;
}
template <>
inline constexpr auto get_cuda_data_type<uint16_t>() -> cudaDataType_t
{
  return CUDA_R_16U;
}
template <>
inline constexpr auto get_cuda_data_type<int32_t>() -> cudaDataType_t
{
  return CUDA_R_32I;
}
template <>
inline constexpr auto get_cuda_data_type<uint32_t>() -> cudaDataType_t
{
  return CUDA_R_32U;
}
template <>
inline constexpr auto get_cuda_data_type<int64_t>() -> cudaDataType_t
{
  return CUDA_R_64I;
}
template <>
inline constexpr auto get_cuda_data_type<uint64_t>() -> cudaDataType_t
{
  return CUDA_R_64U;
}
template <>
inline constexpr auto get_cuda_data_type<half>() -> cudaDataType_t
{
  return CUDA_R_16F;
}
template <>
inline constexpr auto get_cuda_data_type<float>() -> cudaDataType_t
{
  return CUDA_R_32F;
}
template <>
inline constexpr auto get_cuda_data_type<double>() -> cudaDataType_t
{
  return CUDA_R_64F;
}
}  // namespace raft
