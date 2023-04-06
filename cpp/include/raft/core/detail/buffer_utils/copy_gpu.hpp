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
#include "raft/util/cuda_rt_essentials.hpp"
#include "raft/util/cudart_utils.hpp"
#include <cuda_runtime_api.h>
#include <raft/core/device_type.hpp>
#include <raft/core/execution_stream.hpp>
#include <raft/core/device_support.hpp>

#include <thrust/copy.h>
#include <type_traits>

namespace raft {
namespace detail {

template<device_type dst_type, device_type src_type, typename T>
std::enable_if_t<std::conjunction_v<std::disjunction<std::bool_constant<dst_type == device_type::gpu>, std::bool_constant<src_type == device_type::gpu>>, std::bool_constant<GPU_ENABLED>>, void> copy(T* dst, T const* src, uint32_t size, raft::execution_stream stream) {
  RAFT_CUDA_TRY(thrust::copy(rmm::exec_policy(stream), src, src + size, dst));
}

} // namespace detail
} // namespace raft