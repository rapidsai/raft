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
#include <raft/util/cudart_utils.hpp>
#include <raft/core/host_container_policy.hpp>
#include <variant>
#ifndef RAFT_DISABLE_GPU
#include <raft/core/device_container_policy.hpp>
#endif

namespace raft {
#ifdef RAFT_DISABLE_GPU
template <typename T>
using buffer_container_policy = std::variant<raft::host_vector_policy<T>>;
#else
template <typename T>
using buffer_container_policy = std::variant<raft::host_vector_policy<T>, raft::device_uvector_policy<T>>;
#endif
}