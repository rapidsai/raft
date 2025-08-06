/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cuda_fp16.h>

#include <type_traits>

namespace raft {

/**
 * Extension of std::is_floating_point to support CUDA types like __half.
 */
template <typename T>
struct is_floating_point {
  static constexpr bool value =
    std::is_floating_point<T>::value || std::is_same<typename std::remove_cv<T>::type, half>::value;
};

template <typename T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;
}  // namespace raft
