/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
