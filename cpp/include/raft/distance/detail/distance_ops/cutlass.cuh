/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <type_traits>  // std::false_type
#include <utility>      // std::declval

namespace raft::distance::detail::ops {

// This file defines the named requirement "has_cutlass_op" that can be used to
// determine if a distance operation has a CUTLASS op that can be used to pass
// to CUTLASS. Examples of distance operations that satisfy this requirement are
// cosine_distance_op and l2_exp_distance_op.

// Primary template handles types that do not support CUTLASS.
// This pattern is described in:
// https://en.cppreference.com/w/cpp/types/void_t
template <typename, typename = void>
struct has_cutlass_op : std::false_type {};

// Specialization recognizes types that do support CUTLASS
template <typename T>
struct has_cutlass_op<T, std::void_t<decltype(std::declval<T>().get_cutlass_op())>>
  : std::true_type {};

}  // namespace raft::distance::detail::ops
