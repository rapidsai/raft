/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cmath>

#include <raft/core/detail/macros.hpp>

namespace raft {

/**
 * @defgroup Functors Commonly used functors.
 * The optional index argument is mostly to be used for MainLambda in reduction kernels
 * @{
 */

struct identity_op {
  template <typename Type, typename IdxType = int>
  constexpr RAFT_INLINE_FUNCTION Type operator()(Type in, IdxType i = 0) const
  {
    return in;
  }
};

template <typename OutT>
struct cast_op {
  template <typename InT, typename IdxType = int>
  constexpr RAFT_INLINE_FUNCTION OutT operator()(InT in, IdxType i = 0) const
  {
    return static_cast<OutT>(in);
  }
};

struct key_op {
  template <typename KVP, typename IdxType = int>
  constexpr RAFT_INLINE_FUNCTION typename KVP::Key operator()(const KVP& p, IdxType i = 0) const
  {
    return p.key;
  }
};

struct value_op {
  template <typename KVP, typename IdxType = int>
  constexpr RAFT_INLINE_FUNCTION typename KVP::Value operator()(const KVP& p, IdxType i = 0) const
  {
    return p.value;
  }
};

struct sqrt_op {
  template <typename Type, typename IdxType = int>
  constexpr RAFT_INLINE_FUNCTION Type operator()(Type in, IdxType i = 0) const
  {
    return std::sqrt(in);
  }
};

struct nz_op {
  template <typename Type, typename IdxType = int>
  constexpr RAFT_INLINE_FUNCTION Type operator()(Type in, IdxType i = 0) const
  {
    return in != Type(0) ? Type(1) : Type(0);
  }
};

struct abs_op {
  template <typename Type, typename IdxType = int>
  constexpr RAFT_INLINE_FUNCTION Type operator()(Type in, IdxType i = 0) const
  {
    return std::abs(in);
  }
};

struct sq_op {
  template <typename Type, typename IdxType = int>
  constexpr RAFT_INLINE_FUNCTION Type operator()(Type in, IdxType i = 0) const
  {
    return in * in;
  }
};

struct add_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(T1 a, T2 b) const
  {
    return a + b;
  }
};

struct sub_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(T1 a, T2 b) const
  {
    return a - b;
  }
};

struct mul_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(T1 a, T2 b) const
  {
    return a * b;
  }
};

struct div_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(T1 a, T2 b) const
  {
    return a / b;
  }
};

struct div_checkzero_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION Type operator()(Type a, Type b) const
  {
    if (b == Type{0}) { return Type{0}; }
    return a / b;
  }
};

struct pow_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION Type operator()(Type a, Type b) const
  {
    return std::pow(a, b);
  }
};

struct min_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION Type operator()(Type a, Type b) const
  {
    if (a > b) { return b; }
    return a;
  }
};

struct max_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION Type operator()(Type a, Type b) const
  {
    if (b > a) { return b; }
    return a;
  }
};

struct sqdiff_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION Type operator()(Type a, Type b) const
  {
    Type diff = a - b;
    return diff * diff;
  }
};

struct argmin_op {
  template <typename KVP>
  constexpr RAFT_INLINE_FUNCTION KVP operator()(const KVP& a, const KVP& b) const
  {
    if ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key))) { return b; }
    return a;
  }
};

struct argmax_op {
  template <typename KVP>
  constexpr RAFT_INLINE_FUNCTION KVP operator()(const KVP& a, const KVP& b) const
  {
    if ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key))) { return b; }
    return a;
  }
};

template <typename ScalarT>
struct const_op {
  const ScalarT scalar;

  constexpr const_op(ScalarT s) : scalar{s} {}

  template <typename InT>
  constexpr RAFT_INLINE_FUNCTION ScalarT operator()(InT unused) const
  {
    return scalar;
  }
};

template <typename ComposedOpT, typename ScalarT>
struct scalar_op {
  ComposedOpT composed_op;
  const ScalarT scalar;

  constexpr scalar_op(ScalarT s) : scalar{s} {}

  template <typename InT>
  constexpr RAFT_INLINE_FUNCTION auto operator()(InT a) const
  {
    return composed_op(a, scalar);
  }
};

template <typename Type>
using scalar_add_op = scalar_op<add_op, Type>;

template <typename Type>
using scalar_sub_op = scalar_op<sub_op, Type>;

template <typename Type>
using scalar_mul_op = scalar_op<mul_op, Type>;

template <typename Type>
using scalar_div_op = scalar_op<div_op, Type>;

template <typename Type>
using scalar_div_checkzero_op = scalar_op<div_checkzero_op, Type>;

template <typename Type>
using scalar_pow_op = scalar_op<pow_op, Type>;

template <typename OuterOpT, typename InnerOpT>
struct unary_compose_op {
  OuterOpT outer_op;
  InnerOpT inner_op;

  template <typename... Args>
  constexpr RAFT_INLINE_FUNCTION auto operator()(Args... args) const
  {
    return outer_op(inner_op(args...));
  }
};

template <typename OuterOpT, typename InnerOpT1, typename InnerOpT2 = raft::identity_op>
struct binary_compose_op {
  OuterOpT outer_op;
  InnerOpT1 inner_op1;
  InnerOpT2 inner_op2;

  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(T1 a, T2 b) const
  {
    return outer_op(inner_op1(a), inner_op2(b));
  }
};

/** @} */
}  // namespace raft
