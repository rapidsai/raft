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
#include <type_traits>
#include <utility>

#include <raft/core/detail/macros.hpp>

namespace raft {

/**
 * @defgroup Functors Commonly used functors.
 * The optional unused arguments are useful for kernels that pass the index along with the value.
 * @{
 */

struct identity_op {
  template <typename Type, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION Type operator()(const Type& in, UnusedArgs...) const
  {
    return in;
  }
};

template <typename OutT>
struct cast_op {
  template <typename InT, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION OutT operator()(InT in, UnusedArgs...) const
  {
    return static_cast<OutT>(in);
  }
};

struct key_op {
  template <typename KVP, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION typename KVP::Key operator()(const KVP& p, UnusedArgs...) const
  {
    return p.key;
  }
};

struct value_op {
  template <typename KVP, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION typename KVP::Value operator()(const KVP& p, UnusedArgs...) const
  {
    return p.value;
  }
};

struct sqrt_op {
  template <typename Type, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION Type operator()(const Type& in, UnusedArgs...) const
  {
    return std::sqrt(in);
  }
};

struct nz_op {
  template <typename Type, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION Type operator()(const Type& in, UnusedArgs...) const
  {
    return in != Type(0) ? Type(1) : Type(0);
  }
};

struct abs_op {
  template <typename Type, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION Type operator()(const Type& in, UnusedArgs...) const
  {
    return std::abs(in);
  }
};

struct sq_op {
  template <typename Type, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION Type operator()(const Type& in, UnusedArgs...) const
  {
    return in * in;
  }
};

struct add_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T1& a, const T2& b) const
  {
    return a + b;
  }
};

struct sub_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T1& a, const T2& b) const
  {
    return a - b;
  }
};

struct mul_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T1& a, const T2& b) const
  {
    return a * b;
  }
};

struct div_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T1& a, const T2& b) const
  {
    return a / b;
  }
};

struct div_checkzero_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION Type operator()(const Type& a, const Type& b) const
  {
    if (b == Type{0}) { return Type{0}; }
    return a / b;
  }
};

struct pow_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION Type operator()(const Type& a, const Type& b) const
  {
    return std::pow(a, b);
  }
};

struct min_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION Type operator()(const Type& a, const Type& b) const
  {
    if (a > b) { return b; }
    return a;
  }
};

struct max_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION Type operator()(const Type& a, const Type& b) const
  {
    if (b > a) { return b; }
    return a;
  }
};

struct sqdiff_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION Type operator()(const Type& a, const Type& b) const
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

  constexpr const_op(const ScalarT& s) : scalar{s} {}

  template <typename... Args>
  constexpr RAFT_INLINE_FUNCTION ScalarT operator()(Args...) const
  {
    return scalar;
  }
};

template <typename ScalarT, typename BinaryOpT>
struct scalar_op {
  const ScalarT scalar;
  const BinaryOpT composed_op;

  template <typename OpT     = BinaryOpT,
            typename UnusedT = std::enable_if_t<std::is_default_constructible_v<OpT>>>
  constexpr scalar_op(const ScalarT& s) : scalar{s}, composed_op{}
  {
  }
  constexpr scalar_op(const ScalarT& s, BinaryOpT o) : scalar{s}, composed_op{o} {}

  template <typename InT>
  constexpr RAFT_INLINE_FUNCTION auto operator()(InT a) const
  {
    return composed_op(a, scalar);
  }
};

template <typename Type>
using scalar_add_op = scalar_op<Type, add_op>;

template <typename Type>
using scalar_sub_op = scalar_op<Type, sub_op>;

template <typename Type>
using scalar_mul_op = scalar_op<Type, mul_op>;

template <typename Type>
using scalar_div_op = scalar_op<Type, div_op>;

template <typename Type>
using scalar_div_checkzero_op = scalar_op<Type, div_checkzero_op>;

template <typename Type>
using scalar_pow_op = scalar_op<Type, pow_op>;

template <typename OuterOpT, typename InnerOpT>
struct compose_op {
  const OuterOpT outer_op;
  const InnerOpT inner_op;

  template <typename OOpT    = OuterOpT,
            typename IOpT    = InnerOpT,
            typename UnusedT = std::enable_if_t<std::is_default_constructible_v<OOpT> &&
                                                std::is_default_constructible_v<IOpT>>>
  constexpr compose_op() : outer_op{}, inner_op{}
  {
  }
  constexpr compose_op(OuterOpT out_op, InnerOpT in_op) : outer_op{out_op}, inner_op{in_op} {}

  template <typename... Args>
  constexpr RAFT_INLINE_FUNCTION auto operator()(Args&&... args) const
  {
    return outer_op(inner_op(std::forward<Args>(args)...));
  }
};

template <typename OuterOpT, typename InnerOpT1, typename InnerOpT2 = raft::identity_op>
struct binary_compose_op {
  const OuterOpT outer_op;
  const InnerOpT1 inner_op1;
  const InnerOpT2 inner_op2;

  template <typename OOpT    = OuterOpT,
            typename IOpT1   = InnerOpT1,
            typename IOpT2   = InnerOpT2,
            typename UnusedT = std::enable_if_t<std::is_default_constructible_v<OOpT> &&
                                                std::is_default_constructible_v<IOpT1> &&
                                                std::is_default_constructible_v<IOpT2>>>
  constexpr binary_compose_op() : outer_op{}, inner_op1{}, inner_op2{}
  {
  }
  constexpr binary_compose_op(OuterOpT out_op, InnerOpT1 in_op1, InnerOpT2 in_op2)
    : outer_op{out_op}, inner_op1{in_op1}, inner_op2{in_op2}
  {
  }

  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T1& a, const T2& b) const
  {
    return outer_op(inner_op1(a), inner_op2(b));
  }
};

/** @} */
}  // namespace raft
