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

/**
 * @brief Wraps around a binary operator, passing a given scalar on the right-hand side.
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/operators.hpp>
 *
 *  raft::scalar_op<float, raft::mul_op> op(2.0f);
 *  std::cout << op(2.1f) << std::endl;  // 4.2
 * @endcode
 *
 * @tparam ScalarT
 * @tparam BinaryOpT
 */
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

/**
 * @brief Constructs an operator by composing a chain of operators.
 *
 * Note that all arguments are passed to the innermost operator.
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/operators.hpp>
 *
 *  auto op = raft::compose_op(raft::sqrt_op(), raft::abs_op(), raft::cast_op<float>(),
 *                             raft::scalar_add_op<int>(8));
 *  std::cout << op(-50) << std::endl;  // 6.48074
 * @endcode
 *
 * @tparam OpsT Any number of operation types.
 */
template <typename... OpsT>
struct compose_op {
  const std::tuple<OpsT...> ops;

  template <typename TupleT = std::tuple<OpsT...>,
            typename CondT  = std::enable_if_t<std::is_default_constructible_v<TupleT>>>
  constexpr compose_op() : ops{}
  {
  }
  constexpr explicit compose_op(OpsT... ops) : ops{ops...} {}

  template <typename... Args>
  constexpr RAFT_INLINE_FUNCTION auto operator()(Args&&... args) const
  {
    return compose<sizeof...(OpsT)>(std::forward<Args>(args)...);
  }

 private:
  template <size_t RemOps, typename... Args>
  constexpr RAFT_INLINE_FUNCTION auto compose(Args&&... args) const
  {
    if constexpr (RemOps > 0) {
      return compose<RemOps - 1>(std::get<RemOps - 1>(ops)(std::forward<Args>(args)...));
    } else {
      return identity_op{}(std::forward<Args>(args)...);
    }
  }
};

/**
 * @brief Constructs an operator by composing an outer op with one inner op for each of its inputs.
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/operators.hpp>
 *
 *  raft::map_args_op<raft::add_op, raft::sqrt_op, raft::cast_op<float>> op;
 *  std::cout << op(42.0f, 10) << std::endl;  // 16.4807
 * @endcode
 *
 * @tparam OuterOpT Outer operation type
 * @tparam ArgOpsT Operation types for each input of the outer operation
 */
template <typename OuterOpT, typename... ArgOpsT>
struct map_args_op {
  const OuterOpT outer_op;
  const std::tuple<ArgOpsT...> arg_ops;

  template <typename T1    = OuterOpT,
            typename T2    = std::tuple<ArgOpsT...>,
            typename CondT = std::enable_if_t<std::is_default_constructible_v<T1> &&
                                              std::is_default_constructible_v<T2>>>
  constexpr map_args_op() : outer_op{}, arg_ops{}
  {
  }
  constexpr explicit map_args_op(OuterOpT outer_op, ArgOpsT... arg_ops)
    : outer_op{outer_op}, arg_ops{arg_ops...}
  {
  }

  template <typename... Args>
  constexpr RAFT_INLINE_FUNCTION auto operator()(Args&&... args) const
  {
    constexpr size_t kNumOps = sizeof...(ArgOpsT);
    static_assert(kNumOps == sizeof...(Args),
                  "The number of arguments does not match the number of mapping operators");
    return map_args(std::make_index_sequence<kNumOps>{}, std::forward<Args>(args)...);
  }

 private:
  template <size_t... I, typename... Args>
  constexpr RAFT_INLINE_FUNCTION auto map_args(std::index_sequence<I...>, Args&&... args) const
  {
    return outer_op(std::get<I>(arg_ops)(std::forward<Args>(args))...);
  }
};

/** @} */
}  // namespace raft
