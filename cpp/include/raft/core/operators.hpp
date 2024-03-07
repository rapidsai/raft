/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/detail/macros.hpp>
#include <raft/core/math.hpp>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>

namespace raft {

/**
 * @defgroup operators Commonly used functors.
 * The optional unused arguments are useful for kernels that pass the index along with the value.
 * @{
 */

struct identity_op {
  template <typename Type, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in, UnusedArgs...) const
  {
    return in;
  }
};

struct void_op {
  template <typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION void operator()(UnusedArgs...) const
  {
    return;
  }
};

template <typename OutT>
struct cast_op {
  template <typename InT, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION auto operator()(InT in, UnusedArgs...) const
  {
    return static_cast<OutT>(in);
  }
};

struct key_op {
  template <typename KVP, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const KVP& p, UnusedArgs...) const
  {
    return p.key;
  }
};

struct value_op {
  template <typename KVP, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const KVP& p, UnusedArgs...) const
  {
    return p.value;
  }
};

struct sqrt_op {
  template <typename Type, typename... UnusedArgs>
  RAFT_INLINE_FUNCTION auto operator()(const Type& in, UnusedArgs...) const
  {
    return raft::sqrt(in);
  }
};

struct nz_op {
  template <typename Type, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in, UnusedArgs...) const
  {
    return in != Type(0) ? Type(1) : Type(0);
  }
};

struct abs_op {
  template <typename Type, typename... UnusedArgs>
  RAFT_INLINE_FUNCTION auto operator()(const Type& in, UnusedArgs...) const
  {
    return raft::abs(in);
  }
};

struct sq_op {
  template <typename Type, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in, UnusedArgs...) const
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
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T1& a, const T2& b) const
  {
    if (b == T2{0}) { return T1{0} / T2{1}; }
    return a / b;
  }
};

struct pow_op {
  template <typename Type>
  RAFT_INLINE_FUNCTION auto operator()(const Type& a, const Type& b) const
  {
    return raft::pow(a, b);
  }
};

struct mod_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T1& a, const T2& b) const
  {
    return a % b;
  }
};

struct min_op {
  template <typename... Args>
  RAFT_INLINE_FUNCTION auto operator()(Args&&... args) const
  {
    return raft::min(std::forward<Args>(args)...);
  }
};

struct max_op {
  template <typename... Args>
  RAFT_INLINE_FUNCTION auto operator()(Args&&... args) const
  {
    return raft::max(std::forward<Args>(args)...);
  }
};

struct argmin_op {
  template <typename KVP>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const KVP& a, const KVP& b) const
  {
    if ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key))) { return b; }
    return a;
  }
};

struct argmax_op {
  template <typename KVP>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const KVP& a, const KVP& b) const
  {
    if ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key))) { return b; }
    return a;
  }
};

struct greater_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T1& a, const T2& b) const
  {
    return a > b;
  }
};

struct less_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T1& a, const T2& b) const
  {
    return a < b;
  }
};

struct greater_or_equal_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T1& a, const T2& b) const
  {
    return a >= b;
  }
};

struct less_or_equal_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T1& a, const T2& b) const
  {
    return a <= b;
  }
};

struct equal_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T1& a, const T2& b) const
  {
    return a == b;
  }
};

struct notequal_op {
  template <typename T1, typename T2>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T1& a, const T2& b) const
  {
    return a != b;
  }
};

template <typename ScalarT>
struct const_op {
  const ScalarT scalar;

  constexpr explicit const_op(const ScalarT& s) : scalar{s} {}

  template <typename... Args>
  constexpr RAFT_INLINE_FUNCTION auto operator()(Args...) const
  {
    return scalar;
  }
};

/**
 * @brief Wraps around a binary operator, passing a constant on the right-hand side.
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/operators.hpp>
 *
 *  raft::plug_const_op<float, raft::mul_op> op(2.0f);
 *  std::cout << op(2.1f) << std::endl;  // 4.2
 * @endcode
 *
 * @tparam ConstT
 * @tparam BinaryOpT
 */
template <typename ConstT, typename BinaryOpT>
struct plug_const_op {
  const ConstT c;
  const BinaryOpT composed_op;

  template <typename OpT     = BinaryOpT,
            typename UnusedT = std::enable_if_t<std::is_default_constructible_v<OpT>>>
  constexpr explicit plug_const_op(const ConstT& s)
    : c{s}, composed_op{}  // The compiler complains if composed_op is not initialized explicitly
  {
  }
  constexpr plug_const_op(const ConstT& s, BinaryOpT o) : c{s}, composed_op{o} {}

  template <typename InT>
  constexpr RAFT_INLINE_FUNCTION auto operator()(InT a) const
  {
    return composed_op(a, c);
  }
};

template <typename Type>
using add_const_op = plug_const_op<Type, add_op>;

template <typename Type>
using sub_const_op = plug_const_op<Type, sub_op>;

template <typename Type>
using mul_const_op = plug_const_op<Type, mul_op>;

template <typename Type>
using div_const_op = plug_const_op<Type, div_op>;

template <typename Type>
using div_checkzero_const_op = plug_const_op<Type, div_checkzero_op>;

template <typename Type>
using pow_const_op = plug_const_op<Type, pow_op>;

template <typename Type>
using mod_const_op = plug_const_op<Type, mod_op>;

template <typename Type>
using mod_const_op = plug_const_op<Type, mod_op>;

template <typename Type>
using equal_const_op = plug_const_op<Type, equal_op>;

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
 *                             raft::add_const_op<int>(8));
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
  constexpr compose_op()
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

using absdiff_op = compose_op<abs_op, sub_op>;

using sqdiff_op = compose_op<sq_op, sub_op>;

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
  constexpr map_args_op()
    : outer_op{}  // The compiler complains if outer_op is not initialized explicitly
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
