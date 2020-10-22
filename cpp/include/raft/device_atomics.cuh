/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

/**
 * @brief overloads for CUDA atomic operations
 * @file device_atomics.cuh
 *
 * Provides the overloads for arithmetic data types, where CUDA atomic operations are, `atomicAdd`,
 * `atomicMin`, `atomicMax`, and `atomicCAS`.
 * `atomicAnd`, `atomicOr`, `atomicXor` are also supported for integer data types.
 * Also provides `raft::genericAtomicOperation` which performs atomic operation with the given
 * binary operator.
 */

#include <type_traits>

namespace raft {

namespace device_atomics {
namespace detail {

// -------------------------------------------------------------------------------------------------
// Binary operators

/* @brief binary `sum` operator */
struct device_sum {
  template <typename T,
            typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  __device__ T operator()(const T& lhs, const T& rhs) {
    return lhs + rhs;
  }
};

/* @brief binary `min` operator */
struct device_min {
  template <typename T>
  __device__ T operator()(const T& lhs, const T& rhs) {
    return lhs < rhs ? lhs : rhs;
  }
};

/* @brief binary `max` operator */
struct device_max {
  template <typename T>
  __device__ T operator()(const T& lhs, const T& rhs) {
    return lhs > rhs ? lhs : rhs;
  }
};

/* @brief binary `product` operator */
struct device_product {
  template <typename T,
            typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  __device__ T operator()(const T& lhs, const T& rhs) {
    return lhs * rhs;
  }
};

/* @brief binary `and` operator */
struct device_and {
  template <typename T,
            typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  __device__ T operator()(const T& lhs, const T& rhs) {
    return (lhs & rhs);
  }
};

/* @brief binary `or` operator */
struct device_or {
  template <typename T,
            typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  __device__ T operator()(const T& lhs, const T& rhs) {
    return (lhs | rhs);
  }
};

/* @brief binary `xor` operator */
struct device_xor {
  template <typename T,
            typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  __device__ T operator()(const T& lhs, const T& rhs) {
    return (lhs ^ rhs);
  }
};

// FIXME: remove this if C++17 is supported.
// `static_assert` requires a string literal at C++14.
#define errmsg_cast "size mismatch."

template <typename OutputT, typename InputT>
__forceinline__ __device__ OutputT type_reinterpret(InputT value) {
  static_assert(sizeof(OutputT) == sizeof(InputT),
                "type_reinterpret for different size");
  return *(reinterpret_cast<OutputT*>(&value));
}

// -------------------------------------------------------------------------------------------------
// the implementation of `genericAtomicOperation`

template <typename T, typename Op, size_t N = sizeof(T)>
struct genericAtomicOperationImpl;

// single byte atomic operation
template <typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 1> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value,
                                          Op op) {
    using int_t = unsigned int;

    auto* address_uint32 =
      reinterpret_cast<int_t*>(addr - (reinterpret_cast<size_t>(addr) & 3));
    int_t shift = ((reinterpret_cast<size_t>(addr) & 3) * 8);

    int_t old = *address_uint32;
    int_t assumed;

    do {
      assumed = old;
      T target_value = T((old >> shift) & 0xff);
      auto updating_value =
        type_reinterpret<uint8_t, T>(op(target_value, update_value));
      int_t new_value =
        (old & ~(0x000000ff << shift)) | (int_t(updating_value) << shift);
      old = atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return T((old >> shift) & 0xff);
  }
};

// 2 bytes atomic operation
template <typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 2> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value,
                                          Op op) {
    using int_t = unsigned int;
    bool is_32_align = (reinterpret_cast<size_t>(addr) & 2) ? false : true;
    auto* address_uint32 = reinterpret_cast<int_t*>(
      reinterpret_cast<size_t>(addr) - (is_32_align ? 0 : 2));

    int_t old = *address_uint32;
    int_t assumed;

    do {
      assumed = old;
      T target_value = (is_32_align) ? T(old & 0xffff) : T(old >> 16);
      auto updating_value =
        type_reinterpret<uint16_t, T>(op(target_value, update_value));

      int_t new_value = (is_32_align)
                          ? (old & 0xffff0000) | updating_value
                          : (old & 0xffff) | (int_t(updating_value) << 16);
      old = atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return (is_32_align) ? T(old & 0xffff) : T(old >> 16);
    ;
  }
};

// 4 bytes atomic operation
template <typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 4> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value,
                                          Op op) {
    using int_t = unsigned int;

    T old_value = *addr;
    T assumed{old_value};

    do {
      assumed = old_value;
      const T new_value = op(old_value, update_value);

      int_t ret = atomicCAS(reinterpret_cast<int_t*>(addr),
                            type_reinterpret<int_t, T>(assumed),
                            type_reinterpret<int_t, T>(new_value));
      old_value = type_reinterpret<T, int_t>(ret);

    } while (assumed != old_value);

    return old_value;
  }
};

// 8 bytes atomic operation
template <typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 8> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value,
                                          Op op) {
    using int_t = unsigned long long int;  // NOLINT
    static_assert(sizeof(T) == sizeof(int_t), errmsg_cast);

    T old_value = *addr;
    T assumed{old_value};

    do {
      assumed = old_value;
      const T new_value = op(old_value, update_value);

      int_t ret = atomicCAS(reinterpret_cast<int_t*>(addr),
                            type_reinterpret<int_t, T>(assumed),
                            type_reinterpret<int_t, T>(new_value));
      old_value = type_reinterpret<T, int_t>(ret);

    } while (assumed != old_value);

    return old_value;
  }
};

// -------------------------------------------------------------------------------------------------
// specialized functions for operators
// `atomicAdd` supports int, unsigned int, unsigend long long int, float, double (long long int is not supproted.)
// `atomicMin`, `atomicMax` support int, unsigned int, unsigned long long int
// `atomicAnd`, `atomicOr`, `atomicXor` support int, unsigned int, unsigned long long int

// CUDA natively supports `unsigned long long int` for `atomicAdd`,
// but doesn't supports `long int`.
// However, since the signed integer is represented as Two's complement,
// the fundamental arithmetic operations of addition are identical to
// those for unsigned binary numbers.
// Then, this computes as `unsigned long long int` with `atomicAdd`
// @sa https://en.wikipedia.org/wiki/Two%27s_complement
template <>
struct genericAtomicOperationImpl<long int, device_sum, 8> {  // NOLINT
  using input_t = long int;  // NOLINT
  __forceinline__ __device__ input_t operator()(input_t* addr, input_t const& update_value,
                                                device_sum op) {
    using int_t = unsigned long long int;  // NOLINT
    static_assert(sizeof(input_t) == sizeof(int_t), errmsg_cast);
    int_t ret = atomicAdd(reinterpret_cast<int_t*>(addr),
                          type_reinterpret<int_t, input_t>(update_value));
    return type_reinterpret<input_t, int_t>(ret);
  }
};

template <>
struct genericAtomicOperationImpl<unsigned long int, device_sum, 8> {  // NOLINT
  using input_t = unsigned long int;  // NOLINT
  __forceinline__ __device__ input_t operator()(input_t* addr, input_t const& update_value,
                                          device_sum op) {
    using int_t = unsigned long long int;  // NOLINT
    static_assert(sizeof(input_t) == sizeof(int_t), errmsg_cast);
    int_t ret = atomicAdd(reinterpret_cast<int_t*>(addr),
                          type_reinterpret<int_t, input_t>(update_value));
    return type_reinterpret<input_t, int_t>(ret);
  }
};

// CUDA natively supports `unsigned long long int` for `atomicAdd`,
// but doesn't supports `long long int`.
// However, since the signed integer is represented as Two's complement,
// the fundamental arithmetic operations of addition are identical to
// those for unsigned binary numbers.
// Then, this computes as `unsigned long long int` with `atomicAdd`
// @sa https://en.wikipedia.org/wiki/Two%27s_complement
template <>
struct genericAtomicOperationImpl<long long int, device_sum, 8> {  // NOLINT
  using input_t = long long int;  // NOLINT
  __forceinline__ __device__ input_t operator()(input_t* addr, input_t const& update_value,
                                          device_sum op) {
    using int_t = unsigned long long int;  // NOLINT
    static_assert(sizeof(input_t) == sizeof(int_t), errmsg_cast);
    int_t ret = atomicAdd(reinterpret_cast<int_t*>(addr),
                          type_reinterpret<int_t, T>(update_value));
    return type_reinterpret<T, int_t>(ret);
  }
};

template <>
struct genericAtomicOperationImpl<unsigned long int, device_min, 8> {  // NOLINT
  using input_t = unsigned long int;  // NOLINT
  __forceinline__ __device__ input_t operator()(input_t* addr, input_t const& update_value,
                                          device_min op) {
    using int_t = unsigned long long int;  // NOLINT
    static_assert(sizeof(input_t) == sizeof(int_t), errmsg_cast);
    input_t ret = atomicMin(reinterpret_cast<int_t*>(addr),
                      type_reinterpret<int_t, input_t>(update_value));
    return type_reinterpret<input_t, int_t>(ret);
  }
};

template <>
struct genericAtomicOperationImpl<unsigned long int, device_max, 8> {  // NOLINT
  using input_t = unsigned long int;  // NOLINT
  __forceinline__ __device__ input_t operator()(input_t* addr, input_t const& update_value,
                                          device_max op) {
    using int_t = unsigned long long int;  // NOLINT
    static_assert(sizeof(T) == sizeof(int_t), errmsg_cast);
    input_t ret = atomicMax(reinterpret_cast<int_t*>(addr),
                            type_reinterpret<int_t, input_t>(update_value));
    return type_reinterpret<input_t, int_t>(ret);
  }
};

template <typename T>
struct genericAtomicOperationImpl<T, device_and, 8> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value,
                                          device_and op) {
    using int_t = unsigned long long int;  // NOLINT
    static_assert(sizeof(T) == sizeof(int_t), errmsg_cast);
    int_t ret = atomicAnd(reinterpret_cast<int_t*>(addr),
                          type_reinterpret<int_t, T>(update_value));
    return type_reinterpret<T, int_t>(ret);
  }
};

template <typename T>
struct genericAtomicOperationImpl<T, device_or, 8> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value,
                                          device_or op) {
    using int_t = unsigned long long int;  // NOLINT
    static_assert(sizeof(T) == sizeof(int_t), errmsg_cast);
    int_t ret = atomicOr(reinterpret_cast<int_t*>(addr),
                         type_reinterpret<int_t, T>(update_value));
    return type_reinterpret<T, int_t>(ret);
  }
};

template <typename T>
struct genericAtomicOperationImpl<T, device_xor, 8> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value,
                                          device_xor op) {
    using int_t = unsigned long long int;  // NOLINT
    static_assert(sizeof(T) == sizeof(int_t), errmsg_cast);
    int_t ret = atomicXor(reinterpret_cast<int_t*>(addr),
                          type_reinterpret<int_t, T>(update_value));
    return type_reinterpret<T, int_t>(ret);
  }
};

// -------------------------------------------------------------------------------------------------
// the implementation of `typesAtomicCASImpl`

template <typename T, size_t N = sizeof(T)>
struct typesAtomicCASImpl;

template <typename T>
struct typesAtomicCASImpl<T, 1> {
  __forceinline__ __device__ T operator()(T* addr, T const& compare,
                                          T const& update_value) {
    using int_t = unsigned int;

    int_t shift = ((reinterpret_cast<size_t>(addr) & 3) * 8);
    auto* address_uint32 =
      reinterpret_cast<int_t*>(addr - (reinterpret_cast<size_t>(addr) & 3));

    // the 'target_value' in `old` can be different from `compare`
    // because other thread may update the value
    // before fetching a value from `address_uint32` in this function
    int_t old = *address_uint32;
    int_t assumed;
    T target_value;
    uint8_t u_val = type_reinterpret<uint8_t, T>(update_value);

    do {
      assumed = old;
      target_value = T((old >> shift) & 0xff);
      // have to compare `target_value` and `compare` before calling atomicCAS
      // the `target_value` in `old` can be different with `compare`
      if (target_value != compare) break;

      int_t new_value =
        (old & ~(0x000000ff << shift)) | (int_t(u_val) << shift);
      old = atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return target_value;
  }
};

template <typename T>
struct typesAtomicCASImpl<T, 2> {
  __forceinline__ __device__ T operator()(T* addr, T const& compare,
                                          T const& update_value) {
    using int_t = unsigned int;

    bool is_32_align = (reinterpret_cast<size_t>(addr) & 2) ? false : true;
    auto* address_uint32 = reinterpret_cast<int_t*>(
      reinterpret_cast<size_t>(addr) - (is_32_align ? 0 : 2));

    int_t old = *address_uint32;
    int_t assumed;
    T target_value;
    uint16_t u_val = type_reinterpret<uint16_t, T>(update_value);

    do {
      assumed = old;
      target_value = (is_32_align) ? T(old & 0xffff) : T(old >> 16);
      if (target_value != compare) break;

      int_t new_value = (is_32_align) ? (old & 0xffff0000) | u_val
                                      : (old & 0xffff) | (int_t(u_val) << 16);
      old = atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return target_value;
  }
};

template <typename T>
struct typesAtomicCASImpl<T, 4> {
  __forceinline__ __device__ T operator()(T* addr, T const& compare,
                                          T const& update_value) {
    using int_t = unsigned int;

    int_t ret = atomicCAS(reinterpret_cast<int_t*>(addr),
                          type_reinterpret<int_t, T>(compare),
                          type_reinterpret<int_t, T>(update_value));

    return type_reinterpret<T, int_t>(ret);
  }
};

// 8 bytes atomic operation
template <typename T>
struct typesAtomicCASImpl<T, 8> {
  __forceinline__ __device__ T operator()(T* addr, T const& compare,
                                          T const& update_value) {
    using int_t = unsigned long long int;  // NOLINT
    static_assert(sizeof(T) == sizeof(int_t), errmsg_cast);

    int_t ret = atomicCAS(reinterpret_cast<int_t*>(addr),
                          type_reinterpret<int_t, T>(compare),
                          type_reinterpret<int_t, T>(update_value));

    return type_reinterpret<T, int_t>(ret);
  }
};

}  // namespace detail
}  // namespace device_atomics

/** -------------------------------------------------------------------------*
 * @brief compute atomic binary operation
 * reads the `old` located at the `address` in global or shared memory,
 * computes 'BinaryOp'('old', 'update_value'),
 * and stores the result back to memory at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `genericAtomicOperation` are:
 * int8_t, int16_t, int32_t, int64_t, float, double
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 * @param[in] op  The binary operator used for compute
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
template <typename T, typename BinaryOp>
typename std::enable_if_t<std::is_arithmetic<T>::value, T> __forceinline__
  __device__
  genericAtomicOperation(T* address, T const& update_value, BinaryOp op) {
  auto fun =
    raft::device_atomics::detail::genericAtomicOperationImpl<T, BinaryOp>{};
  return T(fun(address, update_value, op));
}

// specialization for bool types
template <typename BinaryOp>
__forceinline__ __device__ bool genericAtomicOperation(bool* address,
                                                       bool const& update_value,
                                                       BinaryOp op) {
  using T = bool;
  // don't use underlying type to apply operation for bool
  auto fun =
    raft::device_atomics::detail::genericAtomicOperationImpl<T, BinaryOp>{};
  return T(fun(address, update_value, op));
}

}  // namespace raft

// NOTE: the below method names have NOLINT against them because we want their
//       names to reflect those corresponding ones in the cudart library

/**
 * @brief Overloads for `atomicAdd`
 *
 * reads the `old` located at the `address` in global or shared memory, computes (old + val), and
 * stores the result back to memory at the same address. These three operations are performed in one
 * atomic transaction.
 *
 * The supported types for `atomicAdd` are: integers are floating point numbers.
 * CUDA natively supports `int`, `unsigned int`, `unsigned long long int`, `float`, `double.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be added
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomicAdd(T* address, T val) {  // NOLINT
  return raft::genericAtomicOperation(
    address, val, raft::device_atomics::detail::device_sum{});
}

/**
 * @brief Overloads for `atomicMin`
 *
 * reads the `old` located at the `address` in global or shared memory, computes the minimum of old
 * and val, and stores the result back to memory at the same address. These three operations are
 * performed in one atomic transaction.
 *
 * The supported types for `atomicMin` are: integers are floating point numbers.
 * CUDA natively supports `int`, `unsigend int`, `unsigned long long int`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomicMin(T* address, T val) {  // NOLINT
  return raft::genericAtomicOperation(
    address, val, raft::device_atomics::detail::device_min{});
}

/**
 * @brief Overloads for `atomicMax`
 *
 * reads the `old` located at the `address` in global or shared memory, computes the maximum of old
 * and val, and stores the result back to memory at the same address. These three operations are
 * performed in one atomic transaction.
 *
 * The supported types for `atomicMax` are: integers are floating point numbers.
 * CUDA natively supports `int`, `unsigend int`, `unsigned long long int`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomicMax(T* address, T val) {  // NOLINT
  return raft::genericAtomicOperation(
    address, val, raft::device_atomics::detail::device_max{});
}

/**
 * @brief Overloads for `atomicCAS`
 *
 * reads the `old` located at the `address` in global or shared memory, computes
 * (`old` == `compare` ? `val` : `old`), and stores the result back to memory at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported types for `atomicCAS` are: integers are floating point numbers.
 * CUDA natively supports `int`, `unsigned int`, `unsigned long long int`, `unsigned short int`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] compare The value to be compared
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomicCAS(T* address, T compare, T val) {  // NOLINT
  return raft::device_atomics::detail::typesAtomicCASImpl<T>()(address, compare,
                                                               val);
}

/**
 * @brief Overloads for `atomicAnd`
 *
 * reads the `old` located at the `address` in global or shared memory, computes (old & val), and
 * stores the result back to memory at the same address. These three operations are performed in
 * one atomic transaction.
 *
 * The supported types for `atomicAnd` are: integers.
 * CUDA natively supports `int`, `unsigned int`, `unsigned long long int`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T,
          typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
__forceinline__ __device__ T atomicAnd(T* address, T val) {  // NOLINT
  return raft::genericAtomicOperation(
    address, val, raft::device_atomics::detail::device_and{});
}

/**
 * @brief Overloads for `atomicOr`
 *
 * reads the `old` located at the `address` in global or shared memory, computes (old | val), and
 * stores the result back to memory at the same address. These three operations are performed in
 * one atomic transaction.
 *
 * The supported types for `atomicOr` are: integers.
 * CUDA natively supports `int`, `unsigned int`, `unsigned long long int`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T,
          typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
__forceinline__ __device__ T atomicOr(T* address, T val) {  // NOLINT
  return raft::genericAtomicOperation(address, val,
                                      raft::device_atomics::detail::device_or{});
}

/**
 * @brief Overloads for `atomicXor`
 *
 * reads the `old` located at the `address` in global or shared memory, computes (old ^ val), and
 * stores the result back to memory at the same address. These three operations are performed in
 * one atomic transaction.
 *
 * The supported types for `atomicXor` are: integers.
 * CUDA natively supports `int`, `unsigned int`, `unsigned long long int`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T,
          typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
__forceinline__ __device__ T atomicXor(T* address, T val) {  // NOLINT
  return raft::genericAtomicOperation(
    address, val, raft::device_atomics::detail::device_xor{});
}
