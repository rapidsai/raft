/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cooperative_groups.h>

#include <type_traits>

namespace raft {

namespace device_atomics {
namespace detail {

// -------------------------------------------------------------------------------------------------
// Binary operators

/* @brief binary `sum` operator */
struct DeviceSum {
  template <typename T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  __device__ T operator()(const T& lhs, const T& rhs)
  {
    return lhs + rhs;
  }
};

/* @brief binary `min` operator */
struct DeviceMin {
  template <typename T>
  __device__ T operator()(const T& lhs, const T& rhs)
  {
    return lhs < rhs ? lhs : rhs;
  }
};

/* @brief binary `max` operator */
struct DeviceMax {
  template <typename T>
  __device__ T operator()(const T& lhs, const T& rhs)
  {
    return lhs > rhs ? lhs : rhs;
  }
};

/* @brief binary `product` operator */
struct DeviceProduct {
  template <typename T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  __device__ T operator()(const T& lhs, const T& rhs)
  {
    return lhs * rhs;
  }
};

/* @brief binary `and` operator */
struct DeviceAnd {
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  __device__ T operator()(const T& lhs, const T& rhs)
  {
    return (lhs & rhs);
  }
};

/* @brief binary `or` operator */
struct DeviceOr {
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  __device__ T operator()(const T& lhs, const T& rhs)
  {
    return (lhs | rhs);
  }
};

/* @brief binary `xor` operator */
struct DeviceXor {
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  __device__ T operator()(const T& lhs, const T& rhs)
  {
    return (lhs ^ rhs);
  }
};

// FIXME: remove this if C++17 is supported.
// `static_assert` requires a string literal at C++14.
#define errmsg_cast "size mismatch."

template <typename T_output, typename T_input>
__forceinline__ __device__ T_output type_reinterpret(T_input value)
{
  static_assert(sizeof(T_output) == sizeof(T_input), "type_reinterpret for different size");
  return *(reinterpret_cast<T_output*>(&value));
}

// -------------------------------------------------------------------------------------------------
// the implementation of `genericAtomicOperation`

template <typename T, typename Op, size_t N = sizeof(T)>
struct genericAtomicOperationImpl;

// single byte atomic operation
template <typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 1> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, Op op)
  {
    using T_int = unsigned int;

    T_int* address_uint32 = reinterpret_cast<T_int*>(addr - (reinterpret_cast<size_t>(addr) & 3));
    T_int shift           = ((reinterpret_cast<size_t>(addr) & 3) * 8);

    T_int old = *address_uint32;
    T_int assumed;

    do {
      assumed                = old;
      T target_value         = T((old >> shift) & 0xff);
      uint8_t updating_value = type_reinterpret<uint8_t, T>(op(target_value, update_value));
      T_int new_value        = (old & ~(0x000000ff << shift)) | (T_int(updating_value) << shift);
      old                    = atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return T((old >> shift) & 0xff);
  }
};

// 2 bytes atomic operation
template <typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 2> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, Op op)
  {
    using T_int      = unsigned int;
    bool is_32_align = (reinterpret_cast<size_t>(addr) & 2) ? false : true;
    T_int* address_uint32 =
      reinterpret_cast<T_int*>(reinterpret_cast<size_t>(addr) - (is_32_align ? 0 : 2));

    T_int old = *address_uint32;
    T_int assumed;

    do {
      assumed                 = old;
      T target_value          = (is_32_align) ? T(old & 0xffff) : T(old >> 16);
      uint16_t updating_value = type_reinterpret<uint16_t, T>(op(target_value, update_value));

      T_int new_value = (is_32_align) ? (old & 0xffff0000) | updating_value
                                      : (old & 0xffff) | (T_int(updating_value) << 16);
      old             = atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return (is_32_align) ? T(old & 0xffff) : T(old >> 16);
    ;
  }
};

// 4 bytes atomic operation
template <typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 4> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, Op op)
  {
    using T_int = unsigned int;
    T old_value = *addr;
    T assumed{old_value};

    if constexpr (std::is_same<T, float>{} && (std::is_same<Op, DeviceMin>{})) {
      if (isnan(update_value)) { return old_value; }
    }

    do {
      assumed           = old_value;
      const T new_value = op(old_value, update_value);

      T_int ret = atomicCAS(reinterpret_cast<T_int*>(addr),
                            type_reinterpret<T_int, T>(assumed),
                            type_reinterpret<T_int, T>(new_value));
      old_value = type_reinterpret<T, T_int>(ret);
    } while (assumed != old_value);

    return old_value;
  }
};

// 4 bytes fp32 atomic Max operation
template <>
struct genericAtomicOperationImpl<float, DeviceMax, 4> {
  using T = float;
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceMax op)
  {
    if (isnan(update_value)) { return *addr; }

    T old = (update_value >= 0)
              ? __int_as_float(atomicMax((int*)addr, __float_as_int(update_value)))
              : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(update_value)));

    return old;
  }
};

// 8 bytes atomic operation
template <typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 8> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, Op op)
  {
    using T_int = unsigned long long int;
    static_assert(sizeof(T) == sizeof(T_int), errmsg_cast);

    T old_value = *addr;
    T assumed{old_value};

    do {
      assumed           = old_value;
      const T new_value = op(old_value, update_value);

      T_int ret = atomicCAS(reinterpret_cast<T_int*>(addr),
                            type_reinterpret<T_int, T>(assumed),
                            type_reinterpret<T_int, T>(new_value));
      old_value = type_reinterpret<T, T_int>(ret);

    } while (assumed != old_value);

    return old_value;
  }
};

// -------------------------------------------------------------------------------------------------
// specialized functions for operators
// `atomicAdd` supports int, unsigned int, unsigned long long int, float, double (long long int is
// not supported.) `atomicMin`, `atomicMax` support int, unsigned int, unsigned long long int
// `atomicAnd`, `atomicOr`, `atomicXor` support int, unsigned int, unsigned long long int

// CUDA natively supports `unsigned long long int` for `atomicAdd`,
// but doesn't supports `long int`.
// However, since the signed integer is represented as Two's complement,
// the fundamental arithmetic operations of addition are identical to
// those for unsigned binary numbers.
// Then, this computes as `unsigned long long int` with `atomicAdd`
// @sa https://en.wikipedia.org/wiki/Two%27s_complement
template <>
struct genericAtomicOperationImpl<long int, DeviceSum, 8> {
  using T = long int;
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceSum op)
  {
    using T_int = unsigned long long int;
    static_assert(sizeof(T) == sizeof(T_int), errmsg_cast);
    T_int ret = atomicAdd(reinterpret_cast<T_int*>(addr), type_reinterpret<T_int, T>(update_value));
    return type_reinterpret<T, T_int>(ret);
  }
};

template <>
struct genericAtomicOperationImpl<unsigned long int, DeviceSum, 8> {
  using T = unsigned long int;
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceSum op)
  {
    using T_int = unsigned long long int;
    static_assert(sizeof(T) == sizeof(T_int), errmsg_cast);
    T_int ret = atomicAdd(reinterpret_cast<T_int*>(addr), type_reinterpret<T_int, T>(update_value));
    return type_reinterpret<T, T_int>(ret);
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
struct genericAtomicOperationImpl<long long int, DeviceSum, 8> {
  using T = long long int;
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceSum op)
  {
    using T_int = unsigned long long int;
    static_assert(sizeof(T) == sizeof(T_int), errmsg_cast);
    T_int ret = atomicAdd(reinterpret_cast<T_int*>(addr), type_reinterpret<T_int, T>(update_value));
    return type_reinterpret<T, T_int>(ret);
  }
};

template <>
struct genericAtomicOperationImpl<unsigned long int, DeviceMin, 8> {
  using T = unsigned long int;
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceMin op)
  {
    using T_int = unsigned long long int;
    static_assert(sizeof(T) == sizeof(T_int), errmsg_cast);
    T ret = atomicMin(reinterpret_cast<T_int*>(addr), type_reinterpret<T_int, T>(update_value));
    return type_reinterpret<T, T_int>(ret);
  }
};

template <>
struct genericAtomicOperationImpl<unsigned long int, DeviceMax, 8> {
  using T = unsigned long int;
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceMax op)
  {
    using T_int = unsigned long long int;
    static_assert(sizeof(T) == sizeof(T_int), errmsg_cast);
    T ret = atomicMax(reinterpret_cast<T_int*>(addr), type_reinterpret<T_int, T>(update_value));
    return type_reinterpret<T, T_int>(ret);
  }
};

template <typename T>
struct genericAtomicOperationImpl<T, DeviceAnd, 8> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceAnd op)
  {
    using T_int = unsigned long long int;
    static_assert(sizeof(T) == sizeof(T_int), errmsg_cast);
    T_int ret = atomicAnd(reinterpret_cast<T_int*>(addr), type_reinterpret<T_int, T>(update_value));
    return type_reinterpret<T, T_int>(ret);
  }
};

template <typename T>
struct genericAtomicOperationImpl<T, DeviceOr, 8> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceOr op)
  {
    using T_int = unsigned long long int;
    static_assert(sizeof(T) == sizeof(T_int), errmsg_cast);
    T_int ret = atomicOr(reinterpret_cast<T_int*>(addr), type_reinterpret<T_int, T>(update_value));
    return type_reinterpret<T, T_int>(ret);
  }
};

template <typename T>
struct genericAtomicOperationImpl<T, DeviceXor, 8> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceXor op)
  {
    using T_int = unsigned long long int;
    static_assert(sizeof(T) == sizeof(T_int), errmsg_cast);
    T_int ret = atomicXor(reinterpret_cast<T_int*>(addr), type_reinterpret<T_int, T>(update_value));
    return type_reinterpret<T, T_int>(ret);
  }
};

// -------------------------------------------------------------------------------------------------
// the implementation of `typesAtomicCASImpl`

template <typename T, size_t N = sizeof(T)>
struct typesAtomicCASImpl;

template <typename T>
struct typesAtomicCASImpl<T, 1> {
  __forceinline__ __device__ T operator()(T* addr, T const& compare, T const& update_value)
  {
    using T_int = unsigned int;

    T_int shift           = ((reinterpret_cast<size_t>(addr) & 3) * 8);
    T_int* address_uint32 = reinterpret_cast<T_int*>(addr - (reinterpret_cast<size_t>(addr) & 3));

    // the 'target_value' in `old` can be different from `compare`
    // because other thread may update the value
    // before fetching a value from `address_uint32` in this function
    T_int old = *address_uint32;
    T_int assumed;
    T target_value;
    uint8_t u_val = type_reinterpret<uint8_t, T>(update_value);

    do {
      assumed      = old;
      target_value = T((old >> shift) & 0xff);
      // have to compare `target_value` and `compare` before calling atomicCAS
      // the `target_value` in `old` can be different with `compare`
      if (target_value != compare) break;

      T_int new_value = (old & ~(0x000000ff << shift)) | (T_int(u_val) << shift);
      old             = atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return target_value;
  }
};

template <typename T>
struct typesAtomicCASImpl<T, 2> {
  __forceinline__ __device__ T operator()(T* addr, T const& compare, T const& update_value)
  {
    using T_int = unsigned int;

    bool is_32_align = (reinterpret_cast<size_t>(addr) & 2) ? false : true;
    T_int* address_uint32 =
      reinterpret_cast<T_int*>(reinterpret_cast<size_t>(addr) - (is_32_align ? 0 : 2));

    T_int old = *address_uint32;
    T_int assumed;
    T target_value;
    uint16_t u_val = type_reinterpret<uint16_t, T>(update_value);

    do {
      assumed      = old;
      target_value = (is_32_align) ? T(old & 0xffff) : T(old >> 16);
      if (target_value != compare) break;

      T_int new_value =
        (is_32_align) ? (old & 0xffff0000) | u_val : (old & 0xffff) | (T_int(u_val) << 16);
      old = atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return target_value;
  }
};

template <typename T>
struct typesAtomicCASImpl<T, 4> {
  __forceinline__ __device__ T operator()(T* addr, T const& compare, T const& update_value)
  {
    using T_int = unsigned int;

    T_int ret = atomicCAS(reinterpret_cast<T_int*>(addr),
                          type_reinterpret<T_int, T>(compare),
                          type_reinterpret<T_int, T>(update_value));
    return type_reinterpret<T, T_int>(ret);
  }
};

// 8 bytes atomic operation
template <typename T>
struct typesAtomicCASImpl<T, 8> {
  __forceinline__ __device__ T operator()(T* addr, T const& compare, T const& update_value)
  {
    using T_int = unsigned long long int;
    static_assert(sizeof(T) == sizeof(T_int), errmsg_cast);

    T_int ret = atomicCAS(reinterpret_cast<T_int*>(addr),
                          type_reinterpret<T_int, T>(compare),
                          type_reinterpret<T_int, T>(update_value));

    return type_reinterpret<T, T_int>(ret);
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
 * @param[in] update_value The value to be computed
 * @param[in] op  The binary operator used for compute
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
template <typename T, typename BinaryOp>
typename std::enable_if_t<std::is_arithmetic<T>::value, T> __forceinline__ __device__
genericAtomicOperation(T* address, T const& update_value, BinaryOp op)
{
  auto fun = raft::device_atomics::detail::genericAtomicOperationImpl<T, BinaryOp>{};
  return T(fun(address, update_value, op));
}

// specialization for bool types
template <typename BinaryOp>
__forceinline__ __device__ bool genericAtomicOperation(bool* address,
                                                       bool const& update_value,
                                                       BinaryOp op)
{
  using T = bool;
  // don't use underlying type to apply operation for bool
  auto fun = raft::device_atomics::detail::genericAtomicOperationImpl<T, BinaryOp>{};
  return T(fun(address, update_value, op));
}

}  // namespace raft

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
__forceinline__ __device__ T atomicAdd(T* address, T val)
{
  return raft::genericAtomicOperation(address, val, raft::device_atomics::detail::DeviceSum{});
}

/**
 * @brief Overloads for `atomicMin`
 *
 * reads the `old` located at the `address` in global or shared memory, computes the minimum of old
 * and val, and stores the result back to memory at the same address. These three operations are
 * performed in one atomic transaction.
 *
 * The supported types for `atomicMin` are: integers are floating point numbers.
 * CUDA natively supports `int`, `unsigned int`, `unsigned long long int`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomicMin(T* address, T val)
{
  return raft::genericAtomicOperation(address, val, raft::device_atomics::detail::DeviceMin{});
}

/**
 * @brief Overloads for `atomicMax`
 *
 * reads the `old` located at the `address` in global or shared memory, computes the maximum of old
 * and val, and stores the result back to memory at the same address. These three operations are
 * performed in one atomic transaction.
 *
 * The supported types for `atomicMax` are: integers are floating point numbers.
 * CUDA natively supports `int`, `unsigned int`, `unsigned long long int`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomicMax(T* address, T val)
{
  return raft::genericAtomicOperation(address, val, raft::device_atomics::detail::DeviceMax{});
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
__forceinline__ __device__ T atomicCAS(T* address, T compare, T val)
{
  return raft::device_atomics::detail::typesAtomicCASImpl<T>()(address, compare, val);
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
template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
__forceinline__ __device__ T atomicAnd(T* address, T val)
{
  return raft::genericAtomicOperation(address, val, raft::device_atomics::detail::DeviceAnd{});
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
template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
__forceinline__ __device__ T atomicOr(T* address, T val)
{
  return raft::genericAtomicOperation(address, val, raft::device_atomics::detail::DeviceOr{});
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
template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
__forceinline__ __device__ T atomicXor(T* address, T val)
{
  return raft::genericAtomicOperation(address, val, raft::device_atomics::detail::DeviceXor{});
}

/**
 * @brief: Warp aggregated atomic increment
 *
 * increments an atomic counter using all active threads in a warp. The return
 * value is the original value of the counter plus the rank of the calling
 * thread.
 *
 * The use of atomicIncWarp is a performance optimization. It can reduce the
 * amount of atomic memory traffic by a factor of 32.
 *
 * Adapted from:
 * https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
 *
 * @tparam          T An integral type
 * @param[in,out] ctr The address of old value
 *
 * @return The old value of the counter plus the rank of the calling thread.
 */
template <typename T                                                = unsigned int,
          typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
__device__ T atomicIncWarp(T* ctr)
{
  namespace cg = cooperative_groups;
  auto g       = cg::coalesced_threads();
  T warp_res;
  if (g.thread_rank() == 0) { warp_res = atomicAdd(ctr, static_cast<T>(g.size())); }
  return g.shfl(warp_res, 0) + g.thread_rank();
}
