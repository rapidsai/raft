/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <raft/spatial/knn/detail/faiss_select/Float16.cuh>

//
// Templated wrappers to express math for different scalar and vector
// types, so kernels can have the same written form but can operate
// over half and float, and on vector types transparently
//

namespace raft::spatial::knn::detail::faiss_select {

template <typename T>
struct Math {
  typedef T ScalarType;

  static inline __device__ T add(T a, T b) { return a + b; }

  static inline __device__ T sub(T a, T b) { return a - b; }

  static inline __device__ T mul(T a, T b) { return a * b; }

  static inline __device__ T neg(T v) { return -v; }

  static inline __device__ bool lt(T a, T b) { return a < b; }

  static inline __device__ bool gt(T a, T b) { return a > b; }

  static inline __device__ bool eq(T a, T b) { return a == b; }

  static inline __device__ T zero() { return (T)0; }
};

template <>
struct Math<half> {
  typedef half ScalarType;

  static inline __device__ half add(half a, half b)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    return __hadd(a, b);
#else
    return __float2half(__half2float(a) + __half2float(b));
#endif
  }

  static inline __device__ half sub(half a, half b)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    return __hsub(a, b);
#else
    return __float2half(__half2float(a) - __half2float(b));
#endif
  }

  static inline __device__ half mul(half a, half b)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    return __hmul(a, b);
#else
    return __float2half(__half2float(a) * __half2float(b));
#endif
  }

  static inline __device__ half neg(half v)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    return __hneg(v);
#else
    return __float2half(-__half2float(v));
#endif
  }

  static inline __device__ bool lt(half a, half b)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    return __hlt(a, b);
#else
    return __half2float(a) < __half2float(b);
#endif
  }

  static inline __device__ bool gt(half a, half b)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    return __hgt(a, b);
#else
    return __half2float(a) > __half2float(b);
#endif
  }

  static inline __device__ bool eq(half a, half b)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    return __heq(a, b);
#else
    return __half2float(a) == __half2float(b);
#endif
  }

  static inline __device__ half zero()
  {
#if CUDA_VERSION >= 9000
    return 0;
#else
    half h;
    h.x = 0;
    return h;
#endif
  }
};

template <>
struct Math<half2> {
  typedef half ScalarType;

  static inline __device__ half2 add(half2 a, half2 b)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    return __hadd2(a, b);
#else
    float2 af = __half22float2(a);
    float2 bf = __half22float2(b);

    af.x += bf.x;
    af.y += bf.y;

    return __float22half2_rn(af);
#endif
  }

  static inline __device__ half2 sub(half2 a, half2 b)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    return __hsub2(a, b);
#else
    float2 af = __half22float2(a);
    float2 bf = __half22float2(b);

    af.x -= bf.x;
    af.y -= bf.y;

    return __float22half2_rn(af);
#endif
  }

  static inline __device__ half2 add(half2 a, half b)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    half2 b2 = __half2half2(b);
    return __hadd2(a, b2);
#else
    float2 af = __half22float2(a);
    float bf  = __half2float(b);

    af.x += bf;
    af.y += bf;

    return __float22half2_rn(af);
#endif
  }

  static inline __device__ half2 sub(half2 a, half b)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    half2 b2 = __half2half2(b);
    return __hsub2(a, b2);
#else
    float2 af = __half22float2(a);
    float bf  = __half2float(b);

    af.x -= bf;
    af.y -= bf;

    return __float22half2_rn(af);
#endif
  }

  static inline __device__ half2 mul(half2 a, half2 b)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    return __hmul2(a, b);
#else
    float2 af = __half22float2(a);
    float2 bf = __half22float2(b);

    af.x *= bf.x;
    af.y *= bf.y;

    return __float22half2_rn(af);
#endif
  }

  static inline __device__ half2 mul(half2 a, half b)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    half2 b2 = __half2half2(b);
    return __hmul2(a, b2);
#else
    float2 af = __half22float2(a);
    float bf  = __half2float(b);

    af.x *= bf;
    af.y *= bf;

    return __float22half2_rn(af);
#endif
  }

  static inline __device__ half2 neg(half2 v)
  {
#ifdef FAISS_USE_FULL_FLOAT16
    return __hneg2(v);
#else
    float2 vf = __half22float2(v);
    vf.x      = -vf.x;
    vf.y      = -vf.y;

    return __float22half2_rn(vf);
#endif
  }

  // not implemented for vector types
  // static inline __device__ bool lt(half2 a, half2 b);
  // static inline __device__ bool gt(half2 a, half2 b);
  // static inline __device__ bool eq(half2 a, half2 b);

  static inline __device__ half2 zero() { return __half2half2(Math<half>::zero()); }
};
}  // namespace raft::spatial::knn::detail::faiss_select
