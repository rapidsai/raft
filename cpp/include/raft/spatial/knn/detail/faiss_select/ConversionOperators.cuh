/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <raft/spatial/knn/detail/faiss_select/Float16.cuh>

#include <cuda.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace raft::spatial::knn::detail::faiss_select {

//
// Conversion utilities
//

template <typename From, typename To>
struct Convert {
  inline __device__ To operator()(From v) const { return (To)v; }
};

template <>
struct Convert<float, half> {
  inline __device__ half operator()(float v) const { return __float2half(v); }
};

template <>
struct Convert<half, float> {
  inline __device__ float operator()(half v) const { return __half2float(v); }
};

template <typename T>
struct ConvertTo {
};

template <>
struct ConvertTo<float> {
  static inline __device__ float to(float v) { return v; }
  static inline __device__ float to(half v) { return __half2float(v); }
};

template <>
struct ConvertTo<float2> {
  static inline __device__ float2 to(float2 v) { return v; }
  static inline __device__ float2 to(half2 v) { return __half22float2(v); }
};

template <>
struct ConvertTo<float4> {
  static inline __device__ float4 to(float4 v) { return v; }
  static inline __device__ float4 to(Half4 v) { return half4ToFloat4(v); }
};

template <>
struct ConvertTo<half> {
  static inline __device__ half to(float v) { return __float2half(v); }
  static inline __device__ half to(half v) { return v; }
};

template <>
struct ConvertTo<half2> {
  static inline __device__ half2 to(float2 v) { return __float22half2_rn(v); }
  static inline __device__ half2 to(half2 v) { return v; }
};

template <>
struct ConvertTo<Half4> {
  static inline __device__ Half4 to(float4 v) { return float4ToHalf4(v); }
  static inline __device__ Half4 to(Half4 v) { return v; }
};
}  // namespace raft::spatial::knn::detail::faiss_select
