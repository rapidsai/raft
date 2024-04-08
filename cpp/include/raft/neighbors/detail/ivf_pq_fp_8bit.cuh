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

#include <raft/core/cudart_utils.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/detail/select_k.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/device_atomics.cuh>
#include <raft/util/device_loads_stores.cuh>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cub/cub.cuh>
#include <cuda_fp16.h>

#include <optional>

namespace raft::neighbors::ivf_pq::detail {

/** 8-bit floating-point storage type.
 *
 * This is a custom type for the current IVF-PQ implementation. No arithmetic operations defined
 * only conversion to and from fp32. This type is unrelated to the proposed FP8 specification.
 */
template <uint32_t ExpBits, bool Signed>
struct fp_8bit {
  static_assert(ExpBits + uint8_t{Signed} <= 8, "The type does not fit in 8 bits.");
  constexpr static uint32_t ExpMask = (1u << (ExpBits - 1u)) - 1u;  // NOLINT
  constexpr static uint32_t ValBits = 8u - ExpBits;                 // NOLINT

 public:
  uint8_t bitstring;

  HDI explicit fp_8bit(uint8_t bs) : bitstring(bs) {}
  HDI explicit fp_8bit(float fp) : fp_8bit(float2fp_8bit(fp).bitstring) {}
  HDI auto operator=(float fp) -> fp_8bit<ExpBits, Signed>&
  {
    bitstring = float2fp_8bit(fp).bitstring;
    return *this;
  }
  HDI explicit operator float() const { return fp_8bit2float(*this); }
  HDI explicit operator half() const { return fp_8bit2half(*this); }

 private:
  static constexpr float kMin = 1.0f / float(1u << ExpMask);
  static constexpr float kMax = float(1u << (ExpMask + 1)) * (2.0f - 1.0f / float(1u << ValBits));

  static HDI auto float2fp_8bit(float v) -> fp_8bit<ExpBits, Signed>
  {
    if constexpr (Signed) {
      auto u = fp_8bit<ExpBits, false>(std::abs(v)).bitstring;
      u      = (u & 0xfeu) | uint8_t{v < 0};  // set the sign bit
      return fp_8bit<ExpBits, true>(u);
    } else {
      // sic! all small and negative numbers are truncated to zero.
      if (v < kMin) { return fp_8bit<ExpBits, false>{static_cast<uint8_t>(0)}; }
      // protect from overflow
      if (v >= kMax) { return fp_8bit<ExpBits, false>{static_cast<uint8_t>(0xffu)}; }
      // the rest of possible float values should be within the normalized range
      return fp_8bit<ExpBits, false>{static_cast<uint8_t>(
        (*reinterpret_cast<uint32_t*>(&v) + (ExpMask << 23u) - 0x3f800000u) >> (15u + ExpBits))};
    }
  }

  static HDI auto fp_8bit2float(const fp_8bit<ExpBits, Signed>& v) -> float
  {
    uint32_t u = v.bitstring;
    if constexpr (Signed) {
      u &= ~1;  // zero the sign bit
    }
    float r;
    constexpr uint32_t kBase32       = (0x3f800000u | (0x00400000u >> ValBits)) - (ExpMask << 23);
    *reinterpret_cast<uint32_t*>(&r) = kBase32 + (u << (15u + ExpBits));
    if constexpr (Signed) {  // recover the sign bit
      if (v.bitstring & 1) { r = -r; }
    }
    return r;
  }

  static HDI auto fp_8bit2half(const fp_8bit<ExpBits, Signed>& v) -> half
  {
    uint16_t u = v.bitstring;
    if constexpr (Signed) {
      u &= ~1;  // zero the sign bit
    }
    half r;
    constexpr uint16_t kBase16       = (0x3c00u | (0x0200u >> ValBits)) - (ExpMask << 10);
    *reinterpret_cast<uint16_t*>(&r) = kBase16 + (u << (2u + ExpBits));
    if constexpr (Signed) {  // recover the sign bit
      if (v.bitstring & 1) { r = -r; }
    }
    return r;
  }
};

}  // namespace raft::neighbors::ivf_pq::detail
