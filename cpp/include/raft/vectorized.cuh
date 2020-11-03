/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <cuda_fp16.h>
#include "cuda_utils.cuh"

namespace raft {

template <typename Math, int VecLen>
struct IOType {};  // NOLINT
template <>
struct IOType<bool, 1> {
  static_assert(sizeof(bool) == sizeof(int8_t),
                "IOType bool size assumption failed");
  using Type = int8_t;  // NOLINT
};
template <>
struct IOType<bool, 2> {
  using Type = int16_t;  // NOLINT
};
template <>
struct IOType<bool, 4> {
  using Type = int32_t;  // NOLINT
};
template <>
struct IOType<bool, 8> {
  using Type = int2;  // NOLINT
};
template <>
struct IOType<bool, 16> {
  using Type = int4;  // NOLINT
};
template <>
struct IOType<int8_t, 1> {
  using Type = int8_t;  // NOLINT
};
template <>
struct IOType<int8_t, 2> {
  using Type = int16_t;  // NOLINT
};
template <>
struct IOType<int8_t, 4> {
  using Type = int32_t;  // NOLINT
};
template <>
struct IOType<int8_t, 8> {
  using Type = int2;  // NOLINT
};
template <>
struct IOType<int8_t, 16> {
  using Type = int4;  // NOLINT
};
template <>
struct IOType<uint8_t, 1> {
  using Type = uint8_t;  // NOLINT
};
template <>
struct IOType<uint8_t, 2> {
  using Type = uint16_t;  // NOLINT
};
template <>
struct IOType<uint8_t, 4> {
  using Type = uint32_t;  // NOLINT
};
template <>
struct IOType<uint8_t, 8> {
  using Type = uint2;  // NOLINT
};
template <>
struct IOType<uint8_t, 16> {
  using Type = uint4;  // NOLINT
};
template <>
struct IOType<int16_t, 1> {
  using Type = int16_t;  // NOLINT
};
template <>
struct IOType<int16_t, 2> {
  using Type = int32_t;  // NOLINT
};
template <>
struct IOType<int16_t, 4> {
  using Type = int2;  // NOLINT
};
template <>
struct IOType<int16_t, 8> {
  using Type = int4;  // NOLINT
};
template <>
struct IOType<uint16_t, 1> {
  using Type = uint16_t;  // NOLINT
};
template <>
struct IOType<uint16_t, 2> {
  using Type = uint32_t;  // NOLINT
};
template <>
struct IOType<uint16_t, 4> {
  using Type = uint2;  // NOLINT
};
template <>
struct IOType<uint16_t, 8> {
  using Type = uint4;  // NOLINT
};
template <>
struct IOType<__half, 1> {
  using Type = __half;  // NOLINT
};
template <>
struct IOType<__half, 2> {
  using Type = __half2;  // NOLINT
};
template <>
struct IOType<__half, 4> {
  using Type = uint2;  // NOLINT
};
template <>
struct IOType<__half, 8> {
  using Type = uint4;  // NOLINT
};
template <>
struct IOType<__half2, 1> {
  using Type = __half2;  // NOLINT
};
template <>
struct IOType<__half2, 2> {
  using Type = uint2;  // NOLINT
};
template <>
struct IOType<__half2, 4> {
  using Type = uint4;  // NOLINT
};
template <>
struct IOType<int32_t, 1> {
  using Type = int32_t;  // NOLINT
};
template <>
struct IOType<int32_t, 2> {
  using Type = uint2;  // NOLINT
};
template <>
struct IOType<int32_t, 4> {
  using Type = uint4;  // NOLINT
};
template <>
struct IOType<uint32_t, 1> {
  using Type = uint32_t;  // NOLINT
};
template <>
struct IOType<uint32_t, 2> {
  using Type = uint2;  // NOLINT
};
template <>
struct IOType<uint32_t, 4> {
  using Type = uint4;  // NOLINT
};
template <>
struct IOType<float, 1> {
  using Type = float;  // NOLINT
};
template <>
struct IOType<float, 2> {
  using Type = float2;  // NOLINT
};
template <>
struct IOType<float, 4> {
  using Type = float4;  // NOLINT
};
template <>
struct IOType<int64_t, 1> {
  using Type = int64_t;  // NOLINT
};
template <>
struct IOType<int64_t, 2> {
  using Type = uint4;  // NOLINT
};
template <>
struct IOType<uint64_t, 1> {
  using Type = uint64_t;  // NOLINT
};
template <>
struct IOType<uint64_t, 2> {
  using Type = uint4;  // NOLINT
};
// there's no guarantee that uint64_t and unsigned long long will be same types!
template <>
struct IOType<unsigned long long, 1> {  // NOLINT
  using Type = unsigned long long;  // NOLINT
};
// there's no guarantee that uint64_t and unsigned long long will be same types!
template <>
struct IOType<unsigned long long, 2> {  // NOLINT
  using Type = uint4;  // NOLINT
};
template <>
struct IOType<double, 1> {
  using Type = double;  // NOLINT
};
template <>
struct IOType<double, 2> {
  using Type = double2;  // NOLINT
};

/**
 * @struct TxN_t
 *
 * @brief Internal data structure that is used to define a facade for vectorized
 * loads/stores across the most common POD types. The goal of his file is to
 * provide with CUDA programmers, an easy way to have compiler issue vectorized
 * load or store instructions to memory (either global or shared). Vectorized
 * accesses to memory are important as they'll utilize its resources
 * efficiently,
 * when compared to their non-vectorized counterparts. Obviously, for whatever
 * reasons if one is unable to issue such vectorized operations, one can always
 * fallback to using POD types.
 *
 * Example demonstrating the use of load operations, performing math on such
 * loaded data and finally storing it back.
 * @code{.cu}
 * TxN_t<uint8_t,8> mydata1, mydata2;
 * int idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * mydata1.Ratio;
 * mydata1.load(ptr1, idx);
 * mydata2.load(ptr2, idx);
 * #pragma unroll
 * for(int i=0;i<mydata1.Ratio;++i) {
 *     mydata1.val.data[i] += mydata2.val.data[i];
 * }
 * mydata1.store(ptr1, idx);
 * @endcode
 *
 * By doing as above, the interesting thing is that the code effectively remains
 * almost the same, in case one wants to upgrade to TxN_t<uint16_t,16> type.
 * Only change required is to replace variable declaration appropriately.
 *
 * Obviously, it's caller's responsibility to take care of pointer alignment!
 *
 * @tparam math_ the data-type in which the compute/math needs to happen
 * @tparam veclen_ the number of 'math_' types to be loaded/stored per
 * instruction
 */
template <typename Math, int VecLen>
struct TxN_t {  // NOLINT
  /** underlying math data type */
  using math_t = Matht;  // NOLINT
  /** internal storage data type */
  using io_t = typename IOType<math_t, VecLen>::Type;  // NOLINT

  /** defines the number of 'math_t' types stored by this struct */
  static const int Ratio = VecLen;  // NOLINT

  union {
    /** the vectorized data that is used for subsequent operations */
    math_t data[Ratio];
    /** internal data used to ensure vectorized loads/stores */
    io_t internal;
  } val;

  ///@todo: add default constructor

  /**
   * @brief Fill the contents of this structure with a constant value
   * @param _val the constant to be filled
   */
  DI void fill(math_t _val) {
#pragma unroll
    for (int i = 0; i < Ratio; ++i) {
      val.data[i] = _val;
    }
  }

  ///@todo: how to handle out-of-bounds!!?

  /**
   * @defgroup LoadsStores Global/Shared vectored loads or stores
   *
   * @brief Perform vectored loads/stores on this structure
   * @tparam idx_t index data type
   * @param ptr base pointer from where to load (or store) the data. It must
   *  be aligned to 'sizeof(io_t)'!
   * @param idx the offset from the base pointer which will be loaded
   *  (or stored) by the current thread. This must be aligned to 'Ratio'!
   *
   * @note: In case of loads, after a successful execution, the val.data will
   *  be populated with the desired data loaded from the pointer location. In
   * case of stores, the data in the val.data will be stored to that location.
   * @{
   */
  template <typename IdxT = int>
  DI void load(const math_t *ptr, IdxT idx) {
    const io_t *bptr = reinterpret_cast<const io_t *>(&ptr[idx]);
    val.internal = __ldg(bptr);
  }

  template <typename IdxT = int>
  DI void load(math_t *ptr, IdxT idx) {
    io_t *bptr = reinterpret_cast<io_t *>(&ptr[idx]);
    val.internal = *bptr;
  }

  template <typename IdxT = int>
  DI void store(math_t *ptr, IdxT idx) {
    io_t *bptr = reinterpret_cast<io_t *>(&ptr[idx]);
    *bptr = val.internal;
  }
  /** @} */
};

/** this is just to keep the compiler happy! */
template <typename Math>
struct TxN_t<Math, 0> {
  using math_t = Math;  // NOLINT
  static const int Ratio = 1;  // NOLINT

  union {
    math_t data[1];  // NOLINT
  } val;

  DI void fill(math_t _val) {}
  template <typename IdxT = int>
  DI void load(const math_t *ptr, IdxT idx) {}
  template <typename IdxT = int>
  DI void load(math_t *ptr, IdxT idx) {}
  template <typename IdxT = int>
  DI void store(math_t *ptr, IdxT idx) {}
};

}  // namespace raft
