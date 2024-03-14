/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <raft/core/operators.hpp>
#include <raft/util/cuda_dev_essentials.cuh>

#include <stdint.h>

namespace raft {

/** True CUDA alignment of a type (adapted from CUB) */
template <typename T>
struct cuda_alignment {
  struct Pad {
    T val;
    char byte;
  };

  static constexpr int bytes = sizeof(Pad) - sizeof(T);
};

template <typename LargeT, typename UnitT>
struct is_multiple {
  static constexpr int large_align_bytes = cuda_alignment<LargeT>::bytes;
  static constexpr int unit_align_bytes  = cuda_alignment<UnitT>::bytes;
  static constexpr bool value =
    (sizeof(LargeT) % sizeof(UnitT) == 0) && (large_align_bytes % unit_align_bytes == 0);
};

template <typename LargeT, typename UnitT>
inline constexpr bool is_multiple_v = is_multiple<LargeT, UnitT>::value;

/** apply a warp-wide fence (useful from Volta+ archs) */
DI void warpFence()
{
#if __CUDA_ARCH__ >= 700
  __syncwarp();
#endif
}

/** warp-wide any boolean aggregator */
DI bool any(bool inFlag, uint32_t mask = 0xffffffffu)
{
#if CUDART_VERSION >= 9000
  inFlag = __any_sync(mask, inFlag);
#else
  inFlag = __any(inFlag);
#endif
  return inFlag;
}

/** warp-wide all boolean aggregator */
DI bool all(bool inFlag, uint32_t mask = 0xffffffffu)
{
#if CUDART_VERSION >= 9000
  inFlag = __all_sync(mask, inFlag);
#else
  inFlag = __all(inFlag);
#endif
  return inFlag;
}

/** For every thread in the warp, set the corresponding bit to the thread's flag value.  */
DI uint32_t ballot(bool inFlag, uint32_t mask = 0xffffffffu)
{
#if CUDART_VERSION >= 9000
  return __ballot_sync(mask, inFlag);
#else
  return __ballot(inFlag);
#endif
}

template <typename T>
struct is_shuffleable {
  static constexpr bool value =
    std::is_same_v<T, int> || std::is_same_v<T, unsigned int> || std::is_same_v<T, long> ||
    std::is_same_v<T, unsigned long> || std::is_same_v<T, long long> ||
    std::is_same_v<T, unsigned long long> || std::is_same_v<T, float> || std::is_same_v<T, double>;
};

template <typename T>
inline constexpr bool is_shuffleable_v = is_shuffleable<T>::value;

/**
 * @brief Shuffle the data inside a warp
 * @tparam T the data type
 * @param val value to be shuffled
 * @param srcLane lane from where to shuffle
 * @param width lane width
 * @param mask mask of participating threads (Volta+)
 * @return the shuffled data
 */
template <typename T>
DI std::enable_if_t<is_shuffleable_v<T>, T> shfl(T val,
                                                 int srcLane,
                                                 int width     = WarpSize,
                                                 uint32_t mask = 0xffffffffu)
{
#if CUDART_VERSION >= 9000
  return __shfl_sync(mask, val, srcLane, width);
#else
  return __shfl(val, srcLane, width);
#endif
}

/// Overload of shfl for data types not supported by the CUDA intrinsics
template <typename T>
DI std::enable_if_t<!is_shuffleable_v<T>, T> shfl(T val,
                                                  int srcLane,
                                                  int width     = WarpSize,
                                                  uint32_t mask = 0xffffffffu)
{
  using UnitT =
    std::conditional_t<is_multiple_v<T, int>,
                       unsigned int,
                       std::conditional_t<is_multiple_v<T, short>, unsigned short, unsigned char>>;

  constexpr int n_words = sizeof(T) / sizeof(UnitT);

  T output;
  UnitT* output_alias = reinterpret_cast<UnitT*>(&output);
  UnitT* input_alias  = reinterpret_cast<UnitT*>(&val);

  unsigned int shuffle_word;
  shuffle_word    = shfl((unsigned int)input_alias[0], srcLane, width, mask);
  output_alias[0] = shuffle_word;

#pragma unroll
  for (int i = 1; i < n_words; ++i) {
    shuffle_word    = shfl((unsigned int)input_alias[i], srcLane, width, mask);
    output_alias[i] = shuffle_word;
  }

  return output;
}

/**
 * @brief Shuffle the data inside a warp from lower lane IDs
 * @tparam T the data type
 * @param val value to be shuffled
 * @param delta lower lane ID delta from where to shuffle
 * @param width lane width
 * @param mask mask of participating threads (Volta+)
 * @return the shuffled data
 */
template <typename T>
DI std::enable_if_t<is_shuffleable_v<T>, T> shfl_up(T val,
                                                    int delta,
                                                    int width     = WarpSize,
                                                    uint32_t mask = 0xffffffffu)
{
#if CUDART_VERSION >= 9000
  return __shfl_up_sync(mask, val, delta, width);
#else
  return __shfl_up(val, delta, width);
#endif
}

/// Overload of shfl_up for data types not supported by the CUDA intrinsics
template <typename T>
DI std::enable_if_t<!is_shuffleable_v<T>, T> shfl_up(T val,
                                                     int delta,
                                                     int width     = WarpSize,
                                                     uint32_t mask = 0xffffffffu)
{
  using UnitT =
    std::conditional_t<is_multiple_v<T, int>,
                       unsigned int,
                       std::conditional_t<is_multiple_v<T, short>, unsigned short, unsigned char>>;

  constexpr int n_words = sizeof(T) / sizeof(UnitT);

  T output;
  UnitT* output_alias = reinterpret_cast<UnitT*>(&output);
  UnitT* input_alias  = reinterpret_cast<UnitT*>(&val);

  unsigned int shuffle_word;
  shuffle_word    = shfl_up((unsigned int)input_alias[0], delta, width, mask);
  output_alias[0] = shuffle_word;

#pragma unroll
  for (int i = 1; i < n_words; ++i) {
    shuffle_word    = shfl_up((unsigned int)input_alias[i], delta, width, mask);
    output_alias[i] = shuffle_word;
  }

  return output;
}

/**
 * @brief Shuffle the data inside a warp
 * @tparam T the data type
 * @param val value to be shuffled
 * @param laneMask mask to be applied in order to perform xor shuffle
 * @param width lane width
 * @param mask mask of participating threads (Volta+)
 * @return the shuffled data
 */
template <typename T>
DI std::enable_if_t<is_shuffleable_v<T>, T> shfl_xor(T val,
                                                     int laneMask,
                                                     int width     = WarpSize,
                                                     uint32_t mask = 0xffffffffu)
{
#if CUDART_VERSION >= 9000
  return __shfl_xor_sync(mask, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}

/// Overload of shfl_xor for data types not supported by the CUDA intrinsics
template <typename T>
DI std::enable_if_t<!is_shuffleable_v<T>, T> shfl_xor(T val,
                                                      int laneMask,
                                                      int width     = WarpSize,
                                                      uint32_t mask = 0xffffffffu)
{
  using UnitT =
    std::conditional_t<is_multiple_v<T, int>,
                       unsigned int,
                       std::conditional_t<is_multiple_v<T, short>, unsigned short, unsigned char>>;

  constexpr int n_words = sizeof(T) / sizeof(UnitT);

  T output;
  UnitT* output_alias = reinterpret_cast<UnitT*>(&output);
  UnitT* input_alias  = reinterpret_cast<UnitT*>(&val);

  unsigned int shuffle_word;
  shuffle_word    = shfl_xor((unsigned int)input_alias[0], laneMask, width, mask);
  output_alias[0] = shuffle_word;

#pragma unroll
  for (int i = 1; i < n_words; ++i) {
    shuffle_word    = shfl_xor((unsigned int)input_alias[i], laneMask, width, mask);
    output_alias[i] = shuffle_word;
  }

  return output;
}

}  // namespace raft