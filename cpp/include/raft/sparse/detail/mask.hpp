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

#include <cstdint>

namespace raft {
    namespace detail {

        /**
         * Zeros the bit at location h in a one-hot encoded 32-bit int array
         */
        __device__ __host__ inline void _zero_bit(std::uint32_t* arr, std::uint32_t h)
        {
            int bit = h % 32;
            int idx = h / 32;

            std::uint32_t assumed;
            std::uint32_t old = arr[idx];
            do {
                assumed = old;
                old     = atomicCAS(arr + idx, assumed, assumed & ~(1 << bit));
            } while (assumed != old);
        }

        /**
         * Returns whether or not bit at location h is nonzero in a one-hot
         * encoded 32-bit in array.
         */
        __device__ __host__ inline bool _get_val(std::uint32_t* arr, std::uint32_t h)
        {
            int bit = h % 32;
            int idx = h / 32;
            return (arr[idx] & (1 << bit)) > 0;
        }

    }
}

