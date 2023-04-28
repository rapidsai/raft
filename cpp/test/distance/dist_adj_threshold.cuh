/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cstdint>  // uint8_t

namespace raft::distance {

template <typename AccT, typename DataT, typename OutT, typename Index>
struct threshold_final_op {
  DataT threshold_val;

  __device__ __host__ threshold_final_op() noexcept : threshold_val(0.0) {}
  __device__ __host__ threshold_final_op(DataT val) noexcept : threshold_val(val) {}
  __device__ __host__ OutT operator()(AccT d_val, Index g_idx) const noexcept
  {
    return d_val <= threshold_val;
  }
};

using threshold_float  = threshold_final_op<float, float, uint8_t, int>;
using threshold_double = threshold_final_op<double, double, uint8_t, int>;

}  // namespace raft::distance
