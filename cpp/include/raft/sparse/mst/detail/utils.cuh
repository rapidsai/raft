
/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <rmm/thrust_rmm_allocator.h>

namespace raft {
namespace mst {
namespace detail {

// TODO make this work in 64bit
__device__ int get_1D_idx() { return blockIdx.x * blockDim.x + threadIdx.x; }

//FIXME this should live elswhere
template <typename T>
void printv(rmm::device_vector<T>& vec) {
  std::cout.precision(15);
  std::cout << "Size = " << vec.size() << std::endl;
  thrust::copy(vec.begin(), vec.end(),
               std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl << std::endl;
}

}  // namespace detail
}  // namespace mst
}  // namespace raft
