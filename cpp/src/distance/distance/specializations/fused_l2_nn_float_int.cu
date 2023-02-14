/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <raft/core/kvp.hpp>
#include <raft/distance/fused_l2_nn.cuh>

namespace raft {
namespace distance {

template void fusedL2NNMinReduce<float, raft::KeyValuePair<int, float>, int>(
  raft::KeyValuePair<int, float>* min,
  const float* x,
  const float* y,
  const float* xn,
  const float* yn,
  int m,
  int n,
  int k,
  void* workspace,
  bool sqrt,
  bool initOutBuffer,
  cudaStream_t stream);
template void fusedL2NNMinReduce<float, float, int>(float* min,
                                                    const float* x,
                                                    const float* y,
                                                    const float* xn,
                                                    const float* yn,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    void* workspace,
                                                    bool sqrt,
                                                    bool initOutBuffer,
                                                    cudaStream_t stream);

}  // namespace distance
}  // namespace raft
