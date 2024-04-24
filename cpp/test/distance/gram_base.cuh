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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/distance/kernels.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <iostream>
#include <memory>

namespace raft {
namespace distance {
namespace kernels {

// Get the offset of element [i,k].
HDI int get_offset(int i, int k, int ld, bool is_row_major)
{
  return is_row_major ? i * ld + k : i + k * ld;
}

// Calculate the Gram matrix on the host.
template <typename math_t>
void naiveGramMatrixKernel(int n1,
                           int n2,
                           int n_cols,
                           const rmm::device_uvector<math_t>& x1,
                           const rmm::device_uvector<math_t>& x2,
                           math_t* gram_host,
                           int ld1,
                           int ld2,
                           int ld_out,
                           bool is_row_major,
                           KernelParams kernel,
                           cudaStream_t stream,
                           const raft::resources& handle)
{
  std::vector<math_t> x1_host(x1.size());
  raft::update_host(x1_host.data(), x1.data(), x1.size(), stream);
  std::vector<math_t> x2_host(x2.size());
  raft::update_host(x2_host.data(), x2.data(), x2.size(), stream);
  resource::sync_stream(handle, stream);

  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      float d = 0;
      for (int k = 0; k < n_cols; k++) {
        if (kernel.kernel == KernelType::RBF) {
          math_t diff = x1_host[get_offset(i, k, ld1, is_row_major)] -
                        x2_host[get_offset(j, k, ld2, is_row_major)];
          d += diff * diff;
        } else {
          d += x1_host[get_offset(i, k, ld1, is_row_major)] *
               x2_host[get_offset(j, k, ld2, is_row_major)];
        }
      }
      int idx  = get_offset(i, j, ld_out, is_row_major);
      math_t v = 0;
      switch (kernel.kernel) {
        case (KernelType::LINEAR): gram_host[idx] = d; break;
        case (KernelType::POLYNOMIAL):
          v              = kernel.gamma * d + kernel.coef0;
          gram_host[idx] = std::pow(v, kernel.degree);
          break;
        case (KernelType::TANH): gram_host[idx] = std::tanh(kernel.gamma * d + kernel.coef0); break;
        case (KernelType::RBF): gram_host[idx] = exp(-kernel.gamma * d); break;
      }
    }
  }
}

}  // namespace kernels
}  // namespace distance
}  // namespace raft
