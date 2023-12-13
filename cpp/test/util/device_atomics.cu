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

#include <algorithm>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_atomics.cuh>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace raft {

RAFT_KERNEL test_atomic_inc_warp_kernel(int* counter, int* out_array)
{
  int global_tid                    = blockDim.x * blockIdx.x + threadIdx.x;
  out_array[atomicIncWarp(counter)] = global_tid;
}

TEST(Raft, AtomicIncWarp)
{
  const int num_blocks        = 1024;
  const int threads_per_block = 1024;
  const int num_elts          = num_blocks * threads_per_block;

  rmm::cuda_stream_pool pool{1};
  auto s = pool.get_stream();

  rmm::device_scalar<int> counter{0, s};
  rmm::device_uvector<int> out_device{num_elts, s};
  std::array<int, num_elts> out_host{0};

  // Write all 1M thread indices to a unique location in `out_device`
  test_atomic_inc_warp_kernel<<<num_blocks, threads_per_block, 0, s>>>(counter.data(),
                                                                       out_device.data());
  // Copy data to host
  RAFT_CUDA_TRY(cudaMemcpyAsync(out_host.data(),
                                (const void*)out_device.data(),
                                num_elts * sizeof(int),
                                cudaMemcpyDeviceToHost,
                                s));

  // Check that count is correct and that each thread index is contained in the
  // array exactly once.
  ASSERT_EQ(num_elts, counter.value(s));  // NB: accessing the counter synchronizes `s`
  std::sort(out_host.begin(), out_host.end());
  for (int i = 0; i < num_elts; ++i) {
    ASSERT_EQ(i, out_host[i]);
  }
}

}  // namespace raft
