/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cub/cub.cuh>
#include <raft/cuda_utils.cuh>

namespace raft {
namespace matrix {
namespace detail {

// Computes the argmax(d_in) column-wise in a DxN matrix
template <typename T, int TPB>
__global__ void argmaxKernel(const T *d_in, int D, int N, T *argmax) {
  typedef cub::BlockReduce<cub::KeyValuePair<int, T>, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // compute maxIndex=argMax  index for column
  using KVP = cub::KeyValuePair<int, T>;
  int rowStart = blockIdx.x * D;
  KVP thread_data(-1, -raft::myInf<T>());

  for (int i = threadIdx.x; i < D; i += TPB) {
    int idx = rowStart + i;
    thread_data = cub::ArgMax()(thread_data, KVP(i, d_in[idx]));
  }

  auto maxKV = BlockReduce(temp_storage).Reduce(thread_data, cub::ArgMax());

  if (threadIdx.x == 0) {
    argmax[blockIdx.x] = maxKV.key;
  }
}

// Utility kernel needed for signFlip.
// Computes the argmax(abs(d_in)) column-wise in a DxN matrix followed by
// flipping the sign if the |max| value for each column is negative.
template <typename T, int TPB>
__global__ void signFlipKernel(T *d_in, int D, int N) {
  typedef cub::BlockReduce<cub::KeyValuePair<int, T>, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // compute maxIndex=argMax (with abs()) index for column
  using KVP = cub::KeyValuePair<int, T>;
  int rowStart = blockIdx.x * D;
  KVP thread_data(0, 0);
  for (int i = threadIdx.x; i < D; i += TPB) {
    int idx = rowStart + i;
    thread_data = cub::ArgMax()(thread_data, KVP(idx, abs(d_in[idx])));
  }
  auto maxKV = BlockReduce(temp_storage).Reduce(thread_data, cub::ArgMax());

  // flip column sign if d_in[maxIndex] < 0
  __shared__ bool need_sign_flip;
  if (threadIdx.x == 0) {
    need_sign_flip = d_in[maxKV.key] < T(0);
  }
  __syncthreads();

  if (need_sign_flip) {
    for (int i = threadIdx.x; i < D; i += TPB) {
      int idx = rowStart + i;
      d_in[idx] = -d_in[idx];
    }
  }
}

}  // end namespace detail
}  // end namespace matrix
}  // end namespace raft