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

#include <raft/distance/distance_types.hpp>
#include <raft/spatial/knn/knn.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

namespace raft::spatial::knn {
template <typename DataType, typename AccT>
__global__ void naiveDistanceKernel(float* dist,
                                    const DataType* x,
                                    const DataType* y,
                                    size_t m,
                                    size_t n,
                                    size_t k,
                                    raft::distance::DistanceType type)
{
  size_t midx = threadIdx.x + blockIdx.x * blockDim.x;
  if (midx >= m) return;
  for (size_t nidx = threadIdx.y + blockIdx.y * blockDim.y; nidx < n;
       nidx += blockDim.y * gridDim.y) {
    AccT acc = AccT(0);
    for (size_t i = 0; i < k; ++i) {
      size_t xidx = i + midx * k;
      size_t yidx = i + nidx * k;
      if (type == raft::distance::DistanceType::InnerProduct) {
        acc += x[xidx] * y[yidx];
      } else {
        AccT diff = x[xidx] - y[yidx];
        acc += diff * diff;
      }
    }
    float dist_val = (float)acc;
    if (type == raft::distance::DistanceType::L2SqrtExpanded ||
        type == raft::distance::DistanceType::L2SqrtUnexpanded)
      dist_val = raft::mySqrt(dist_val);
    dist[midx * n + nidx] = dist_val;
  }
}

/**
 * TODO: either replace this with brute_force_knn or with distance+select_k
 *       when either distance or brute_force_knn support 8-bit int inputs.
 */
template <typename DataType, typename AccT>
void naiveBfKnn(float* dist_topk,
                int64_t* indices_topk,
                const DataType* x,
                const DataType* y,
                size_t n_inputs,
                size_t input_len,
                size_t dim,
                uint32_t k,
                raft::distance::DistanceType type,
                DataType metric_arg = 2.0f,
                cudaStream_t stream = 0)
{
  dim3 block_dim(16, 32, 1);
  // maximum reasonable grid size in `y` direction
  uint16_t grid_y =
    static_cast<uint16_t>(std::min<size_t>(raft::ceildiv<size_t>(input_len, block_dim.y), 32768));

  // bound the memory used by this function
  size_t max_batch_size =
    std::min(n_inputs, raft::ceildiv<size_t>(size_t(1) << size_t(27), input_len));
  rmm::device_uvector<float> dist(max_batch_size * input_len, stream);

  for (size_t offset = 0; offset < n_inputs; offset += max_batch_size) {
    size_t batch_size = std::min(max_batch_size, n_inputs - offset);
    dim3 grid_dim(raft::ceildiv<size_t>(batch_size, block_dim.x), grid_y, 1);

    naiveDistanceKernel<DataType, AccT><<<grid_dim, block_dim, 0, stream>>>(
      dist.data(), x + offset * dim, y, batch_size, input_len, dim, type);

    select_k<int64_t, float>(dist.data(),
                             nullptr,
                             batch_size,
                             input_len,
                             dist_topk + offset * k,
                             indices_topk + offset * k,
                             type != raft::distance::DistanceType::InnerProduct,
                             static_cast<int>(k),
                             stream,
                             SelectKAlgo::WARP_SORT);
  }
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

}  // namespace raft::spatial::knn
