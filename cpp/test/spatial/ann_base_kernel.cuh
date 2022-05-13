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

#include <raft/cuda_utils.cuh>
#include <raft/distance/distance_type.hpp>
#include <raft/spatial/knn/detail/selection_faiss.cuh>

#include <rmm/device_uvector.hpp>

namespace raft {
namespace spatial {
namespace knn {
template <typename DataType, typename AccT>
__global__ void naiveDistanceKernel(float* dist,
                                    int64_t* indices,
                                    const DataType* x,
                                    const DataType* y,
                                    int m,
                                    int n,
                                    int k,
                                    raft::distance::DistanceType type,
                                    bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  AccT acc = AccT(0);
  for (int i = 0; i < k; ++i) {
    int xidx = isRowMajor ? i + midx * k : i * m + midx;
    int yidx = isRowMajor ? i + nidx * k : i * n + nidx;
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
  int outidx      = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx]    = dist_val;
  indices[outidx] = outidx;  // This is required because of the select_k API.
}

// currently using this naive kernel as FAISS & fusedL2kNN doesn't support 8-bit
template <typename DataType, typename AccT>
void naiveBfKnn(float* dist_topk,
                int64_t* indices_topk,
                const DataType* x,
                const DataType* y,
                int m,
                int n,
                int k,
                int numOfNN,
                raft::distance::DistanceType type,
                bool isRowMajor,
                DataType metric_arg = 2.0f,
                cudaStream_t stream = 0)
{
  static const dim3 TPB(16, 32, 1);
  dim3 nblks(raft::ceildiv(m, (int)TPB.x), raft::ceildiv(n, (int)TPB.y), 1);

  rmm::device_uvector<float> dist(m * n, stream);
  rmm::device_uvector<int64_t> indices(m * n, stream);
  naiveDistanceKernel<DataType, AccT>
    <<<nblks, TPB, 0, stream>>>(dist.data(), indices.data(), x, y, m, n, k, type, isRowMajor);
  detail::select_k(
    dist.data(), indices.data(), m, n, dist_topk, indices_topk, true, numOfNN, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}
}  // namespace knn
}  // namespace spatial
}  // namespace raft
