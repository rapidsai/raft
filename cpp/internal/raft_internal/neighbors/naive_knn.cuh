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

#pragma once

#include <raft/distance/distance_types.hpp>
#include <raft/matrix/detail/select_k.cuh>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <raft/util/cuda_utils.cuh>

#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace raft::neighbors {

template <typename EvalT, typename DataT, typename IdxT>
RAFT_KERNEL naive_distance_kernel(EvalT* dist,
                                  const DataT* x,
                                  const DataT* y,
                                  IdxT m,
                                  IdxT n,
                                  IdxT k,
                                  raft::distance::DistanceType metric)
{
  IdxT midx = IdxT(threadIdx.x) + IdxT(blockIdx.x) * IdxT(blockDim.x);
  if (midx >= m) return;
  IdxT grid_size = IdxT(blockDim.y) * IdxT(gridDim.y);
  for (IdxT nidx = threadIdx.y + blockIdx.y * blockDim.y; nidx < n; nidx += grid_size) {
    EvalT acc = EvalT(0);
    for (IdxT i = 0; i < k; ++i) {
      IdxT xidx = i + midx * k;
      IdxT yidx = i + nidx * k;
      auto xv   = EvalT(x[xidx]);
      auto yv   = EvalT(y[yidx]);
      switch (metric) {
        case raft::distance::DistanceType::InnerProduct: {
          acc += xv * yv;
        } break;
        case raft::distance::DistanceType::L2SqrtExpanded:
        case raft::distance::DistanceType::L2SqrtUnexpanded:
        case raft::distance::DistanceType::L2Expanded:
        case raft::distance::DistanceType::L2Unexpanded: {
          auto diff = xv - yv;
          acc += diff * diff;
        } break;
        default: break;
      }
    }
    switch (metric) {
      case raft::distance::DistanceType::L2SqrtExpanded:
      case raft::distance::DistanceType::L2SqrtUnexpanded: {
        acc = raft::sqrt(acc);
      } break;
      default: break;
    }
    dist[midx * n + nidx] = acc;
  }
}

/**
 * Naive, but flexible bruteforce KNN search.
 *
 * TODO: either replace this with brute_force_knn or with distance+select_k
 *       when either distance or brute_force_knn support 8-bit int inputs.
 */
template <typename EvalT, typename DataT, typename IdxT>
void naive_knn(raft::resources const& handle,
               EvalT* dist_topk,
               IdxT* indices_topk,
               const DataT* x,
               const DataT* y,
               size_t n_inputs,
               size_t input_len,
               size_t dim,
               uint32_t k,
               raft::distance::DistanceType type)
{
  rmm::mr::device_memory_resource* mr = nullptr;
  auto pool_guard                     = raft::get_pool_memory_resource(mr, 1024 * 1024);

  auto stream = raft::resource::get_cuda_stream(handle);
  dim3 block_dim(16, 32, 1);
  // maximum reasonable grid size in `y` direction
  auto grid_y =
    static_cast<uint16_t>(std::min<size_t>(raft::ceildiv<size_t>(input_len, block_dim.y), 32768));

  // bound the memory used by this function
  size_t max_batch_size =
    std::min<size_t>(n_inputs, raft::ceildiv<size_t>(size_t(1) << size_t(27), input_len));
  rmm::device_uvector<EvalT> dist(max_batch_size * input_len, stream, mr);

  for (size_t offset = 0; offset < n_inputs; offset += max_batch_size) {
    size_t batch_size = std::min(max_batch_size, n_inputs - offset);
    dim3 grid_dim(raft::ceildiv<size_t>(batch_size, block_dim.x), grid_y, 1);

    naive_distance_kernel<EvalT, DataT, IdxT><<<grid_dim, block_dim, 0, stream>>>(
      dist.data(), x + offset * dim, y, batch_size, input_len, dim, type);

    matrix::detail::select_k<EvalT, IdxT>(handle,
                                          dist.data(),
                                          nullptr,
                                          batch_size,
                                          input_len,
                                          static_cast<int>(k),
                                          dist_topk + offset * k,
                                          indices_topk + offset * k,
                                          type != raft::distance::DistanceType::InnerProduct,
                                          mr);
  }
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

}  // namespace raft::neighbors
