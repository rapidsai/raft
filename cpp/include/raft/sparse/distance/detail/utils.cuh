/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <raft/core/math.hpp>
#include <raft/distance/distance_types.hpp>

#include <cub/cub.cuh>

namespace raft {
namespace sparse {
namespace distance {
namespace detail {

/**
 * Computes the maximum number of columns that can be stored
 * in shared memory in dense form with the given block size
 * and precision.
 * @return the maximum number of columns that can be stored in smem
 */
template <typename value_idx, typename value_t, int tpb = 1024>
inline int max_cols_per_block()
{
  // max cols = (total smem available - cub reduction smem)
  return (raft::getSharedMemPerBlock() - ((tpb / raft::warp_size()) * sizeof(value_t))) /
         sizeof(value_t);
}

template <typename value_idx, typename value_t, typename expansion_f>
RAFT_KERNEL epilogue_on_csr_kernel(value_t* __restrict__ compressed_C,
                                   const value_idx* __restrict__ rows,
                                   const value_idx* __restrict__ cols,
                                   const value_t* __restrict__ Q_sq_norms,
                                   const value_t* __restrict__ R_sq_norms,
                                   value_idx nnz,
                                   expansion_f expansion_func)
{
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= nnz) return;
  const value_idx i = rows[tid];
  const value_idx j = cols[tid];

  compressed_C[tid] = expansion_func(compressed_C[tid], Q_sq_norms[i], R_sq_norms[j]);
}

template <typename value_idx, typename value_t, int tpb = 256>
void epilogue_on_csr(raft::resources const& handle,
                     value_t* compressed_C,
                     const value_idx* indptr,
                     const value_idx nnz,
                     const value_idx n_rows,
                     const value_idx* cols,
                     const value_t* Q_sq_norms,
                     const value_t* R_sq_norms,
                     raft::distance::DistanceType metric)
{
  auto stream = resource::get_cuda_stream(handle);

  rmm::device_uvector<value_idx> rows(nnz, stream);
  raft::sparse::convert::csr_to_coo(indptr, n_rows, rows.data(), nnz, stream);

  int blocks = raft::ceildiv<size_t>((size_t)nnz, tpb);
  if (metric == raft::distance::DistanceType::L2Expanded) {
    epilogue_on_csr_kernel<<<blocks, tpb, 0, stream>>>(
      compressed_C,
      rows.data(),
      cols,
      Q_sq_norms,
      R_sq_norms,
      nnz,
      [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) -> value_t {
        return value_t(-2.0) * dot + q_norm + r_norm;
      });
  } else if (metric == raft::distance::DistanceType::L2SqrtExpanded) {
    epilogue_on_csr_kernel<<<blocks, tpb, 0, stream>>>(
      compressed_C,
      rows.data(),
      cols,
      Q_sq_norms,
      R_sq_norms,
      nnz,
      [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) -> value_t {
        return raft::sqrt(value_t(-2.0) * dot + q_norm + r_norm);
      });
  } else if (metric == raft::distance::DistanceType::CosineExpanded) {
    epilogue_on_csr_kernel<<<blocks, tpb, 0, stream>>>(
      compressed_C,
      rows.data(),
      cols,
      Q_sq_norms,
      R_sq_norms,
      nnz,
      [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) -> value_t {
        return value_t(1.0) - dot / (q_norm * r_norm);
      });
  }
}
}  // namespace detail
}  // namespace distance
}  // namespace sparse
}  // namespace raft
