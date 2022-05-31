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

#pragma once

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rng_device.cuh>
#include <raft/random/rng_state.hpp>

namespace raft {
namespace random {
namespace detail {

template <typename IdxT, typename ProbT>
__global__ void rmat_gen_kernel(IdxT* out, const ProbT* theta, IdxT r_scale, IdxT c_scale,
                                IdxT n_edges, IdxT max_scale, raft::random::RngState r)
{
  IdxT idx = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  extern __shared__ ProbT s_theta[];
  auto lid = raft::laneId();
  unsigned mask = 0xfu << (lid / 4);
  // NOTE: assumes that blockDim.x is a multiple of 4!
  for (int i = threadIdx.x; i < max_scale * 2 * 2; i += blockDim.x) {
    // for each consecutive 4 lanes compute the cdf of a, b, c, d (RMAT numbers)
    // this will be used to determine which quadrant to be selected at each level
    auto r_theta = theta[i];
    auto other = raft::shfl_up(r_theta, 0x1);
    if (lid % 4 >= 1) {
      r_theta += other;
    }
    other = raft::shfl_up(r_theta, 0x2);
    if (lid % 4 >= 2) {
      r_theta += other;
    }
    s_theta[i] = r_theta;
  }
  __syncthreads();
  IdxT src_id{0}, dst_id{0};
  raft::random::PCGenerator gen{r.seed, r.base_subsequence + idx, 0};
  for (IdxT i = 0; i < max_scale; ++i) {
    auto a = s_theta[i * 4], ab = s_theta[i * 4 + 1], abc = s_theta[i * 4 + 2];
    ProbT val;
    gen.next(val);
    IdxT src_bit{0}, dst_bit{0};
    if (val <= a) {
      src_bit = dst_bit = 0;
    } else if (val <= ab) {
      src_bit = 0;
      dst_bit = 1;
    } else if (val <= abc) {
      src_bit = 1;
      dst_bit = 0;
    } else {
      src_bit = dst_bit = 1;
    }
    if (i < r_scale) {
      src_id += (src_bit << (r_scale - i));
    }
    if (i < c_scale) {
      dst_id += (dst_bit << (c_scale - i));
    }
  }
  if (idx < n_edges) {
    // uncoalesced gmem accesses!
    out[idx * 2] = src_id;
    out[idx * 2 + 1] = dst_id;
  }
}

/**
 * @brief Generate RMAT for a rectangular shaped adjacency matrices (useful when
 *        graphs to be generated are bipartite)
 *
 * @tparam IdxT  node indices type
 * @tparam ProbT data type used for probability distributions (either fp32 or fp64)
 *
 * @param[out] out     generated edgelist [on device] [dim = n_edges x 2]. On each row
 *                     the first element corresponds to the source node id while the
 *                     second, the destination node id.
 * @param[in]  theta   distribution of each quadrant at each level of resolution.
 *                     Since these are probabilities, each of the 2x2 matrix for
 *                     each level of the RMAT must sum to one. [on device]
 *                     [dim = 1+log2(max(n_rows, n_cols)) x 2 x 2]
 * @param[in]  n_rows  number of source nodes
 * @param[in]  n_cols  number of destination nodes
 * @param[in]  n_edges number of edges to generate
 * @param[in]  stream  cuda stream to schedule the work on
 * @param[in]  r       underlying state of the random generator. Especially useful when
 *                     one wants to call this API for multiple times in order to generate
 *                     a larger graph. For that case, just create this object with the
 *                     initial seed once and after every call continue to pass the same
 *                     object for the successive calls.
 *
 * @note This can generate duplicate edges and self-loops. It is the responsibility of the
 *       caller to clean them up accordingly.

 * @note This also only generates directed graphs. If undirected graphs are needed, then a
 *       separate post-processing step is expected to be done by the caller.
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen_caller(IdxT* out,
				 const ProbT* theta,
				 IdxT n_rows,
				 IdxT n_cols,
				 IdxT n_edges,
				 cudaStream_t stream,
				 raft::random::RngState& r)
{
  static constexpr int N_THREADS = 512;
  auto r_scale = raft::log2(n_rows) + 1;
  auto c_scale = raft::log2(n_cols) + 1;
  auto max_scale = max(r_scale, c_scale);
  size_t smem_size = sizeof(ProbT) * max_scale * 2 * 2;
  auto n_blks = raft::ceildiv<IdxT>(n_edges, N_THREADS);
  rmat_gen_kernel<<<n_blks, N_THREADS, smem_size, stream>>>(
    out, theta, r_scale, c_scale, n_edges, max_scale, r);
  RAFT_CUDA_TRY(cudaGetLastError());
  r.advance(n_edges, max_scale);
}

}  // end namespace detail
}  // end namespace random
}  // end namespace raft
