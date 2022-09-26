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

#include <raft/random/rng_device.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace random {
namespace detail {

template <typename IdxT, typename ProbT>
DI void gen_and_update_bits(IdxT& src_id,
                            IdxT& dst_id,
                            ProbT a,
                            ProbT ab,
                            ProbT abc,
                            IdxT r_scale,
                            IdxT c_scale,
                            IdxT curr_depth,
                            raft::random::PCGenerator& gen)
{
  bool src_bit, dst_bit;
  ProbT val;
  gen.next(val);
  if (val <= a) {
    src_bit = dst_bit = false;
  } else if (val <= ab) {
    src_bit = false;
    dst_bit = true;
  } else if (val <= abc) {
    src_bit = true;
    dst_bit = false;
  } else {
    src_bit = dst_bit = true;
  }
  if (curr_depth < r_scale) { src_id |= (IdxT(src_bit) << (r_scale - curr_depth - 1)); }
  if (curr_depth < c_scale) { dst_id |= (IdxT(dst_bit) << (c_scale - curr_depth - 1)); }
}

template <typename IdxT>
DI void store_ids(
  IdxT* out, IdxT* out_src, IdxT* out_dst, IdxT src_id, IdxT dst_id, IdxT idx, IdxT n_edges)
{
  if (idx < n_edges) {
    if (out != nullptr) {
      // uncoalesced gmem accesses!
      out[idx * 2]     = src_id;
      out[idx * 2 + 1] = dst_id;
    }
    if (out_src != nullptr) { out_src[idx] = src_id; }
    if (out_dst != nullptr) { out_dst[idx] = dst_id; }
  }
}

template <typename IdxT, typename ProbT>
__global__ void rmat_gen_kernel(IdxT* out,
                                IdxT* out_src,
                                IdxT* out_dst,
                                const ProbT* theta,
                                IdxT r_scale,
                                IdxT c_scale,
                                IdxT n_edges,
                                IdxT max_scale,
                                raft::random::RngState r)
{
  IdxT idx = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  extern __shared__ ProbT s_theta[];
  auto theta_len = max_scale * 2 * 2;
  // load the probabilities into shared memory and then convert them into cdf's
  // currently there are smem bank conflicts due to the way these are accessed
  for (int i = threadIdx.x; i < theta_len; i += blockDim.x) {
    s_theta[i] = theta[i];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < max_scale; i += blockDim.x) {
    auto a             = s_theta[4 * i];
    auto b             = s_theta[4 * i + 1];
    auto c             = s_theta[4 * i + 2];
    s_theta[4 * i + 1] = a + b;
    s_theta[4 * i + 2] = a + b + c;
    s_theta[4 * i + 3] += a + b + c;
  }
  __syncthreads();
  IdxT src_id{0}, dst_id{0};
  raft::random::PCGenerator gen{r.seed, r.base_subsequence + idx, 0};
  for (IdxT i = 0; i < max_scale; ++i) {
    auto a = s_theta[i * 4], ab = s_theta[i * 4 + 1], abc = s_theta[i * 4 + 2];
    gen_and_update_bits(src_id, dst_id, a, ab, abc, r_scale, c_scale, i, gen);
  }
  store_ids(out, out_src, out_dst, src_id, dst_id, idx, n_edges);
}

template <typename IdxT, typename ProbT>
void rmat_rectangular_gen_caller(IdxT* out,
                                 IdxT* out_src,
                                 IdxT* out_dst,
                                 const ProbT* theta,
                                 IdxT r_scale,
                                 IdxT c_scale,
                                 IdxT n_edges,
                                 cudaStream_t stream,
                                 raft::random::RngState& r)
{
  if (n_edges <= 0) return;
  static constexpr int N_THREADS = 512;
  auto max_scale                 = max(r_scale, c_scale);
  size_t smem_size               = sizeof(ProbT) * max_scale * 2 * 2;
  auto n_blks                    = raft::ceildiv<IdxT>(n_edges, N_THREADS);
  rmat_gen_kernel<<<n_blks, N_THREADS, smem_size, stream>>>(
    out, out_src, out_dst, theta, r_scale, c_scale, n_edges, max_scale, r);
  RAFT_CUDA_TRY(cudaGetLastError());
  r.advance(n_edges, max_scale);
}

template <typename IdxT, typename ProbT>
__global__ void rmat_gen_kernel(IdxT* out,
                                IdxT* out_src,
                                IdxT* out_dst,
                                ProbT a,
                                ProbT b,
                                ProbT c,
                                IdxT r_scale,
                                IdxT c_scale,
                                IdxT n_edges,
                                IdxT max_scale,
                                raft::random::RngState r)
{
  IdxT idx = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  IdxT src_id{0}, dst_id{0};
  raft::random::PCGenerator gen{r.seed, r.base_subsequence + idx, 0};
  auto min_scale = min(r_scale, c_scale);
  IdxT i         = 0;
  for (; i < min_scale; ++i) {
    gen_and_update_bits(src_id, dst_id, a, a + b, a + b + c, r_scale, c_scale, i, gen);
  }
  for (; i < r_scale; ++i) {
    gen_and_update_bits(src_id, dst_id, a + b, a + b, ProbT(1), r_scale, c_scale, i, gen);
  }
  for (; i < c_scale; ++i) {
    gen_and_update_bits(src_id, dst_id, a + c, ProbT(1), ProbT(1), r_scale, c_scale, i, gen);
  }
  store_ids(out, out_src, out_dst, src_id, dst_id, idx, n_edges);
}

template <typename IdxT, typename ProbT>
void rmat_rectangular_gen_caller(IdxT* out,
                                 IdxT* out_src,
                                 IdxT* out_dst,
                                 ProbT a,
                                 ProbT b,
                                 ProbT c,
                                 IdxT r_scale,
                                 IdxT c_scale,
                                 IdxT n_edges,
                                 cudaStream_t stream,
                                 raft::random::RngState& r)
{
  if (n_edges <= 0) return;
  static constexpr int N_THREADS = 512;
  auto max_scale                 = max(r_scale, c_scale);
  auto n_blks                    = raft::ceildiv<IdxT>(n_edges, N_THREADS);
  rmat_gen_kernel<<<n_blks, N_THREADS, 0, stream>>>(
    out, out_src, out_dst, a, b, c, r_scale, c_scale, n_edges, max_scale, r);
  RAFT_CUDA_TRY(cudaGetLastError());
  r.advance(n_edges, max_scale);
}

}  // end namespace detail
}  // end namespace random
}  // end namespace raft
