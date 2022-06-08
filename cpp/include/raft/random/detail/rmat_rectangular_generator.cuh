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
DI void gen_bits(IdxT& src_bit, IdxT& dst_bit, ProbT a, ProbT ab, ProbT abc, bool clip_and_flip,
		 raft::random::PCGenerator& gen)
{
  src_bit = dst_bit = 0;
  ProbT val;
  gen.next(val);
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
  // Courtesy: clip-and-flip from the existing RMAT generator in cuGraph
  if (clip_and_flip) {
    if (src_id == dst_id) {
      if (!src_bit && dst_bit) {
	src_bit = !src_bit;
	dst_bit = !dst_bit;
      }
    }
  }
}

template <typename IdxT>
DI void store_ids(IdxT* out, IdxT* out_src, IdxT* out_dst, IdxT src_id, IdxT dst_id,
		  IdxT idx, IdxT n_edges)
{
  if (idx < n_edges) {
    if (out != nullptr) {
      // uncoalesced gmem accesses!
      out[idx * 2] = src_id;
      out[idx * 2 + 1] = dst_id;
    }
    if (out_src != nullptr) {
      out_src[idx] = src_id;
    }
    if (out_dst != nullptr) {
      out_dst[idx] = dst_id;
    }
  }
}

template <typename IdxT, typename ProbT>
__global__ void rmat_gen_kernel(
  IdxT* out, IdxT* out_src, IdxT* out_dst, const ProbT* theta, IdxT r_scale, IdxT c_scale,
  IdxT n_edges, bool clip_and_flip, IdxT max_scale, raft::random::RngState r)
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
    IdxT src_bit, dst_bit;
    gen_bits(src_bit, dst_bit, a, ab, abc, clip_and_flip, gen);
    if (i < r_scale) {
      src_id += (src_bit << (r_scale - i));
    }
    if (i < c_scale) {
      dst_id += (dst_bit << (c_scale - i));
    }
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
				 bool clip_and_flip,
				 cudaStream_t stream,
				 raft::random::RngState& r)
{
  if (n_edges <= 0) return;
  static constexpr int N_THREADS = 512;
  auto max_scale = max(r_scale, c_scale);
  size_t smem_size = sizeof(ProbT) * max_scale * 2 * 2;
  auto n_blks = raft::ceildiv<IdxT>(n_edges, N_THREADS);
  rmat_gen_kernel<<<n_blks, N_THREADS, smem_size, stream>>>(
    out, out_src, out_dst, theta, r_scale, c_scale, n_edges, clip_and_flip, max_scale, r);
  RAFT_CUDA_TRY(cudaGetLastError());
  r.advance(n_edges, max_scale);
}

template <typename IdxT, typename ProbT>
__global__ void rmat_gen_kernel(
  IdxT* out, IdxT* out_src, IdxT* out_dst, ProbT a, ProbT ab, ProbT abc, IdxT r_scale, IdxT c_scale,
  IdxT n_edges, bool clip_and_flip, IdxT max_scale, raft::random::RngState r)
{
  IdxT idx = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  IdxT src_id{0}, dst_id{0};
  raft::random::PCGenerator gen{r.seed, r.base_subsequence + idx, 0};
  for (IdxT i = 0; i < max_scale; ++i) {
    IdxT src_bit, dst_bit;
    gen_bits(src_bit, dst_bit, a, ab, abc, clip_and_flip, gen);
    if (i < r_scale) {
      src_id += (src_bit << (r_scale - i));
    }
    if (i < c_scale) {
      dst_id += (dst_bit << (c_scale - i));
    }
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
				 bool clip_and_flip,
				 cudaStream_t stream,
				 raft::random::RngState& r)
{
  if (n_edges <= 0) return;
  static constexpr int N_THREADS = 512;
  auto max_scale = max(r_scale, c_scale);
  auto n_blks = raft::ceildiv<IdxT>(n_edges, N_THREADS);
  auto ab = a + b, abc = ab + c;
  rmat_gen_kernel<<<n_blks, N_THREADS, smem_size, stream>>>(
    out, out_src, out_dst, a, ab, abc, r_scale, c_scale, n_edges, clip_and_flip, max_scale, r);
  RAFT_CUDA_TRY(cudaGetLastError());
  r.advance(n_edges, max_scale);
}

}  // end namespace detail
}  // end namespace random
}  // end namespace raft
