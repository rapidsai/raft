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

#include "common.cuh"

#include "../block_select_faiss.cuh"
#include "../selection_faiss.cuh"

#include <limits.h>

#include <raft/cuda_utils.cuh>

#include <faiss/utils/Heap.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

static constexpr float warp_reciprocal = 1.0 / faiss::gpu::kWarpSize;

/**
 * Uses a whole warp to compute distances, maximizing uniformity,
 * parallelism, and coalesced reads from global memory. Each warp
 * thread holds the index to a potential distance and determines
 * whether that distance should be computed. The whole warp
 * collectively iterates through each distance that needs to be
 * computed, performing the actual distance computation in parallel.
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam Dir
 * @tparam Comp
 * @tparam numwarpq
 * @tparam numthreadq
 * @tparam tpb
 * @param heap
 * @param lane_id
 * @param n_cols
 * @param x_ptr
 * @param X
 * @param z
 * @param cur_candidate_dist
 * @param cur_candidate_ind
 */
template <typename value_idx, typename value_t, bool Dir, typename Comp,
          int numwarpq, int numthreadq, int tpb, typename reduce_func,
          typename accum_func>
__device__ int compute_dist_full_warp(
  faiss::gpu::KeyValueBlockSelect<value_t, int, Dir, Comp, numwarpq, numthreadq,
                                  tpb> &heap,
  const int lane_id, const int n_cols, const value_t *__restrict__ x_ptr,
  const value_t *__restrict__ X, value_t z, value_t cur_candidate_dist,
  value_idx cur_candidate_ind, reduce_func redfunc, accum_func accfunc) {
  unsigned int mask = __ballot_sync(
    0xffffffff, z <= heap.warpKTop && cur_candidate_ind > -1);

  int n_dists = 0;
  while (__popc(mask) > 0) {
    const unsigned int lowest_peer = __ffs(mask) - 1;
    const value_idx idx = __shfl_sync(0xffffffff, cur_candidate_ind, lowest_peer);

    value_t sq_diff = 0;
    for (int j = lane_id; j < n_cols; j += 32) {
      const value_t d = redfunc(X[n_cols * idx + j], x_ptr[j]);//X[n_cols * idx + j] - x_ptr[j];
      sq_diff = accfunc(sq_diff, d);
    }

    for (int offset = 16; offset > 0; offset /= 2)
      sq_diff = accfunc(sq_diff, __shfl_down_sync(0xffffffff, sq_diff, offset));

    sq_diff = __shfl_sync(0xffffffff, sq_diff, 0);

    if(lane_id == lowest_peer)
      ++n_dists;

    mask = mask ^ (1 << lowest_peer);
    if (lowest_peer == lane_id) {
      heap.addThreadQ(sq_diff, cur_candidate_dist, (int)cur_candidate_ind);
    }
  }

  return n_dists;
}

/**
 * Helper function to use the triangle inequality for the points
 * in a single row of a sparse matrix and computing distances /
 * adding to k-select heap as necessary. This function parallelizes
 * over the entire block but any distances that need to be computed
 * are parallelized over entire warps to increase uniformity and
 * coalesced reads to the global memory where possible.
 *
 * While this approach is very fast and makes good use of the hardware
 * resources, it does use an excessive amount of registers (~72 at the
 * time of writing for a moderately small size of k). Future upgrades
 * to this design should consider ways to reduce the register usage.
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam Dir
 * @tparam Comp
 * @tparam numwarpq
 * @tparam numthreadq
 * @tparam tpb
 * @param heap
 * @param warp_id
 * @param lane_id
 * @param kNumWarps
 * @param n_cols
 * @param k
 * @param local_x_ptr
 * @param X
 * @param R_1nn_cols
 * @param R_1nn_dists
 * @param R_start_offset
 * @param R_size
 */
template <typename value_idx, typename value_t, bool Dir, typename Comp,
          int numwarpq, int numthreadq, int tpb, typename reduce_func,
          typename accum_func>
__device__ int compute_dist_block(
  faiss::gpu::KeyValueBlockSelect<value_t, int, Dir, Comp, numwarpq, numthreadq,
                                  tpb> &heap,
  const int warp_id, const int lane_id, const int kNumWarps, const int n_cols,
  const int k, const value_t *__restrict__ local_x_ptr,
  const value_t *__restrict__ X, const value_idx *__restrict__ R_1nn_cols,
  const value_t *__restrict__ R_1nn_dists, value_idx R_start_offset,
  value_idx R_size, reduce_func redfunc, accum_func accfunc) {

  const value_idx limit = faiss::gpu::utils::roundDown(R_size, faiss::gpu::kWarpSize);

  int n_dists = 0;
  int i = threadIdx.x;
  for (; i < limit; i += tpb) {
    // Index and distance of current candidate's nearest landmark
    const value_idx cur_candidate_ind = R_1nn_cols[R_start_offset + i];
    const value_t cur_candidate_dist = R_1nn_dists[R_start_offset + i];

    /** Take 2 landmarks l_1 and l_2 where l_1 is the furthest point in the heap
      * and l_2 is the current landmark R. s is the current data point and
      * t is the new candidate data point. We know that:
      * d(s, t) cannot possibly be any smaller than | d(s, l_1) - d(l_1, l_2) | * | d(l_1, l_2) - d(l_2, t) | - d(s, l_1) * d(l_2, t)
      * Therefore, if d(s, t) >= d(s, l_1) from the computation above, we know that the distance to the candidate point
      * cannot possibly be in the nearest neighbors. However, if d(s, t) < d(s, l_1) then we should compute the
      * distance because it's possible it could be smaller.
      **/
    value_t z = heap.warpKTopRDist == 0.00
                ? 0.0
                : (abs(heap.warpKTop - heap.warpKTopRDist) *
                   abs(heap.warpKTopRDist - cur_candidate_dist) -
                   heap.warpKTop * cur_candidate_dist) /
                  heap.warpKTopRDist;
    z = isnan(z) ? 0.0 : z;

    n_dists += compute_dist_full_warp<value_idx, value_t, false,
      faiss::gpu::Comparator<value_t>,
      numwarpq, numthreadq, tpb> (
      heap, lane_id, n_cols, local_x_ptr, X, z,
      cur_candidate_dist, cur_candidate_ind, redfunc, accfunc);

    heap.checkThreadQ();
  }

  // R_size - `i` should have <=32 elements left to process.
  // e.g. all threads in last warp will need to process it.
  if (warp_id == (R_size / faiss::gpu::kWarpSize) % kNumWarps) {
    const value_idx cur_candidate_ind =
      i < R_size ? R_1nn_cols[R_start_offset + i] : -1;
    const value_t cur_candidate_dist = i < R_size
                                       ? R_1nn_dists[R_start_offset + i]
                                       : std::numeric_limits<value_t>::max();

    value_t z = heap.warpKTopRDist == 0.00
                ? 0.0
                : (abs(heap.warpKTop - heap.warpKTopRDist) *
                   abs(heap.warpKTopRDist - cur_candidate_dist) -
                   heap.warpKTop * cur_candidate_dist) /
                  heap.warpKTopRDist;


    z = isnan(z) ? 0.0 : z;
    n_dists += compute_dist_full_warp<value_idx, value_t, false,
      faiss::gpu::Comparator<value_t>,
      numwarpq, numthreadq, tpb>(
      heap, lane_id, n_cols, local_x_ptr, X, z, cur_candidate_dist,
      cur_candidate_ind, redfunc, accfunc);
  }
  heap.checkThreadQ();

  return n_dists;
}

template <typename value_idx, typename value_t, int warp_q = 32,
          int thread_q = 2, int tpb = 128, int col_q = 1024,
          typename value_int = int, typename bitset_type = uint32_t,
          typename dist_func, typename reduce_func, typename accum_func>
__global__ void compute_final_dists_smem(
  const value_t *__restrict__ X, const value_int n_cols,
  bitset_type *__restrict__ bitset, int bitset_size,
  const value_t *__restrict__ R_knn_dists,
  const value_idx *__restrict__ R_indptr,
  const value_idx *__restrict__ R_1nn_inds,
  const value_t *__restrict__ R_1nn_dists,
  value_idx *knn_inds, value_t *knn_dists,
  int n_landmarks, const int k, dist_func dfunc,
  reduce_func redfunc, accum_func accfunc, value_int *dist_counter) {
  static constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

  __shared__ value_t smemK[kNumWarps * warp_q];
  __shared__ faiss::gpu::KeyValuePair<value_t, int>
    smemV[kNumWarps * warp_q];
  __shared__ value_t shared_x_ptr[col_q];

  const int warp_id = threadIdx.x * warp_reciprocal;
  const int lane_id = threadIdx.x % 32;

  for (int j = threadIdx.x; j < n_cols; j+=tpb) {
    shared_x_ptr[j] = X[n_cols * blockIdx.x + j];
  }
  __syncthreads();

  faiss::gpu::KeyValueBlockSelect<value_t, int, false,
    faiss::gpu::Comparator<value_t>,
    warp_q, thread_q, tpb>
    heap(faiss::gpu::Limits<value_t>::getMax(),
         faiss::gpu::Limits<value_t>::getMax(),
         -1, smemK, smemV, k);

  int n_dists = 0;

  int i = threadIdx.x;
  const value_idx limit = faiss::gpu::utils::roundDown(k, faiss::gpu::kWarpSize);
  for (; i < limit; i += blockDim.x) {
    const value_idx ind = knn_inds[blockIdx.x * k + i];
    heap.add(knn_dists[blockIdx.x * k + i], R_knn_dists[ind * k], ind);
  }

  if (i < k) {
    const value_idx ind = knn_inds[blockIdx.x * k + i];
    heap.addThreadQ(knn_dists[blockIdx.x * k + i], R_knn_dists[ind * k], ind);
  }

  heap.checkThreadQ();

  for (int cur_R_ind = 0; cur_R_ind < n_landmarks; cur_R_ind++) {
    // if cur R overlaps cur point's closest R, it could be a
    // candidate
    if (_get_val(bitset + (blockIdx.x * bitset_size), cur_R_ind)) {

      // TODO: see if putting these values in smem can reduce register count
      const value_idx R_start_offset = R_indptr[cur_R_ind];
      const value_idx R_size = R_indptr[cur_R_ind + 1] - R_start_offset;
      n_dists += compute_dist_block<value_idx, value_t, false,
        faiss::gpu::Comparator<value_t>,
        warp_q, thread_q, tpb>(
        heap, warp_id, lane_id, kNumWarps, n_cols, k, shared_x_ptr, X,
        R_1nn_inds, R_1nn_dists, R_start_offset, R_size, redfunc, accfunc);
    }
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    knn_dists[blockIdx.x * k + i] = smemK[i];
    knn_inds[blockIdx.x * k + i] = smemV[i].value;
  }

  atomicAdd(dist_counter + blockIdx.x, n_dists);
}

/**
 * To find exact neighbors, we perform a post-processing stage
 * that filters out those points which might have neighbors outside
 * of their k closest landmarks. This is usually a very small portion
 * of the total points.
 * @tparam value_idx
 * @tparam value_t
 * @tparam value_int
 * @tparam tpb
 * @param X
 * @param n_cols
 * @param R_knn_inds
 * @param R_knn_dists
 * @param R_radius
 * @param landmarks
 * @param n_landmarks
 * @param bitset_size
 * @param k
 * @param output
 * @param weight
 */
template <typename value_idx, typename value_t, typename value_int = int,
          int tpb = 32, typename reduce_func, typename accum_func>
__global__ void perform_post_filter(
  const value_t *X, value_int n_cols,
  const value_idx *R_knn_inds,
  const value_t *R_knn_dists,
  const value_t *R_radius,
  const value_t *landmarks,
  const value_t *knn_dists,
  int n_landmarks, int bitset_size, int k, uint32_t *output,
  reduce_func redfunc, accum_func accfunc, float weight = 1.0) {
  static constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

  const int warp_id = threadIdx.x * warp_reciprocal;
  const int lane_id = threadIdx.x % 32;

  // allocate array of size n_landmarks / 32 ints
  extern __shared__ uint32_t smem[];

  // Start with all bits on
  for (int i = threadIdx.x; i < bitset_size; i += tpb) smem[i] = 0xffffffff;

  __syncthreads();

  value_t closest_R_dist = R_knn_dists[blockIdx.x * k + (k-1)];

  // zero out bits for closest k landmarks
  for (int j = threadIdx.x; j < k; j += tpb) {
    _zero_bit(smem, (uint32_t)R_knn_inds[blockIdx.x * k + j]);
  }

  __syncthreads();

  // Discard any landmarks where p(q, r) > p(q, r_q) + radius(r)
  // That is, the distance between the current point and the current
  // landmark is > the distance between the current point and
  // its closest landmark + the radius of the current landmark.
  for (int l = warp_id; l < n_landmarks; l += kNumWarps) {
    // compute p(q, r)
    const value_t *y_ptr = landmarks + (n_cols * l);

    bool compute = lane_id == 0 ? _get_val(smem, l) : true;
    compute = __shfl_sync(0xffffffff, compute, 0);

    if (compute) {
      // Euclidean
      value_t p_q_r = 0;
      for (int i = lane_id; i < n_cols; i += 32) {
        value_t d = redfunc(y_ptr[i], X[blockIdx.x * n_cols + i]);
        p_q_r = accfunc(p_q_r, d);
      }

      for (int offset = 16; offset > 0; offset /= 2)
        p_q_r = accfunc(p_q_r, __shfl_down_sync(0xffffffff, p_q_r, offset));

      const bool in_bound = true;//(p_q_r > weight * (closest_R_dist + R_radius[l]) ||
                             //p_q_r > 3 * closest_R_dist ||
                            // knn_dists[blockIdx.x * k + (k-1)] < p_q_r - R_radius[l]);
      if (lane_id == 0 && in_bound) {
        _zero_bit(smem, l);
      }
    }
  }

  __syncthreads();

  /**
   * Output bitset
   */
  for (int l = threadIdx.x; l < bitset_size; l += tpb) {
    output[blockIdx.x * bitset_size + l] = smem[l];
  }
}

/**
 * Random ball cover kernel for high dimensionality (greater than 2 or 3).
 * This kernel loops through each of the k-closest landmarks and tries to
 * maximize uniformity of distance computations by having each warp compute
 * the distances. This also increases coalesced reads and writes and enables
 * random access to each individual datapoint on the right-hand side while
 * still allowing warp threads to load the distances for subsequent columns
 * in parallel.
 *
 * One downside to this technique is that the more complex usage of the
 * warp-level primitives increases register pressure to the point where
 * it's reducing the occupancy. A moderate amount of shared memory is
 * used to store the columns from the left-hand side, since it will
 * need to be loaded each time a distance is computed.
 *
 * This function parallelizes distance computations within
 * warps to maximize the parallelism and uniformity of
 * memory accesses. Each query blockIdx.x is mapped to a threadblock
 * and the flow of logic within each block is as follows:
 * 1. for each 1..k closest landmarks:
 * 2.   - each thread computes possibility that data point exists within boundary of the
 *        closest neighbors found so far
 * 3.   - within each warp, use whole warp to compute distances for this threads that
 *        require it and each individual thread adds to their local heap, compacting
 *        if possible (block synchronously if necessary) after each warp iteration.
 * @tparam value_idx
 * @tparam value_t
 * @tparam warp_q
 * @tparam thread_q
 * @tparam tpb
 * @tparam value_idx
 * @tparam value_t
 * @param R_knn_inds
 * @param R_knn_dists
 * @param m
 * @param k
 * @param R_indptr
 * @param R_1nn_cols
 * @param R_1nn_dists
 */
template <typename value_idx = int64_t, typename value_t, int warp_q = 32,
          int thread_q = 2, int tpb = 128, int col_q = 1024,
          typename value_int = int, typename distance_func,
          typename reduce_func, typename accum_func>
__global__ void block_rbc_kernel_smem(
  const value_t *__restrict__ X,
  const value_int n_cols,  // n_cols should be 2 or 3 dims
  const value_idx *__restrict__ R_knn_inds,
  const value_t *__restrict__ R_knn_dists,
  const value_int m, const int k,
  const value_idx *__restrict__ R_indptr,
  const value_idx *__restrict__ R_1nn_cols,
  const value_t *__restrict__ R_1nn_dists,
  value_idx *out_inds,
  value_t *out_dists,
  int *dist_counter,
  value_t *R_radius, distance_func dfunc,
  reduce_func redfunc,
  accum_func accfunc, float weight = 1.0) {
  static constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

  // Each block works on a single query blockIdx.x
  __shared__ value_t smemK[kNumWarps * warp_q];
  __shared__ faiss::gpu::KeyValuePair<value_t, int>
    smemV[kNumWarps * warp_q];
  __shared__ value_t
    shared_x_ptr[col_q];  // TODO: cols_q should be power of 2 up to some max val

  const int warp_id = threadIdx.x * warp_reciprocal;
  const int lane_id = threadIdx.x % faiss::gpu::kWarpSize;

  for (int i = threadIdx.x; i < n_cols; i+=tpb )
    shared_x_ptr[i] = X[blockIdx.x * n_cols + i];

  __syncthreads();

  faiss::gpu::KeyValueBlockSelect<value_t, int, false,
    faiss::gpu::Comparator<value_t>, warp_q,
    thread_q, tpb>
    heap(faiss::gpu::Limits<value_t>::getMax(),
         faiss::gpu::Limits<value_t>::getMax(), -1, smemK, smemV, k);

  const value_t min_R_dist = R_knn_dists[blockIdx.x * k + (k-1)];

  int n_dists = 0;

  /**
   * First add distances for k closest neighbors of R
   * to the heap
   */
  // Start iterating through elements of each set from closest R elements,
  // determining if the distance could even potentially be in the heap.
  for (int cur_k = 0; cur_k < k; cur_k++) {
    // index and distance to current row's closest landmark
    const value_t cur_R_dist = R_knn_dists[blockIdx.x * k + cur_k];
    const value_idx cur_R_ind = R_knn_inds[blockIdx.x * k + cur_k];

    // Equation (2) in Cayton's paper- prune out R's which are > 3 * p(q, r_q)
    if (cur_R_dist > weight * (min_R_dist + R_radius[cur_R_ind])) continue;
    if (cur_R_dist > 3 * min_R_dist) continue;

    // The whole warp should iterate through the elements in the current R
    const value_idx R_start_offset = R_indptr[cur_R_ind];
    const value_idx R_size = R_indptr[cur_R_ind + 1] - R_start_offset;

    n_dists += compute_dist_block<value_idx, value_t, false,
      faiss::gpu::Comparator<value_t>,
      warp_q, thread_q, tpb>(
      heap, warp_id, lane_id, kNumWarps, n_cols, k, shared_x_ptr, X,
      R_1nn_cols, R_1nn_dists, R_start_offset, R_size, redfunc, accfunc);
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    out_dists[blockIdx.x * k + i] = smemK[i];
    out_inds[blockIdx.x * k + i] = smemV[i].value;
  }

  atomicAdd(dist_counter + blockIdx.x, n_dists);
}

// TODO: Need to create wrapper functions to invoke block_rbc, post_filter_bitset, compute_final_dists

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft