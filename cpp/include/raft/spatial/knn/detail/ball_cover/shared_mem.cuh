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

template <typename value_idx, typename value_t, bool Dir, typename Comp,
          int numwarpq, int numthreadq, int tpb>
__device__ int compute_dist_full_warp(
  faiss::gpu::KeyValueBlockSelect<value_t, value_idx, Dir, Comp, numwarpq,
                                  numthreadq, tpb> &heap,
  faiss::gpu::KeyValuePair<value_t, value_idx> &cur_V, int lane_id, int n_cols,
  int k, const value_t *x_ptr, const value_t *X, value_t z,
  value_t cur_candidate_dist, value_idx cur_candidate_ind) {
  int n_dists_computed = 0;
  value_t dist = std::numeric_limits<value_t>::max();
  bool compute_distance =
    (n_dists_computed < k || z <= heap.warpKTop) && cur_candidate_ind > -1;
  unsigned int mask = __ballot_sync(0xffffffff, compute_distance);

  while (__popc(mask) > 0) {
    unsigned int lowest_peer = __ffs(mask) - 1;
    bool leader = lowest_peer == lane_id;

    value_idx idx = __shfl_sync(0xffffffff, cur_candidate_ind, lowest_peer);

    value_t sq_diff = 0;
    for (int j = lane_id; j < n_cols; j += 32) {
      value_t d = X[n_cols * idx + j] - x_ptr[j];
      sq_diff += d * d;
    }

    for (int offset = 16; offset > 0; offset /= 2)
      sq_diff += __shfl_down_sync(0xffffffff, sq_diff, offset);

    sq_diff = __shfl_sync(0xffffffff, sq_diff, 0);
    dist = sq_diff;
    n_dists_computed += 1;

    mask = mask ^ (1 << lowest_peer);
    cur_V.key = cur_candidate_dist;
    cur_V.value = cur_candidate_ind;

    if (leader) heap.addThreadQ(dist, cur_V);
  }

  return n_dists_computed;
}

template <typename value_idx, typename value_t, bool Dir, typename Comp,
          int numwarpq, int numthreadq, int tpb>
__device__ int compute_dist_block(
  faiss::gpu::KeyValueBlockSelect<value_t, value_idx, Dir, Comp, numwarpq,
                                  numthreadq, tpb> &heap,
  faiss::gpu::KeyValuePair<value_t, value_idx> &cur_V, int warp_id, int lane_id,
  const int kNumWarps, int n_cols, int k, const value_t *local_x_ptr,
  const value_t *X, const value_idx *R_1nn_cols, const value_t *R_1nn_dists,
  value_idx R_start_offset, value_idx R_size) {
  int n_dists_computed = 0;
  int limit = faiss::gpu::utils::roundDown(R_size, faiss::gpu::kWarpSize);
  int i = threadIdx.x;
  for (; i < limit; i += tpb) {
    // Index and distance of current candidate's nearest landmark
    value_idx cur_candidate_ind = R_1nn_cols[R_start_offset + i];
    value_t cur_candidate_dist = R_1nn_dists[R_start_offset + i];

    // Take 2 landmarks l_1 and l_2 where l_1 is the furthest point in the heap
    // and l_2 is the current landmark R. s is the current data point and
    // t is the new candidate data point. We know that:
    // d(s, t) cannot possibly be any smaller than | d(s, l_1) - d(l_1, l_2) | * | d(l_1, l_2) - d(l_2, t) | - d(s, l_1) * d(l_2, t)

    // Therefore, if d(s, t) >= d(s, l_1) from the computation above, we know that the distance to the candidate point
    // cannot possibly be in the nearest neighbors. However, if d(s, t) < d(s, l_1) then we should compute the
    // distance because it's possible it could be smaller.
    //
    value_t z = heap.warpKTopRDist == 0.00
                  ? 0.0
                  : (abs(heap.warpKTop - heap.warpKTopRDist) *
                       abs(heap.warpKTopRDist - cur_candidate_dist) -
                     heap.warpKTop * cur_candidate_dist) /
                      heap.warpKTopRDist;
    z = isnan(z) ? 0.0 : z;
    n_dists_computed += compute_dist_full_warp<value_idx, value_t, false,
                                               faiss::gpu::Comparator<value_t>,
                                               numwarpq, numthreadq, tpb>(
      heap, cur_V, lane_id, n_cols, k, local_x_ptr, X, z, cur_candidate_dist,
      cur_candidate_ind);

    heap.checkThreadQ();
  }

  // R_size - `i` should have <=32 elements left to process.
  // e.g. all threads in last warp will need to process it.
  if (warp_id == (R_size % tpb) / kNumWarps) {
    value_idx cur_candidate_ind =
      i < R_size ? R_1nn_cols[R_start_offset + i] : -1;
    value_t cur_candidate_dist = i < R_size
                                   ? R_1nn_dists[R_start_offset + i]
                                   : std::numeric_limits<value_t>::max();

    value_t z = heap.warpKTopRDist == 0.00
                  ? 0.0
                  : (abs(heap.warpKTop - heap.warpKTopRDist) *
                       abs(heap.warpKTopRDist - cur_candidate_dist) -
                     heap.warpKTop * cur_candidate_dist) /
                      heap.warpKTopRDist;

    z = isnan(z) ? 0.0 : z;

    n_dists_computed += compute_dist_full_warp<value_idx, value_t, false,
                                               faiss::gpu::Comparator<value_t>,
                                               numwarpq, numthreadq, tpb>(
      heap, cur_V, lane_id, n_cols, k, local_x_ptr, X, z, cur_candidate_dist,
      cur_candidate_ind);
  }
  heap.checkThreadQ();

  return n_dists_computed;
}

template <typename value_idx, typename value_t, int warp_q = 32,
          int thread_q = 2, int tpb = 128, int col_q = 1024,
          typename value_int = int, typename bitset_type = uint32_t,
          typename dist_func>
__global__ void compute_final_dists_smem(
  const value_t *X, const value_int n_cols, bitset_type *bitset,
  int bitset_size, const value_t *R_knn_dists, const value_idx *R_indptr,
  const value_idx *R_1nn_inds, const value_t *R_1nn_dists, value_idx *knn_inds,
  value_t *knn_dists, int n_landmarks, int k, dist_func dfunc,
  value_int *dist_counter) {
  static constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

  int row = blockIdx.x;
  int tid = threadIdx.x;

  int warp_id = tid / 32;
  int lane_id = tid % 32;

  int n_dists_computed = 0;

  __shared__ value_t smemK[kNumWarps * warp_q];
  __shared__ faiss::gpu::KeyValuePair<value_t, value_idx>
    smemV[kNumWarps * warp_q];
  __shared__ value_t local_x_ptr[col_q];

  const value_t *x_ptr = X + (n_cols * row);
  for (int j = tid; j < n_cols; j++) local_x_ptr[j] = x_ptr[j];
  __syncthreads();

  faiss::gpu::KeyValuePair<value_t, value_idx> initV(
    faiss::gpu::Limits<value_t>::getMax(), -1);
  faiss::gpu::KeyValuePair cur_V(initV.key, initV.value);

  faiss::gpu::KeyValueBlockSelect<value_t, value_idx, false,
                                  faiss::gpu::Comparator<value_t>, warp_q,
                                  thread_q, tpb>
    heap(faiss::gpu::Limits<value_t>::getMax(), initV, smemK, smemV, k);

  int n_k = faiss::gpu::utils::roundDown(k, faiss::gpu::kWarpSize);
  int i = tid;
  for (; i < n_k; i += blockDim.x) {
    value_idx ind = knn_inds[row * k + i];
    value_t cur_candidate_dist = R_knn_dists[ind * k];

    cur_V.key = cur_candidate_dist;
    cur_V.value = ind;
    heap.add(knn_dists[row * k + i], cur_V);
  }

  if (i < k) {
    value_idx ind = knn_inds[row * k + i];
    value_t cur_candidate_dist = R_knn_dists[ind * k];

    cur_V.key = cur_candidate_dist;
    cur_V.value = ind;
    heap.addThreadQ(knn_dists[row * k + i], cur_V);
  }

  heap.checkThreadQ();

  for (int cur_R_ind = 0; cur_R_ind < n_landmarks; cur_R_ind++) {
    // if cur R overlaps cur point's closest R, it could be a
    // candidate
    if (_get_val(bitset + (row * bitset_size), cur_R_ind)) {
      value_idx R_start_offset = R_indptr[cur_R_ind];
      value_idx R_stop_offset = R_indptr[cur_R_ind + 1];
      value_idx R_size = R_stop_offset - R_start_offset;

      n_dists_computed = compute_dist_block<value_idx, value_t, false,
                                            faiss::gpu::Comparator<value_t>,
                                            warp_q, thread_q, tpb>(
        heap, cur_V, warp_id, lane_id, kNumWarps, n_cols, k, local_x_ptr, X,
        R_1nn_inds, R_1nn_dists, R_start_offset, R_size);
    }
  }

  heap.reduce();

  for (int i = tid; i < k; i += tpb) {
    knn_dists[blockIdx.x * k + i] = smemK[i];
    knn_inds[blockIdx.x * k + i] = smemV[i].value;
  }
  //
  //  if(n_dists_computed > 0)
  //    atomicAdd(dist_counter + row, n_dists_computed / 32);
}

template <typename value_idx, typename value_t, typename value_int = int,
          int tpb = 32>
__global__ void perform_post_filter(const value_t *X, value_int n_cols,
                                    const value_idx *R_knn_inds,
                                    const value_t *R_knn_dists,
                                    const value_t *R_radius,
                                    const value_t *landmarks, int n_landmarks,
                                    int bitset_size, int k, uint32_t *output,
                                    float weight = 1.0) {
  value_idx row = blockIdx.x;

  int num_warps = (tpb / 32);

  int tid = threadIdx.x;
  int lane_id = tid % 32;
  int warp_id = tid / 32;

  const value_t *x_ptr = X + (n_cols * row);

  // allocate array of size n_landmarks / 32 ints
  extern __shared__ uint32_t smem[];

  // Start with all bits on
  for (int i = tid; i < bitset_size; i += blockDim.x) smem[i] = 0xffffffff;

  __syncthreads();

  value_t closest_R_dist = R_knn_dists[row * k];

  // zero out bits for closest k landmarks
  for (int j = tid; j < k; j += blockDim.x) {
    _zero_bit(smem, (uint32_t)R_knn_inds[row * k + j]);
  }

  __syncthreads();

  // Discard any landmarks where p(q, r) > p(q, r_q) + radius(r)
  // That is, the distance between the current point and the current
  // landmark is > the distance between the current point and
  // its closest landmark + the radius of the current landmark.
  for (int l = warp_id; l < n_landmarks; l += num_warps) {
    // compute p(q, r)
    const value_t *y_ptr = landmarks + (n_cols * l);

    bool compute = true;
    if (lane_id == 0) compute = _get_val(smem, l);

    compute = __shfl_sync(0xffffffff, compute, 0);

    if (compute) {
      // Euclidean
      value_t p_q_r = 0;
      for (int i = lane_id; i < n_cols; i += 32) {
        value_t d = y_ptr[i] - x_ptr[i];
        p_q_r += d * d;
      }

      for (int offset = 16; offset > 0; offset /= 2)
        p_q_r += __shfl_down_sync(0xffffffff, p_q_r, offset);

      if (lane_id == 0 && (p_q_r > weight * (closest_R_dist + R_radius[l]) ||
                           p_q_r > 3 * closest_R_dist)) {
        _zero_bit(smem, l);
      }
    }
  }

  __syncthreads();

  /**
   * Output bitset
   */
  for (int l = tid; l < bitset_size; l += tpb) {
    output[row * bitset_size + l] = smem[l];
  }
}

/**
 * Random ball cover kernel for n_dims > 2.
 *
 * This function parallelizes distance computations within
 * warps to maximize the parallelism and uniformity of
 * memory accesses. Each query row is mapped to a threadblock
 * and the flow of logic within each block is as follows:
 * 1. for each 1..k closest landmarks:
 * 2.   - each thread computes possibility that data point exists within boundary of the
 *        closest neighbors found so far
 * 3.   - within each warp, use whole warp to compute distances for this threads that
 *        require it and each individual thread adds to their local heap, compacting
 *        if possible (block synchronously) after each warp iteration.
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
          typename value_int = int, typename distance_func>
__global__ void block_rbc_kernel_smem(
  const value_t *X,
  value_int n_cols,  // n_cols should be 2 or 3 dims
  const value_idx *R_knn_inds, const value_t *R_knn_dists, value_int m, int k,
  const value_idx *R_indptr, const value_idx *R_1nn_cols,
  const value_t *R_1nn_dists, value_idx *out_inds, value_t *out_dists,
  value_idx *sampled_inds_map, int *dist_counter, value_t *R_radius,
  distance_func dfunc, float weight = 1.0) {
  static constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

  // Each block works on a single query row
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  int n_dists_computed = 0;

  __shared__ value_t smemK[kNumWarps * warp_q];
  __shared__ faiss::gpu::KeyValuePair<value_t, value_idx>
    smemV[kNumWarps * warp_q];
  __shared__ value_t
    local_x_ptr[col_q];  // TODO: cols_q should be power of 2 up to some max val

  // TODO: Separate kernels for different widths:
  // 1. Very small (between 3 and 32) just use registers for columns of "row"
  // 2. Can fit comfortably in shared memory (32 to a few thousand?)
  // 3. Load each time individually.
  const value_t *x_ptr = X + (n_cols * row);

  // populate smem for x in parallel
  for (int i = threadIdx.x; i < n_cols; i++) local_x_ptr[i] = x_ptr[i];

  __syncthreads();

  faiss::gpu::KeyValuePair<value_t, value_idx> initV(
    faiss::gpu::Limits<value_t>::getMax(), -1);

  faiss::gpu::KeyValuePair cur_V(initV.key, initV.value);

  faiss::gpu::KeyValueBlockSelect<value_t, value_idx, false,
                                  faiss::gpu::Comparator<value_t>, warp_q,
                                  thread_q, tpb>
    heap(faiss::gpu::Limits<value_t>::getMax(), initV, smemK, smemV, k);

  value_t min_R_dist = R_knn_dists[row * k];

  /**
   * First add distances for k closest neighbors of R
   * to the heap
   */
  // Start iterating through elements of each set from closest R elements,
  // determining if the distance could even potentially be in the heap.
  for (int cur_k = 0; cur_k < k; cur_k++) {
    // index and distance to current row's closest landmark
    value_t cur_R_dist = R_knn_dists[row * k + cur_k];
    value_idx cur_R_ind = R_knn_inds[row * k + cur_k];

    // Equation (2) in Cayton's paper- prune out R's which are > 3 * p(q, r_q)
    if (cur_R_dist > weight * (min_R_dist + R_radius[cur_R_ind])) continue;
    if (cur_R_dist > 3 * min_R_dist) continue;

    // The whole warp should iterate through the elements in the current R
    value_idx R_start_offset = R_indptr[cur_R_ind];
    value_idx R_stop_offset = R_indptr[cur_R_ind + 1];

    value_idx R_size = R_stop_offset - R_start_offset;

    n_dists_computed += compute_dist_block<value_idx, value_t, false,
                                           faiss::gpu::Comparator<value_t>,
                                           warp_q, thread_q, tpb>(
      heap, cur_V, warp_id, lane_id, kNumWarps, n_cols, k, local_x_ptr, X,
      R_1nn_cols, R_1nn_dists, R_start_offset, R_size);
  }

  heap.reduce();

  for (int i = tid; i < k; i += tpb) {
    out_dists[blockIdx.x * k + i] = smemK[i];
    out_inds[blockIdx.x * k + i] = smemV[i].value;
  }
  //  if(n_dists_computed > 0)
  //    atomicAdd(dist_counter + row, n_dists_computed/32);
}

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft