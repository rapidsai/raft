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

#include "../../ball_cover_types.hpp"
#include "../haversine_distance.cuh"
#include "common.cuh"
#include "registers_types.cuh"  // DistFunc

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/neighbors/detail/faiss_select/key_value_block_select.cuh>
#include <raft/util/cuda_utils.cuh>

#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/scan.h>

#include <limits.h>

#include <cstdint>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

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
template <typename value_idx,
          typename value_t,
          typename value_int = std::uint32_t,
          int col_q          = 2,
          int tpb            = 32,
          typename distance_func>
RAFT_KERNEL perform_post_filter_registers(const value_t* X,
                                          value_int n_cols,
                                          const value_idx* R_knn_inds,
                                          const value_t* R_knn_dists,
                                          const value_t* R_radius,
                                          const value_t* landmarks,
                                          int n_landmarks,
                                          value_int bitset_size,
                                          value_int k,
                                          distance_func dfunc,
                                          std::uint32_t* output,
                                          float weight = 1.0)
{
  // allocate array of size n_landmarks / 32 ints
  extern __shared__ std::uint32_t shared_mem[];

  // Start with all bits on
  for (value_int i = threadIdx.x; i < bitset_size; i += tpb) {
    shared_mem[i] = 0xffffffff;
  }

  __syncthreads();

  // TODO: Would it be faster to use L1 for this?
  value_t local_x_ptr[col_q];
  for (value_int j = 0; j < n_cols; ++j) {
    local_x_ptr[j] = X[n_cols * blockIdx.x + j];
  }

  value_t closest_R_dist = R_knn_dists[blockIdx.x * k + (k - 1)];

  // zero out bits for closest k landmarks
  for (value_int j = threadIdx.x; j < k; j += tpb) {
    _zero_bit(shared_mem, (std::uint32_t)R_knn_inds[blockIdx.x * k + j]);
  }

  __syncthreads();

  // Discard any landmarks where p(q, r) > p(q, r_q) + radius(r)
  // That is, the distance between the current point and the current
  // landmark is > the distance between the current point and
  // its closest landmark + the radius of the current landmark.
  for (value_int l = threadIdx.x; l < n_landmarks; l += tpb) {
    // compute p(q, r)
    value_t dist = dfunc(local_x_ptr, landmarks + (n_cols * l), n_cols);
    if (dist > weight * (closest_R_dist + R_radius[l]) || dist > 3 * closest_R_dist) {
      _zero_bit(shared_mem, l);
    }
  }

  __syncthreads();

  /**
   * Output bitset
   */
  for (value_int l = threadIdx.x; l < bitset_size; l += tpb) {
    output[blockIdx.x * bitset_size + l] = shared_mem[l];
  }
}

/**
 * @tparam value_idx
 * @tparam value_t
 * @tparam value_int
 * @tparam bitset_type
 * @tparam warp_q number of registers to use per warp
 * @tparam thread_q number of registers to use within each thread
 * @tparam tpb number of threads per block
 * @param X
 * @param n_cols
 * @param bitset
 * @param bitset_size
 * @param R_knn_dists
 * @param R_indptr
 * @param R_1nn_inds
 * @param R_1nn_dists
 * @param knn_inds
 * @param knn_dists
 * @param n_landmarks
 * @param k
 * @param dist_counter
 */
template <typename value_idx,
          typename value_t,
          typename value_int   = std::uint32_t,
          typename bitset_type = std::uint32_t,
          typename dist_func,
          int warp_q   = 32,
          int thread_q = 2,
          int tpb      = 128,
          int col_q    = 2>
RAFT_KERNEL compute_final_dists_registers(const value_t* X_reordered,
                                          const value_t* X,
                                          const value_int n_cols,
                                          bitset_type* bitset,
                                          value_int bitset_size,
                                          const value_t* R_closest_landmark_dists,
                                          const value_idx* R_indptr,
                                          const value_idx* R_1nn_inds,
                                          const value_t* R_1nn_dists,
                                          value_idx* knn_inds,
                                          value_t* knn_dists,
                                          value_int n_landmarks,
                                          value_int k,
                                          dist_func dfunc,
                                          value_int* dist_counter)
{
  static constexpr int kNumWarps = tpb / WarpSize;

  __shared__ value_t shared_memK[kNumWarps * warp_q];
  __shared__ KeyValuePair<value_t, value_idx> shared_memV[kNumWarps * warp_q];

  const value_t* x_ptr = X + (n_cols * blockIdx.x);
  value_t local_x_ptr[col_q];
  for (value_int j = 0; j < n_cols; ++j) {
    local_x_ptr[j] = x_ptr[j];
  }

  using namespace raft::neighbors::detail::faiss_select;
  KeyValueBlockSelect<value_t, value_idx, false, Comparator<value_t>, warp_q, thread_q, tpb> heap(
    std::numeric_limits<value_t>::max(),
    std::numeric_limits<value_t>::max(),
    -1,
    shared_memK,
    shared_memV,
    k);

  const value_int n_k = Pow2<WarpSize>::roundDown(k);
  value_int i         = threadIdx.x;
  for (; i < n_k; i += tpb) {
    value_idx ind = knn_inds[blockIdx.x * k + i];
    heap.add(knn_dists[blockIdx.x * k + i], R_closest_landmark_dists[ind], ind);
  }

  if (i < k) {
    value_idx ind = knn_inds[blockIdx.x * k + i];
    heap.addThreadQ(knn_dists[blockIdx.x * k + i], R_closest_landmark_dists[ind], ind);
  }

  heap.checkThreadQ();

  for (value_int cur_R_ind = 0; cur_R_ind < n_landmarks; ++cur_R_ind) {
    // if cur R overlaps cur point's closest R, it could be a
    // candidate
    if (_get_val(bitset + (blockIdx.x * bitset_size), cur_R_ind)) {
      value_idx R_start_offset = R_indptr[cur_R_ind];
      value_idx R_stop_offset  = R_indptr[cur_R_ind + 1];
      value_idx R_size         = R_stop_offset - R_start_offset;

      // Loop through R's neighborhood in parallel

      // Round R_size to the nearest warp threads so they can
      // all be computing in parallel.

      const value_int limit = Pow2<WarpSize>::roundDown(R_size);

      i = threadIdx.x;
      for (; i < limit; i += tpb) {
        value_idx cur_candidate_ind = R_1nn_inds[R_start_offset + i];
        value_t cur_candidate_dist  = R_1nn_dists[R_start_offset + i];

        value_t z = heap.warpKTopRDist == 0.00 ? 0.0
                                               : (abs(heap.warpKTop - heap.warpKTopRDist) *
                                                    abs(heap.warpKTopRDist - cur_candidate_dist) -
                                                  heap.warpKTop * cur_candidate_dist) /
                                                   heap.warpKTopRDist;
        z         = isnan(z) || isinf(z) ? 0.0 : z;

        // If lower bound on distance could possibly be in
        // the closest k neighbors, compute it and add to k-select
        value_t dist = std::numeric_limits<value_t>::max();
        if (z <= heap.warpKTop) {
          const value_t* y_ptr = X_reordered + (n_cols * (R_start_offset + i));
          value_t local_y_ptr[col_q];
          for (value_int j = 0; j < n_cols; ++j) {
            local_y_ptr[j] = y_ptr[j];
          }

          dist = dfunc(local_x_ptr, local_y_ptr, n_cols);
        }

        heap.add(dist, cur_candidate_dist, cur_candidate_ind);
      }

      // second round guarantees to be only a single warp.
      if (i < R_size) {
        value_idx cur_candidate_ind = R_1nn_inds[R_start_offset + i];
        value_t cur_candidate_dist  = R_1nn_dists[R_start_offset + i];

        value_t z = heap.warpKTopRDist == 0.00 ? 0.0
                                               : (abs(heap.warpKTop - heap.warpKTopRDist) *
                                                    abs(heap.warpKTopRDist - cur_candidate_dist) -
                                                  heap.warpKTop * cur_candidate_dist) /
                                                   heap.warpKTopRDist;

        z = isnan(z) || isinf(z) ? 0.0 : z;

        // If lower bound on distance could possibly be in
        // the closest k neighbors, compute it and add to k-select
        value_t dist = std::numeric_limits<value_t>::max();
        if (z <= heap.warpKTop) {
          const value_t* y_ptr = X_reordered + (n_cols * (R_start_offset + i));
          value_t local_y_ptr[col_q];
          for (value_int j = 0; j < n_cols; ++j) {
            local_y_ptr[j] = y_ptr[j];
          }
          dist = dfunc(local_x_ptr, local_y_ptr, n_cols);
        }
        heap.addThreadQ(dist, cur_candidate_dist, cur_candidate_ind);
      }
      heap.checkThreadQ();
    }
  }

  heap.reduce();

  for (value_int i = threadIdx.x; i < k; i += tpb) {
    knn_dists[blockIdx.x * k + i] = shared_memK[i];
    knn_inds[blockIdx.x * k + i]  = shared_memV[i].value;
  }
}

/**
 * Random ball cover kernel for n_dims == 2
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
template <typename value_idx = std::int64_t,
          typename value_t,
          int warp_q         = 32,
          int thread_q       = 2,
          int tpb            = 128,
          int col_q          = 2,
          typename value_int = std::uint32_t,
          typename distance_func>
RAFT_KERNEL block_rbc_kernel_registers(const value_t* X_reordered,
                                       const value_t* X,
                                       value_int n_cols,  // n_cols should be 2 or 3 dims
                                       const value_idx* R_knn_inds,
                                       const value_t* R_knn_dists,
                                       value_int m,
                                       value_int k,
                                       const value_idx* R_indptr,
                                       const value_idx* R_1nn_cols,
                                       const value_t* R_1nn_dists,
                                       value_idx* out_inds,
                                       value_t* out_dists,
                                       value_int* dist_counter,
                                       const value_t* R_radius,
                                       distance_func dfunc,
                                       float weight = 1.0)
{
  static constexpr value_int kNumWarps = tpb / WarpSize;

  __shared__ value_t shared_memK[kNumWarps * warp_q];
  __shared__ KeyValuePair<value_t, value_idx> shared_memV[kNumWarps * warp_q];

  // TODO: Separate kernels for different widths:
  // 1. Very small (between 3 and 32) just use registers for columns of "blockIdx.x"
  // 2. Can fit comfortably in shared memory (32 to a few thousand?)
  // 3. Load each time individually.
  const value_t* x_ptr = X + (n_cols * blockIdx.x);

  // Use registers only for 2d or 3d
  value_t local_x_ptr[col_q];
  for (value_int i = 0; i < n_cols; ++i) {
    local_x_ptr[i] = x_ptr[i];
  }

  // Each warp works on 1 R
  using namespace raft::neighbors::detail::faiss_select;
  KeyValueBlockSelect<value_t, value_idx, false, Comparator<value_t>, warp_q, thread_q, tpb> heap(
    std::numeric_limits<value_t>::max(),
    std::numeric_limits<value_t>::max(),
    -1,
    shared_memK,
    shared_memV,
    k);

  value_t min_R_dist         = R_knn_dists[blockIdx.x * k + (k - 1)];
  value_int n_dists_computed = 0;

  /**
   * First add distances for k closest neighbors of R
   * to the heap
   */
  // Start iterating through elements of each set from closest R elements,
  // determining if the distance could even potentially be in the heap.
  for (value_int cur_k = 0; cur_k < k; ++cur_k) {
    // index and distance to current blockIdx.x's closest landmark
    value_t cur_R_dist  = R_knn_dists[blockIdx.x * k + cur_k];
    value_idx cur_R_ind = R_knn_inds[blockIdx.x * k + cur_k];

    // Equation (2) in Cayton's paper- prune out R's which are > 3 * p(q, r_q)
    if (cur_R_dist > weight * (min_R_dist + R_radius[cur_R_ind])) continue;
    if (cur_R_dist > 3 * min_R_dist) return;

    // The whole warp should iterate through the elements in the current R
    value_idx R_start_offset = R_indptr[cur_R_ind];
    value_idx R_stop_offset  = R_indptr[cur_R_ind + 1];

    value_idx R_size = R_stop_offset - R_start_offset;

    value_int limit = Pow2<WarpSize>::roundDown(R_size);
    value_int i     = threadIdx.x;
    for (; i < limit; i += tpb) {
      // Index and distance of current candidate's nearest landmark
      value_idx cur_candidate_ind = R_1nn_cols[R_start_offset + i];
      value_t cur_candidate_dist  = R_1nn_dists[R_start_offset + i];

      // Take 2 landmarks l_1 and l_2 where l_1 is the furthest point in the heap
      // and l_2 is the current landmark R. s is the current data point and
      // t is the new candidate data point. We know that:
      // d(s, t) cannot possibly be any smaller than | d(s, l_1) - d(l_1, l_2) | * | d(l_1, l_2) -
      // d(l_2, t) | - d(s, l_1) * d(l_2, t)

      // Therefore, if d(s, t) >= d(s, l_1) from the computation above, we know that the distance to
      // the candidate point cannot possibly be in the nearest neighbors. However, if d(s, t) < d(s,
      // l_1) then we should compute the distance because it's possible it could be smaller.
      //
      value_t z = heap.warpKTopRDist == 0.00 ? 0.0
                                             : (abs(heap.warpKTop - heap.warpKTopRDist) *
                                                  abs(heap.warpKTopRDist - cur_candidate_dist) -
                                                heap.warpKTop * cur_candidate_dist) /
                                                 heap.warpKTopRDist;

      z            = isnan(z) || isinf(z) ? 0.0 : z;
      value_t dist = std::numeric_limits<value_t>::max();

      if (z <= heap.warpKTop) {
        const value_t* y_ptr = X_reordered + (n_cols * (R_start_offset + i));
        value_t local_y_ptr[col_q];
        for (value_int j = 0; j < n_cols; ++j) {
          local_y_ptr[j] = y_ptr[j];
        }
        dist = dfunc(local_x_ptr, local_y_ptr, n_cols);
        ++n_dists_computed;
      }

      heap.add(dist, cur_candidate_dist, cur_candidate_ind);
    }

    if (i < R_size) {
      value_idx cur_candidate_ind = R_1nn_cols[R_start_offset + i];
      value_t cur_candidate_dist  = R_1nn_dists[R_start_offset + i];
      value_t z                   = heap.warpKTopRDist == 0.0 ? 0.0
                                                              : (abs(heap.warpKTop - heap.warpKTopRDist) *
                                                 abs(heap.warpKTopRDist - cur_candidate_dist) -
                                               heap.warpKTop * cur_candidate_dist) /
                                                heap.warpKTopRDist;

      z            = isnan(z) || isinf(z) ? 0.0 : z;
      value_t dist = std::numeric_limits<value_t>::max();

      if (z <= heap.warpKTop) {
        const value_t* y_ptr = X_reordered + (n_cols * (R_start_offset + i));
        value_t local_y_ptr[col_q];
        for (value_int j = 0; j < n_cols; ++j) {
          local_y_ptr[j] = y_ptr[j];
        }
        dist = dfunc(local_x_ptr, local_y_ptr, n_cols);
        ++n_dists_computed;
      }

      heap.addThreadQ(dist, cur_candidate_dist, cur_candidate_ind);
    }

    heap.checkThreadQ();
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    out_dists[blockIdx.x * k + i] = shared_memK[i];
    out_inds[blockIdx.x * k + i]  = shared_memV[i].value;
  }
}

template <typename value_t>
__device__ value_t squared(const value_t& a)
{
  return a * a;
}

template <typename value_idx = std::int64_t,
          typename value_t,
          int tpb            = 128,
          typename value_int = std::uint32_t,
          typename distance_func>
RAFT_KERNEL block_rbc_kernel_eps_dense(const value_t* X_reordered,
                                       const value_t* X,
                                       const value_int n_queries,
                                       const value_int n_cols,
                                       const value_t* R,
                                       const value_int m,
                                       const value_t eps,
                                       const value_int n_landmarks,
                                       const value_idx* R_indptr,
                                       const value_idx* R_1nn_cols,
                                       const value_t* R_1nn_dists,
                                       const value_t* R_radius,
                                       distance_func dfunc,
                                       bool* adj,
                                       value_idx* vd)
{
  constexpr int num_warps = tpb / WarpSize;

  // process 1 query per warp
  const uint32_t lid = raft::laneId();

  // this should help the compiler to prevent branches
  const int query_id = raft::shfl(blockIdx.x * num_warps + (threadIdx.x / WarpSize), 0);

  // this is an early out for a full warp
  if (query_id >= n_queries) return;

  value_idx column_count = 0;

  const value_t* x_ptr = X + (n_cols * query_id);
  adj += query_id * m;

  // we omit the sqrt() in the inner distance compute
  const value_t eps2 = eps * eps;

#pragma nounroll
  for (uint32_t cur_k0 = 0; cur_k0 < n_landmarks; cur_k0 += WarpSize) {
    // Pre-compute landmark_dist & triangularization checks for 32 iterations
    const uint32_t lane_k        = cur_k0 + lid;
    const value_t lane_R_dist_sq = lane_k < n_landmarks ? dfunc(x_ptr, R + lane_k * n_cols, n_cols)
                                                        : std::numeric_limits<value_idx>::max();
    const int lane_check         = lane_k < n_landmarks
                                     ? static_cast<int>(lane_R_dist_sq <= squared(eps + R_radius[lane_k]))
                                     : 0;

    int lane_mask = raft::ballot(lane_check);
    if (lane_mask == 0) continue;

    // reverse to use __clz instead of __ffs
    lane_mask = __brev(lane_mask);
    do {
      // look for next k_offset
      const uint32_t k_offset = __clz(lane_mask);

      const uint32_t cur_k = cur_k0 + k_offset;

      // The whole warp should iterate through the elements in the current R
      const value_idx R_start_offset = R_indptr[cur_k];

      // update lane_mask for next iteration - erase bits up to k_offset
      lane_mask &= (0x7fffffff >> k_offset);

      const uint32_t R_size = R_indptr[cur_k + 1] - R_start_offset;

      // we have precomputed the query<->landmark distance
      const value_t cur_R_dist = raft::sqrt(raft::shfl(lane_R_dist_sq, k_offset));

      const uint32_t limit = Pow2<WarpSize>::roundDown(R_size);
      uint32_t i           = limit + lid;

      // R_1nn_dists are sorted ascendingly for each landmark
      // Iterating backwards, after pruning the first point w.r.t. triangle
      // inequality all subsequent points can be pruned as well
      const value_t* y_ptr = X_reordered + (n_cols * (R_start_offset + i));
      {
        const value_t min_warp_dist =
          limit < R_size ? R_1nn_dists[R_start_offset + limit] : cur_R_dist;
        const value_t dist =
          (i < R_size) ? dfunc(x_ptr, y_ptr, n_cols) : std::numeric_limits<value_idx>::max();
        const bool in_range = (dist <= eps2);
        if (in_range) {
          auto index = R_1nn_cols[R_start_offset + i];
          column_count++;
          adj[index] = true;
        }
        // abort in case subsequent points cannot possibly be in reach
        i *= (cur_R_dist - min_warp_dist <= eps);
      }

      uint32_t i0 = raft::shfl(i, 0);

      while (i0 >= WarpSize) {
        y_ptr -= WarpSize * n_cols;
        i0 -= WarpSize;
        const value_t min_warp_dist = R_1nn_dists[R_start_offset + i0];
        const value_t dist          = dfunc(x_ptr, y_ptr, n_cols);
        const bool in_range         = (dist <= eps2);
        if (in_range) {
          auto index = R_1nn_cols[R_start_offset + i0 + lid];
          column_count++;
          adj[index] = true;
        }
        // abort in case subsequent points cannot possibly be in reach
        i0 *= (cur_R_dist - min_warp_dist <= eps);
      }
    } while (lane_mask);
  }

  if (vd != nullptr) {
    value_idx row_sum = raft::warpReduce(column_count);
    if (lid == 0) vd[query_id] = row_sum;
  }
}

template <typename value_idx = std::int64_t,
          typename value_t,
          bool write_pass,
          int tpb            = 128,
          typename value_int = std::uint32_t,
          typename distance_func>
RAFT_KERNEL block_rbc_kernel_eps_csr_pass(const value_t* X_reordered,
                                          const value_t* X,
                                          const value_int n_queries,
                                          const value_int n_cols,
                                          const value_t* R,
                                          const value_int m,
                                          const value_t eps,
                                          const value_int n_landmarks,
                                          const value_idx* R_indptr,
                                          const value_idx* R_1nn_cols,
                                          const value_t* R_1nn_dists,
                                          const value_t* R_radius,
                                          distance_func dfunc,
                                          value_idx* adj_ia,
                                          value_idx* adj_ja)
{
  constexpr int num_warps = tpb / WarpSize;

  // process 1 query per warp
  const uint32_t lid      = raft::laneId();
  const uint32_t lid_mask = (1 << lid) - 1;

  // this should help the compiler to prevent branches
  const int query_id = raft::shfl(blockIdx.x * num_warps + (threadIdx.x / WarpSize), 0);

  // this is an early out for a full warp
  if (query_id >= n_queries) return;

  uint32_t column_index_offset = 0;

  if constexpr (write_pass) {
    value_idx offset = adj_ia[query_id];
    // we have no neighbors to fill for this query
    if (offset == adj_ia[query_id + 1]) return;
    adj_ja += offset;
  }

  const value_t* x_ptr = X + (n_cols * query_id);

  // we omit the sqrt() in the inner distance compute
  const value_t eps2 = eps * eps;

#pragma nounroll
  for (uint32_t cur_k0 = 0; cur_k0 < n_landmarks; cur_k0 += WarpSize) {
    // Pre-compute landmark_dist & triangularization checks for 32 iterations
    const uint32_t lane_k        = cur_k0 + lid;
    const value_t lane_R_dist_sq = lane_k < n_landmarks ? dfunc(x_ptr, R + lane_k * n_cols, n_cols)
                                                        : std::numeric_limits<value_idx>::max();
    const int lane_check         = lane_k < n_landmarks
                                     ? static_cast<int>(lane_R_dist_sq <= squared(eps + R_radius[lane_k]))
                                     : 0;

    int lane_mask = raft::ballot(lane_check);
    if (lane_mask == 0) continue;

    // reverse to use __clz instead of __ffs
    lane_mask = __brev(lane_mask);
    do {
      // look for next k_offset
      const uint32_t k_offset = __clz(lane_mask);

      const uint32_t cur_k = cur_k0 + k_offset;

      // The whole warp should iterate through the elements in the current R
      const value_idx R_start_offset = R_indptr[cur_k];

      // update lane_mask for next iteration - erase bits up to k_offset
      lane_mask &= (0x7fffffff >> k_offset);

      const uint32_t R_size = R_indptr[cur_k + 1] - R_start_offset;

      // we have precomputed the query<->landmark distance
      const value_t cur_R_dist = raft::sqrt(raft::shfl(lane_R_dist_sq, k_offset));

      const uint32_t limit = Pow2<WarpSize>::roundDown(R_size);
      uint32_t i           = limit + lid;

      // R_1nn_dists are sorted ascendingly for each landmark
      // Iterating backwards, after pruning the first point w.r.t. triangle
      // inequality all subsequent points can be pruned as well
      const value_t* y_ptr = X_reordered + (n_cols * (R_start_offset + i));
      {
        const value_t min_warp_dist =
          limit < R_size ? R_1nn_dists[R_start_offset + limit] : cur_R_dist;
        const value_t dist =
          (i < R_size) ? dfunc(x_ptr, y_ptr, n_cols) : std::numeric_limits<value_idx>::max();
        const bool in_range = (dist <= eps2);
        if constexpr (write_pass) {
          const int mask = raft::ballot(in_range);
          if (in_range) {
            const uint32_t index   = R_1nn_cols[R_start_offset + i];
            const uint32_t row_pos = __popc(mask & lid_mask);
            adj_ja[row_pos]        = index;
          }
          adj_ja += __popc(mask);
        } else {
          column_index_offset += (in_range);
        }
        // abort in case subsequent points cannot possibly be in reach
        i *= (cur_R_dist - min_warp_dist <= eps);
      }

      uint32_t i0 = raft::shfl(i, 0);

      while (i0 >= WarpSize) {
        y_ptr -= WarpSize * n_cols;
        i0 -= WarpSize;
        const value_t min_warp_dist = R_1nn_dists[R_start_offset + i0];
        const value_t dist          = dfunc(x_ptr, y_ptr, n_cols);
        const bool in_range         = (dist <= eps2);
        if constexpr (write_pass) {
          const int mask = raft::ballot(in_range);
          if (in_range) {
            const uint32_t index   = R_1nn_cols[R_start_offset + i0 + lid];
            const uint32_t row_pos = __popc(mask & lid_mask);
            adj_ja[row_pos]        = index;
          }
          adj_ja += __popc(mask);
        } else {
          column_index_offset += (in_range);
        }
        // abort in case subsequent points cannot possibly be in reach
        i0 *= (cur_R_dist - min_warp_dist <= eps);
      }
    } while (lane_mask);
  }

  if constexpr (!write_pass) {
    value_idx row_sum = raft::warpReduce(column_index_offset);
    if (lid == 0) adj_ia[query_id] = row_sum;
  }
}

template <typename value_idx = std::int64_t,
          typename value_t,
          bool write_pass,
          int tpb            = 128,
          int dim            = 3,
          typename value_int = std::uint32_t,
          typename distance_func>
RAFT_KERNEL __launch_bounds__(tpb)
  block_rbc_kernel_eps_csr_pass_xd(const value_t* __restrict__ X_reordered,
                                   const value_t* __restrict__ X,
                                   const value_int n_queries,
                                   const value_int n_cols,
                                   const value_t* __restrict__ R,
                                   const value_int m,
                                   const value_t eps,
                                   const value_int n_landmarks,
                                   const value_idx* __restrict__ R_indptr,
                                   const value_idx* __restrict__ R_1nn_cols,
                                   const value_t* __restrict__ R_1nn_dists,
                                   const value_t* __restrict__ R_radius,
                                   distance_func dfunc,
                                   value_idx* __restrict__ adj_ia,
                                   value_idx* adj_ja)
{
  constexpr int num_warps = tpb / WarpSize;

  // process 1 query per warp
  const uint32_t lid      = raft::laneId();
  const uint32_t lid_mask = (1 << lid) - 1;

  // this should help the compiler to prevent branches
  const int query_id = raft::shfl(blockIdx.x * num_warps + (threadIdx.x / WarpSize), 0);

  // this is an early out for a full warp
  if (query_id >= n_queries) return;

  uint32_t column_index_offset = 0;

  if constexpr (write_pass) {
    value_idx offset = adj_ia[query_id];
    // we have no neighbors to fill for this query
    if (offset == adj_ia[query_id + 1]) return;
    adj_ja += offset;
  }

  const value_t* x_ptr = X + (dim * query_id);
  value_t local_x_ptr[dim];
#pragma unroll
  for (uint32_t i = 0; i < dim; ++i) {
    local_x_ptr[i] = x_ptr[i];
  }

  // we omit the sqrt() in the inner distance compute
  const value_t eps2 = eps * eps;

#pragma nounroll
  for (uint32_t cur_k0 = 0; cur_k0 < n_landmarks; cur_k0 += WarpSize) {
    // Pre-compute landmark_dist & triangularization checks for 32 iterations
    const uint32_t lane_k        = cur_k0 + lid;
    const value_t lane_R_dist_sq = lane_k < n_landmarks ? dfunc(local_x_ptr, R + lane_k * dim, dim)
                                                        : std::numeric_limits<value_idx>::max();
    const int lane_check         = lane_k < n_landmarks
                                     ? static_cast<int>(lane_R_dist_sq <= squared(eps + R_radius[lane_k]))
                                     : 0;

    int lane_mask = raft::ballot(lane_check);
    if (lane_mask == 0) continue;

    // reverse to use __clz instead of __ffs
    lane_mask = __brev(lane_mask);
    do {
      // look for next k_offset
      const uint32_t k_offset = __clz(lane_mask);

      const uint32_t cur_k = cur_k0 + k_offset;

      // The whole warp should iterate through the elements in the current R
      const value_idx R_start_offset = R_indptr[cur_k];

      // update lane_mask for next iteration - erase bits up to k_offset
      lane_mask &= (0x7fffffff >> k_offset);

      const uint32_t R_size = R_indptr[cur_k + 1] - R_start_offset;

      // we have precomputed the query<->landmark distance
      const value_t cur_R_dist = raft::sqrt(raft::shfl(lane_R_dist_sq, k_offset));

      const uint32_t limit = Pow2<WarpSize>::roundDown(R_size);
      uint32_t i           = limit + lid;

      // R_1nn_dists are sorted ascendingly for each landmark
      // Iterating backwards, after pruning the first point w.r.t. triangle
      // inequality all subsequent points can be pruned as well
      const value_t* y_ptr = X_reordered + (dim * (R_start_offset + i));
      {
        const value_t min_warp_dist =
          limit < R_size ? R_1nn_dists[R_start_offset + limit] : cur_R_dist;
        const value_t dist =
          (i < R_size) ? dfunc(local_x_ptr, y_ptr, dim) : std::numeric_limits<value_idx>::max();
        const bool in_range = (dist <= eps2);
        if constexpr (write_pass) {
          const int mask = raft::ballot(in_range);
          if (in_range) {
            const uint32_t index   = R_1nn_cols[R_start_offset + i];
            const uint32_t row_pos = __popc(mask & lid_mask);
            adj_ja[row_pos]        = index;
          }
          adj_ja += __popc(mask);
        } else {
          column_index_offset += (in_range);
        }
        // abort in case subsequent points cannot possibly be in reach
        i *= (cur_R_dist - min_warp_dist <= eps);
      }

      uint32_t i0 = raft::shfl(i, 0);

      while (i0 >= WarpSize) {
        y_ptr -= WarpSize * dim;
        i0 -= WarpSize;
        const value_t min_warp_dist = R_1nn_dists[R_start_offset + i0];
        const value_t dist          = dfunc(local_x_ptr, y_ptr, dim);
        const bool in_range         = (dist <= eps2);
        if constexpr (write_pass) {
          const int mask = raft::ballot(in_range);
          if (in_range) {
            const uint32_t index   = R_1nn_cols[R_start_offset + i0 + lid];
            const uint32_t row_pos = __popc(mask & lid_mask);
            adj_ja[row_pos]        = index;
          }
          adj_ja += __popc(mask);
        } else {
          column_index_offset += (in_range);
        }
        // abort in case subsequent points cannot possibly be in reach
        i0 *= (cur_R_dist - min_warp_dist <= eps);
      }
    } while (lane_mask);
  }

  if constexpr (!write_pass) {
    value_idx row_sum = raft::warpReduce(column_index_offset);
    if (lid == 0) adj_ia[query_id] = row_sum;
  }
}

template <typename value_idx = std::int64_t,
          typename value_t,
          int tpb            = 128,
          typename value_int = std::uint32_t,
          typename distance_func>
RAFT_KERNEL block_rbc_kernel_eps_max_k(const value_t* X_reordered,
                                       const value_t* X,
                                       const value_int n_queries,
                                       const value_int n_cols,
                                       const value_t* R,
                                       const value_int m,
                                       const value_t eps,
                                       const value_int n_landmarks,
                                       const value_idx* R_indptr,
                                       const value_idx* R_1nn_cols,
                                       const value_t* R_1nn_dists,
                                       const value_t* R_radius,
                                       distance_func dfunc,
                                       value_idx* vd,
                                       const value_int max_k,
                                       value_idx* tmp)
{
  constexpr int num_warps = tpb / WarpSize;

  // process 1 query per warp
  const uint32_t lid      = raft::laneId();
  const uint32_t lid_mask = (1 << lid) - 1;

  // this should help the compiler to prevent branches
  const int query_id = raft::shfl(blockIdx.x * num_warps + (threadIdx.x / WarpSize), 0);

  // this is an early out for a full warp
  if (query_id >= n_queries) return;

  value_idx column_count = 0;

  const value_t* x_ptr = X + (n_cols * query_id);
  tmp += query_id * max_k;

  // we omit the sqrt() in the inner distance compute
  const value_t eps2 = eps * eps;

#pragma nounroll
  for (uint32_t cur_k0 = 0; cur_k0 < n_landmarks; cur_k0 += WarpSize) {
    // Pre-compute landmark_dist & triangularization checks for 32 iterations
    const uint32_t lane_k        = cur_k0 + lid;
    const value_t lane_R_dist_sq = lane_k < n_landmarks ? dfunc(x_ptr, R + lane_k * n_cols, n_cols)
                                                        : std::numeric_limits<value_idx>::max();
    const int lane_check         = lane_k < n_landmarks
                                     ? static_cast<int>(lane_R_dist_sq <= squared(eps + R_radius[lane_k]))
                                     : 0;

    int lane_mask = raft::ballot(lane_check);
    if (lane_mask == 0) continue;

    // reverse to use __clz instead of __ffs
    lane_mask = __brev(lane_mask);
    do {
      // look for next k_offset
      const uint32_t k_offset = __clz(lane_mask);

      const uint32_t cur_k = cur_k0 + k_offset;

      // The whole warp should iterate through the elements in the current R
      const value_idx R_start_offset = R_indptr[cur_k];

      // update lane_mask for next iteration - erase bits up to k_offset
      lane_mask &= (0x7fffffff >> k_offset);

      const uint32_t R_size = R_indptr[cur_k + 1] - R_start_offset;

      // we have precomputed the query<->landmark distance
      const value_t cur_R_dist = raft::sqrt(raft::shfl(lane_R_dist_sq, k_offset));

      const uint32_t limit = Pow2<WarpSize>::roundDown(R_size);
      uint32_t i           = limit + lid;

      // R_1nn_dists are sorted ascendingly for each landmark
      // Iterating backwards, after pruning the first point w.r.t. triangle
      // inequality all subsequent points can be pruned as well
      const value_t* y_ptr = X_reordered + (n_cols * (R_start_offset + i));
      {
        const value_t min_warp_dist =
          limit < R_size ? R_1nn_dists[R_start_offset + limit] : cur_R_dist;
        const value_t dist =
          (i < R_size) ? dfunc(x_ptr, y_ptr, n_cols) : std::numeric_limits<value_idx>::max();
        const bool in_range = (dist <= eps2);
        const int mask      = raft::ballot(in_range);
        if (in_range) {
          auto row_pos = column_count + __popc(mask & lid_mask);
          // we still continue to look for more hits to return valid vd
          if (row_pos < max_k) {
            auto index   = R_1nn_cols[R_start_offset + i];
            tmp[row_pos] = index;
          }
        }
        column_count += __popc(mask);
        // abort in case subsequent points cannot possibly be in reach
        i *= (cur_R_dist - min_warp_dist <= eps);
      }

      uint32_t i0 = raft::shfl(i, 0);

      while (i0 >= WarpSize) {
        y_ptr -= WarpSize * n_cols;
        i0 -= WarpSize;
        const value_t min_warp_dist = R_1nn_dists[R_start_offset + i0];
        const value_t dist          = dfunc(x_ptr, y_ptr, n_cols);
        const bool in_range         = (dist <= eps2);
        const int mask              = raft::ballot(in_range);
        if (in_range) {
          auto row_pos = column_count + __popc(mask & lid_mask);
          // we still continue to look for more hits to return valid vd
          if (row_pos < max_k) {
            auto index   = R_1nn_cols[R_start_offset + i0 + lid];
            tmp[row_pos] = index;
          }
        }
        column_count += __popc(mask);
        // abort in case subsequent points cannot possibly be in reach
        i0 *= (cur_R_dist - min_warp_dist <= eps);
      }
    } while (lane_mask);
  }

  if (lid == 0) vd[query_id] = column_count;
}

template <typename value_idx = std::int64_t, int tpb = 128, typename value_int = std::uint32_t>
RAFT_KERNEL block_rbc_kernel_eps_max_k_copy(const value_int max_k,
                                            const value_idx* adj_ia,
                                            const value_idx* tmp,
                                            value_idx* adj_ja)
{
  value_int offset = blockIdx.x * max_k;

  value_int row_idx       = blockIdx.x;
  value_idx col_start_idx = adj_ia[row_idx];
  value_idx num_cols      = adj_ia[row_idx + 1] - col_start_idx;

  value_int limit = Pow2<WarpSize>::roundDown(num_cols);
  value_int i     = threadIdx.x;
  for (; i < limit; i += tpb) {
    adj_ja[col_start_idx + i] = tmp[offset + i];
  }
  if (i < num_cols) { adj_ja[col_start_idx + i] = tmp[offset + i]; }
}

template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          int dims            = 2,
          typename dist_func>
void rbc_low_dim_pass_one(raft::resources const& handle,
                          const BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
                          const value_t* query,
                          const value_int n_query_rows,
                          value_int k,
                          const value_idx* R_knn_inds,
                          const value_t* R_knn_dists,
                          dist_func& dfunc,
                          value_idx* inds,
                          value_t* dists,
                          float weight,
                          value_int* dists_counter)
{
  if (k <= 32)
    block_rbc_kernel_registers<value_idx, value_t, 32, 2, 128, dims, value_int>
      <<<n_query_rows, 128, 0, resource::get_cuda_stream(handle)>>>(
        index.get_X_reordered().data_handle(),
        query,
        index.n,
        R_knn_inds,
        R_knn_dists,
        index.m,
        k,
        index.get_R_indptr().data_handle(),
        index.get_R_1nn_cols().data_handle(),
        index.get_R_1nn_dists().data_handle(),
        inds,
        dists,
        dists_counter,
        index.get_R_radius().data_handle(),
        dfunc,
        weight);

  else if (k <= 64)
    block_rbc_kernel_registers<value_idx, value_t, 64, 3, 128, 2, value_int>
      <<<n_query_rows, 128, 0, resource::get_cuda_stream(handle)>>>(
        index.get_X_reordered().data_handle(),
        query,
        index.n,
        R_knn_inds,
        R_knn_dists,
        index.m,
        k,
        index.get_R_indptr().data_handle(),
        index.get_R_1nn_cols().data_handle(),
        index.get_R_1nn_dists().data_handle(),
        inds,
        dists,
        dists_counter,
        index.get_R_radius().data_handle(),
        dfunc,
        weight);
  else if (k <= 128)
    block_rbc_kernel_registers<value_idx, value_t, 128, 3, 128, dims, value_int>
      <<<n_query_rows, 128, 0, resource::get_cuda_stream(handle)>>>(
        index.get_X_reordered().data_handle(),
        query,
        index.n,
        R_knn_inds,
        R_knn_dists,
        index.m,
        k,
        index.get_R_indptr().data_handle(),
        index.get_R_1nn_cols().data_handle(),
        index.get_R_1nn_dists().data_handle(),
        inds,
        dists,
        dists_counter,
        index.get_R_radius().data_handle(),
        dfunc,
        weight);

  else if (k <= 256)
    block_rbc_kernel_registers<value_idx, value_t, 256, 4, 128, dims, value_int>
      <<<n_query_rows, 128, 0, resource::get_cuda_stream(handle)>>>(
        index.get_X_reordered().data_handle(),
        query,
        index.n,
        R_knn_inds,
        R_knn_dists,
        index.m,
        k,
        index.get_R_indptr().data_handle(),
        index.get_R_1nn_cols().data_handle(),
        index.get_R_1nn_dists().data_handle(),
        inds,
        dists,
        dists_counter,
        index.get_R_radius().data_handle(),
        dfunc,
        weight);

  else if (k <= 512)
    block_rbc_kernel_registers<value_idx, value_t, 512, 8, 64, dims, value_int>
      <<<n_query_rows, 64, 0, resource::get_cuda_stream(handle)>>>(
        index.get_X_reordered().data_handle(),
        query,
        index.n,
        R_knn_inds,
        R_knn_dists,
        index.m,
        k,
        index.get_R_indptr().data_handle(),
        index.get_R_1nn_cols().data_handle(),
        index.get_R_1nn_dists().data_handle(),
        inds,
        dists,
        dists_counter,
        index.get_R_radius().data_handle(),
        dfunc,
        weight);

  else if (k <= 1024)
    block_rbc_kernel_registers<value_idx, value_t, 1024, 8, 64, dims, value_int>
      <<<n_query_rows, 64, 0, resource::get_cuda_stream(handle)>>>(
        index.get_X_reordered().data_handle(),
        query,
        index.n,
        R_knn_inds,
        R_knn_dists,
        index.m,
        k,
        index.get_R_indptr().data_handle(),
        index.get_R_1nn_cols().data_handle(),
        index.get_R_1nn_dists().data_handle(),
        inds,
        dists,
        dists_counter,
        index.get_R_radius().data_handle(),
        dfunc,
        weight);
}

template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          int dims            = 2,
          typename dist_func>
void rbc_low_dim_pass_two(raft::resources const& handle,
                          const BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
                          const value_t* query,
                          const value_int n_query_rows,
                          value_int k,
                          const value_idx* R_knn_inds,
                          const value_t* R_knn_dists,
                          dist_func& dfunc,
                          value_idx* inds,
                          value_t* dists,
                          float weight,
                          value_int* post_dists_counter)
{
  const value_int bitset_size = ceil(index.n_landmarks / 32.0);

  rmm::device_uvector<std::uint32_t> bitset(bitset_size * n_query_rows,
                                            resource::get_cuda_stream(handle));
  thrust::fill(
    resource::get_thrust_policy(handle), bitset.data(), bitset.data() + bitset.size(), 0);

  perform_post_filter_registers<value_idx, value_t, value_int, dims, 128>
    <<<n_query_rows, 128, bitset_size * sizeof(std::uint32_t), resource::get_cuda_stream(handle)>>>(
      query,
      index.n,
      R_knn_inds,
      R_knn_dists,
      index.get_R_radius().data_handle(),
      index.get_R().data_handle(),
      index.n_landmarks,
      bitset_size,
      k,
      dfunc,
      bitset.data(),
      weight);

  if (k <= 32)
    compute_final_dists_registers<value_idx,
                                  value_t,
                                  value_int,
                                  std::uint32_t,
                                  dist_func,
                                  32,
                                  2,
                                  128,
                                  dims>
      <<<n_query_rows, 128, 0, resource::get_cuda_stream(handle)>>>(
        index.get_X_reordered().data_handle(),
        query,
        index.n,
        bitset.data(),
        bitset_size,
        index.get_R_closest_landmark_dists().data_handle(),
        index.get_R_indptr().data_handle(),
        index.get_R_1nn_cols().data_handle(),
        index.get_R_1nn_dists().data_handle(),
        inds,
        dists,
        index.n_landmarks,
        k,
        dfunc,
        post_dists_counter);
  else if (k <= 64)
    compute_final_dists_registers<value_idx,
                                  value_t,
                                  value_int,
                                  std::uint32_t,
                                  dist_func,
                                  64,
                                  3,
                                  128,
                                  dims>
      <<<n_query_rows, 128, 0, resource::get_cuda_stream(handle)>>>(
        index.get_X_reordered().data_handle(),
        query,
        index.n,
        bitset.data(),
        bitset_size,
        index.get_R_closest_landmark_dists().data_handle(),
        index.get_R_indptr().data_handle(),
        index.get_R_1nn_cols().data_handle(),
        index.get_R_1nn_dists().data_handle(),
        inds,
        dists,
        index.n_landmarks,
        k,
        dfunc,
        post_dists_counter);
  else if (k <= 128)
    compute_final_dists_registers<value_idx,
                                  value_t,
                                  value_int,
                                  std::uint32_t,
                                  dist_func,
                                  128,
                                  3,
                                  128,
                                  dims>
      <<<n_query_rows, 128, 0, resource::get_cuda_stream(handle)>>>(
        index.get_X_reordered().data_handle(),
        query,
        index.n,
        bitset.data(),
        bitset_size,
        index.get_R_closest_landmark_dists().data_handle(),
        index.get_R_indptr().data_handle(),
        index.get_R_1nn_cols().data_handle(),
        index.get_R_1nn_dists().data_handle(),
        inds,
        dists,
        index.n_landmarks,
        k,
        dfunc,
        post_dists_counter);
  else if (k <= 256)
    compute_final_dists_registers<value_idx,
                                  value_t,
                                  value_int,
                                  std::uint32_t,
                                  dist_func,
                                  256,
                                  4,
                                  128,
                                  dims>
      <<<n_query_rows, 128, 0, resource::get_cuda_stream(handle)>>>(
        index.get_X_reordered().data_handle(),
        query,
        index.n,
        bitset.data(),
        bitset_size,
        index.get_R_closest_landmark_dists().data_handle(),
        index.get_R_indptr().data_handle(),
        index.get_R_1nn_cols().data_handle(),
        index.get_R_1nn_dists().data_handle(),
        inds,
        dists,
        index.n_landmarks,
        k,
        dfunc,
        post_dists_counter);
  else if (k <= 512)
    compute_final_dists_registers<value_idx,
                                  value_t,
                                  value_int,
                                  std::uint32_t,
                                  dist_func,
                                  512,
                                  8,
                                  64,
                                  dims><<<n_query_rows, 64, 0, resource::get_cuda_stream(handle)>>>(
      index.get_X_reordered().data_handle(),
      query,
      index.n,
      bitset.data(),
      bitset_size,
      index.get_R_closest_landmark_dists().data_handle(),
      index.get_R_indptr().data_handle(),
      index.get_R_1nn_cols().data_handle(),
      index.get_R_1nn_dists().data_handle(),
      inds,
      dists,
      index.n_landmarks,
      k,
      dfunc,
      post_dists_counter);
  else if (k <= 1024)
    compute_final_dists_registers<value_idx,
                                  value_t,
                                  value_int,
                                  std::uint32_t,
                                  dist_func,
                                  1024,
                                  8,
                                  64,
                                  dims><<<n_query_rows, 64, 0, resource::get_cuda_stream(handle)>>>(
      index.get_X_reordered().data_handle(),
      query,
      index.n,
      bitset.data(),
      bitset_size,
      index.get_R_closest_landmark_dists().data_handle(),
      index.get_R_indptr().data_handle(),
      index.get_R_1nn_cols().data_handle(),
      index.get_R_1nn_dists().data_handle(),
      inds,
      dists,
      index.n_landmarks,
      k,
      dfunc,
      post_dists_counter);
}

template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          typename dist_func>
void rbc_eps_pass(raft::resources const& handle,
                  const BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
                  const value_t* query,
                  const value_int n_query_rows,
                  value_t eps,
                  const value_t* R,
                  dist_func& dfunc,
                  bool* adj,
                  value_idx* vd)
{
  block_rbc_kernel_eps_dense<value_idx, value_t, 64, value_int>
    <<<n_query_rows, 64, 0, resource::get_cuda_stream(handle)>>>(
      index.get_X_reordered().data_handle(),
      query,
      n_query_rows,
      index.n,
      R,
      index.m,
      eps,
      index.n_landmarks,
      index.get_R_indptr().data_handle(),
      index.get_R_1nn_cols().data_handle(),
      index.get_R_1nn_dists().data_handle(),
      index.get_R_radius().data_handle(),
      dfunc,
      adj,
      vd);

  if (vd != nullptr) {
    value_idx sum = thrust::reduce(resource::get_thrust_policy(handle), vd, vd + n_query_rows);
    // copy sum to last element
    RAFT_CUDA_TRY(cudaMemcpyAsync(vd + n_query_rows,
                                  &sum,
                                  sizeof(value_idx),
                                  cudaMemcpyHostToDevice,
                                  resource::get_cuda_stream(handle)));
  }

  resource::sync_stream(handle);
}

template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          typename dist_func>
void rbc_eps_pass(raft::resources const& handle,
                  const BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
                  const value_t* query,
                  const value_int n_query_rows,
                  value_t eps,
                  value_int* max_k,
                  const value_t* R,
                  dist_func& dfunc,
                  value_idx* adj_ia,
                  value_idx* adj_ja,
                  value_idx* vd)
{
  // if max_k == nullptr we are either pass 1 or pass 2
  if (max_k == nullptr) {
    if (adj_ja == nullptr) {
      // pass 1 -> only compute adj_ia / vd
      value_idx* vd_ptr = (vd != nullptr) ? vd : adj_ia;
      if (index.n == 2) {
        block_rbc_kernel_eps_csr_pass_xd<value_idx, value_t, false, 64, 2, value_int>
          <<<raft::ceildiv<value_int>(n_query_rows, 2), 64, 0, resource::get_cuda_stream(handle)>>>(
            index.get_X_reordered().data_handle(),
            query,
            n_query_rows,
            index.n,
            R,
            index.m,
            eps,
            index.n_landmarks,
            index.get_R_indptr().data_handle(),
            index.get_R_1nn_cols().data_handle(),
            index.get_R_1nn_dists().data_handle(),
            index.get_R_radius().data_handle(),
            dfunc,
            vd_ptr,
            nullptr);
      } else if (index.n == 3) {
        block_rbc_kernel_eps_csr_pass_xd<value_idx, value_t, false, 64, 3, value_int>
          <<<raft::ceildiv<value_int>(n_query_rows, 2), 64, 0, resource::get_cuda_stream(handle)>>>(
            index.get_X_reordered().data_handle(),
            query,
            n_query_rows,
            index.n,
            R,
            index.m,
            eps,
            index.n_landmarks,
            index.get_R_indptr().data_handle(),
            index.get_R_1nn_cols().data_handle(),
            index.get_R_1nn_dists().data_handle(),
            index.get_R_radius().data_handle(),
            dfunc,
            vd_ptr,
            nullptr);
      } else {
        block_rbc_kernel_eps_csr_pass<value_idx, value_t, false, 64, value_int>
          <<<raft::ceildiv<value_int>(n_query_rows, 2), 64, 0, resource::get_cuda_stream(handle)>>>(
            index.get_X_reordered().data_handle(),
            query,
            n_query_rows,
            index.n,
            R,
            index.m,
            eps,
            index.n_landmarks,
            index.get_R_indptr().data_handle(),
            index.get_R_1nn_cols().data_handle(),
            index.get_R_1nn_dists().data_handle(),
            index.get_R_radius().data_handle(),
            dfunc,
            vd_ptr,
            nullptr);
      }

      thrust::exclusive_scan(resource::get_thrust_policy(handle),
                             vd_ptr,
                             vd_ptr + n_query_rows + 1,
                             adj_ia,
                             (value_idx)0);

    } else {
      // pass 2 -> fill in adj_ja
      if (index.n == 2) {
        block_rbc_kernel_eps_csr_pass_xd<value_idx, value_t, true, 64, 2, value_int>
          <<<raft::ceildiv<value_int>(n_query_rows, 2), 64, 0, resource::get_cuda_stream(handle)>>>(
            index.get_X_reordered().data_handle(),
            query,
            n_query_rows,
            index.n,
            R,
            index.m,
            eps,
            index.n_landmarks,
            index.get_R_indptr().data_handle(),
            index.get_R_1nn_cols().data_handle(),
            index.get_R_1nn_dists().data_handle(),
            index.get_R_radius().data_handle(),
            dfunc,
            adj_ia,
            adj_ja);
      } else if (index.n == 3) {
        block_rbc_kernel_eps_csr_pass_xd<value_idx, value_t, true, 64, 3, value_int>
          <<<raft::ceildiv<value_int>(n_query_rows, 2), 64, 0, resource::get_cuda_stream(handle)>>>(
            index.get_X_reordered().data_handle(),
            query,
            n_query_rows,
            index.n,
            R,
            index.m,
            eps,
            index.n_landmarks,
            index.get_R_indptr().data_handle(),
            index.get_R_1nn_cols().data_handle(),
            index.get_R_1nn_dists().data_handle(),
            index.get_R_radius().data_handle(),
            dfunc,
            adj_ia,
            adj_ja);
      } else {
        block_rbc_kernel_eps_csr_pass<value_idx, value_t, true, 64, value_int>
          <<<raft::ceildiv<value_int>(n_query_rows, 2), 64, 0, resource::get_cuda_stream(handle)>>>(
            index.get_X_reordered().data_handle(),
            query,
            n_query_rows,
            index.n,
            R,
            index.m,
            eps,
            index.n_landmarks,
            index.get_R_indptr().data_handle(),
            index.get_R_1nn_cols().data_handle(),
            index.get_R_1nn_dists().data_handle(),
            index.get_R_radius().data_handle(),
            dfunc,
            adj_ia,
            adj_ja);
      }
    }
  } else {
    value_int max_k_in = *max_k;
    value_idx* vd_ptr  = (vd != nullptr) ? vd : adj_ia;

    rmm::device_uvector<value_idx> tmp(n_query_rows * max_k_in, resource::get_cuda_stream(handle));

    block_rbc_kernel_eps_max_k<value_idx, value_t, 64, value_int>
      <<<raft::ceildiv<value_int>(n_query_rows, 2), 64, 0, resource::get_cuda_stream(handle)>>>(
        index.get_X_reordered().data_handle(),
        query,
        n_query_rows,
        index.n,
        R,
        index.m,
        eps,
        index.n_landmarks,
        index.get_R_indptr().data_handle(),
        index.get_R_1nn_cols().data_handle(),
        index.get_R_1nn_dists().data_handle(),
        index.get_R_radius().data_handle(),
        dfunc,
        vd_ptr,
        max_k_in,
        tmp.data());

    value_int actual_max = thrust::reduce(resource::get_thrust_policy(handle),
                                          vd_ptr,
                                          vd_ptr + n_query_rows,
                                          (value_idx)0,
                                          thrust::maximum<value_idx>());

    if (actual_max > max_k_in) {
      // ceil vd to max_k
      thrust::transform(resource::get_thrust_policy(handle),
                        vd_ptr,
                        vd_ptr + n_query_rows,
                        vd_ptr,
                        [max_k_in] __device__(value_idx vd_count) {
                          return vd_count > max_k_in ? max_k_in : vd_count;
                        });
    }

    thrust::exclusive_scan(
      resource::get_thrust_policy(handle), vd_ptr, vd_ptr + n_query_rows + 1, adj_ia, (value_idx)0);

    block_rbc_kernel_eps_max_k_copy<value_idx, 32, value_int>
      <<<n_query_rows, 32, 0, resource::get_cuda_stream(handle)>>>(
        max_k_in, adj_ia, tmp.data(), adj_ja);

    // return 'new' max-k
    *max_k = actual_max;
  }

  if (vd != nullptr && (max_k != nullptr || adj_ja == nullptr)) {
    // copy sum to last element
    RAFT_CUDA_TRY(cudaMemcpyAsync(vd + n_query_rows,
                                  adj_ia + n_query_rows,
                                  sizeof(value_idx),
                                  cudaMemcpyDeviceToDevice,
                                  resource::get_cuda_stream(handle)));
  }

  resource::sync_stream(handle);
}

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft
