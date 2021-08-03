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

#include <raft/handle.hpp>
#include "haversine_distance.cuh"
#include "knn_brute_force_faiss.cuh"
#include "selection_faiss.cuh"
#include "warp_select_faiss.cuh"

#include <limits.h>

#include <raft/cuda_utils.cuh>

#include <raft/matrix/matrix.cuh>
#include <raft/random/rng.cuh>
#include <raft/sparse/convert/csr.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <faiss/utils/Heap.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>

#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {


struct NNComp {
  template <typename one, typename two>
  __host__ __device__ bool operator()(const one &t1, const two &t2) {
    // sort first by each sample's reference landmark,
    if (thrust::get<0>(t1) < thrust::get<0>(t2)) return true;
    if (thrust::get<0>(t1) > thrust::get<0>(t2)) return false;

    // then by closest neighbor,
    return thrust::get<1>(t1) < thrust::get<1>(t2);
  }
};

/**
 * Zeros the bit at location h in a one-hot encoded 32-bit int array
 */
__device__ inline void _zero_bit(uint32_t *arr, uint32_t h) {
  int bit = h % 32;
  int idx = h / 32;

  uint32_t assumed;
  uint32_t old = arr[idx];
  do {
    assumed = old;
    old = atomicCAS(arr+idx, assumed, assumed & ~(1 << bit));
  } while(assumed != old);
}

/**
 * Returns whether or not bit at location h is nonzero in a one-hot
 * encoded 32-bit in array.
 */
__device__ inline bool _get_val(uint32_t *arr, uint32_t h) {
  int bit = h % 32;
  int idx = h / 32;
  return (arr[idx] & (1 << bit)) > 0;
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
template<typename value_idx, typename value_t, typename value_int = int,
  typename bitset_type = uint32_t, int warp_q=1024, int thread_q=1, int tpb=32>
__global__ void compute_final_dists(const value_t *X,
                                    const value_int n_cols,
                                    bitset_type *bitset,
                                    int bitset_size,
                                    const value_t *R_knn_dists,
                                    const value_idx *R_indptr,
                                    const value_idx *R_1nn_inds,
                                    const value_t *R_1nn_dists,
                                    value_idx *knn_inds,
                                    value_t *knn_dists,
                                    int n_landmarks,
                                    int k,
                                    value_int *dist_counter) {

  value_idx row = blockIdx.x;
  int tid = threadIdx.x;

  size_t n_dists_computed = 0;

  const value_t *x_ptr = X + (n_cols * row);
  value_t x1 = x_ptr[0];
  value_t x2 = x_ptr[1];

  faiss::gpu::KeyValueWarpSelect<value_t, value_idx, false,
    faiss::gpu::Comparator<value_t>, warp_q,
    thread_q, tpb>
    heap(faiss::gpu::Limits<value_t>::getMax(),
         faiss::gpu::KeyValuePair<value_t, value_idx>(
           faiss::gpu::Limits<value_t>::getMax(), -1),
         k);

  // First add current top k to the k-selector
  // TODO: When k > warp size, we need to do this in 2 steps

  int n_k = k > 32 ? floor(k / 32) * 32 : k;
  for(int i = tid; i < n_k; i+= blockDim.x) {
    value_idx ind = knn_inds[row * k + i];
    value_t cur_candidate_dist = R_knn_dists[ind * k];
    heap.addThreadQ(knn_dists[row * k + i],
                    faiss::gpu::KeyValuePair(cur_candidate_dist, ind));
  }

  heap.checkThreadQ();

  for(int cur_R_ind = 0; cur_R_ind < n_landmarks; cur_R_ind++) {

    // if cur R overlaps cur point's closest R, it could be a
    // candidate
    if(_get_val(bitset, cur_R_ind)) {

      value_idx R_start_offset = R_indptr[cur_R_ind];
      value_idx R_stop_offset = R_indptr[cur_R_ind + 1];
      value_idx R_size = R_stop_offset - R_start_offset;

      // Loop through R's neighborhood in parallel

      // Round R_size to the nearest warp threads so they can
      // all be computing in parallel.

      int n_r = floor(R_size / 32) * 32; // round to warp size
      int i = tid;
      for(; i < n_r; i += blockDim.x) {

        value_idx cur_candidate_ind = R_1nn_inds[R_start_offset + i];
        value_t   cur_candidate_dist = R_1nn_dists[R_start_offset + i];
        value_t z = heap.warpKTopRDist == 0.00
                    ? 0.0
                    : (abs(heap.warpKTop - heap.warpKTopRDist) *
                       abs(heap.warpKTopRDist - cur_candidate_dist) -
                       heap.warpKTop * cur_candidate_dist) /
                      heap.warpKTopRDist;

        if(row == 326622 && (cur_candidate_ind == 326622 || cur_candidate_ind == 326720 || cur_candidate_ind == 326623))
          printf("post_filter: cur_candidate_ind=%ld, z=%f, heap.warKTop=%f, heap.warpKTopRDist=%f, cur_candidate_dist=%f\n",
                 cur_candidate_ind, z, heap.warpKTop, heap.warpKTopRDist, cur_candidate_dist);

        // If lower bound on distance could possibly be in
        // the closest k neighbors, compute it and add to k-select
        value_t dist = std::numeric_limits<value_t>::max();
        if (z < heap.warpKTop) {
          const value_t *y_ptr = X + (n_cols * cur_candidate_ind);
          value_t y1 = y_ptr[0];
          value_t y2 = y_ptr[1];

          dist = compute_haversine(x1, y1, x2, y2);
          n_dists_computed++;
        }
        heap.add(
          dist, faiss::gpu::KeyValuePair(cur_candidate_dist, cur_candidate_ind));
      }

      __syncthreads();

      // second round guarantees to be 1 or less per warp
      if(i < R_size) {

        value_idx cur_candidate_ind = R_1nn_inds[R_start_offset + i];
        value_t   cur_candidate_dist = R_1nn_dists[R_start_offset + i];

        value_t z = heap.warpKTopRDist == 0.00
                    ? 0.0
                    : (abs(heap.warpKTop - heap.warpKTopRDist) *
                       abs(heap.warpKTopRDist - cur_candidate_dist) -
                       heap.warpKTop * cur_candidate_dist) /
                      heap.warpKTopRDist;

        if(row == 326622 && (cur_candidate_ind == 326622 || cur_candidate_ind == 326720 || cur_candidate_ind == 326623))
          printf("post_filter: cur_candidate_ind=%ld, z=%f, heap.warKTop=%f, heap.warpKTopRDist=%f, cur_candidate_dist=%f\n",
                 cur_candidate_ind, z, heap.warpKTop, heap.warpKTopRDist, cur_candidate_dist);


        // If lower bound on distance could possibly be in
        // the closest k neighbors, compute it and add to k-select
        value_t dist = std::numeric_limits<value_t>::max();
        if (z < heap.warpKTop) {
          const value_t *y_ptr = X + (n_cols * cur_candidate_ind);
          value_t y1 = y_ptr[0];
          value_t y2 = y_ptr[1];
          dist = compute_haversine(x1, y1, x2, y2);
          n_dists_computed++;
        }
        heap.addThreadQ(
          dist, faiss::gpu::KeyValuePair(cur_candidate_dist, cur_candidate_ind));
      }
    }
  }

  heap.reduce();

  value_idx cur_idx = row * k;
  heap.writeOut(knn_dists + cur_idx, knn_inds + cur_idx, k);
  atomicAdd(dist_counter + row, n_dists_computed);
}

template <typename value_idx, typename value_t, typename value_int = int, int tpb=32>
__global__ void perform_post_filter(const value_t *X,
                                    value_int n_cols,
                                    const value_idx *R_knn_inds,
                                    const value_t *R_knn_dists,
                                    const value_t *R_radius,
                                    const value_t *landmarks,
                                    int n_landmarks,
                                    int bitset_size,
                                    int k,
                                    uint32_t *output) {
  value_idx row = blockIdx.x;
  int tid = threadIdx.x;

  const value_t *x_ptr = X + (n_cols * row);
  value_t x1 = x_ptr[0];
  value_t x2 = x_ptr[1];

  // allocate array of size n_landmarks / 32 ints
  extern __shared__ uint32_t smem[];

  // Start with all bits on
  for(int i = tid; i < bitset_size; i+=blockDim.x)
    smem[i] = 0xffffffff;

  __syncthreads();

  value_t closest_R_dist = R_knn_dists[row * k];

  // zero out bits for closest k landmarks
  for(int j = tid; j < k; j += blockDim.x) {
    _zero_bit(smem, (uint32_t)R_knn_inds[row * k + j]);
  }

  __syncthreads();

  // Discard any landmarks where p(q, r) > p(q, r_q) + radius(r)
  // That is, the distance between the current point and the current
  // landmark is > the distance between the current point and
  // its closest landmark + the radius of the current landmark.
  for(int k = tid; k < n_landmarks; k+=blockDim.x) {

    // compute p(q, r)
    const value_t *y_ptr = landmarks + (n_cols * k);
    value_t y1 = y_ptr[0];
    value_t y2 = y_ptr[1];

    value_t p_q_r = compute_haversine(x1, y1, x2, y2);

    if(p_q_r > closest_R_dist + R_radius[k]) {
      if(row == 326622 && k == 934)
        printf("p_q_r=%f, closest_R_dist=%f, R_radius=%f\n", p_q_r, closest_R_dist, R_radius[k]);
      _zero_bit(smem, k);
    }
  }

  __syncthreads();

  if(row == 326622)
    printf("bit @ 934 = %d", _get_val(smem, 934));


  /**
   * Output bitset
   */
  for(int l = tid; l < bitset_size; l+=blockDim.x)
    output[row * bitset_size + l] = smem[l];
}

/**
 * Kernel for more narrow data sizes (n_cols <= 32)
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
template <typename value_idx = int64_t, typename value_t,
          typename value_int = int, int warp_q = 1024, int thread_q = 1,
          int tpb = 32>
__global__ void rbc_kernel(const value_t *X, value_int n_cols,
                           const value_idx *R_knn_inds,
                           const value_t *R_knn_dists, value_int m, int k,
                           const value_idx *R_indptr,
                           const value_idx *R_1nn_cols,
                           const value_t *R_1nn_dists, value_idx *out_inds,
                           value_t *out_dists, value_idx *sampled_inds_map,
                           int *dist_counter, value_t *R_radius, value_int debug_row = -1) {
  int row = blockIdx.x / k;
  int cur_k = blockIdx.x % k;

  const value_t *x_ptr = X + (n_cols * row);
  value_t x1 = x_ptr[0];
  value_t x2 = x_ptr[1];

  // Each warp works on 1 R
  faiss::gpu::KeyValueWarpSelect<value_t, value_idx, false,
                                 faiss::gpu::Comparator<value_t>, warp_q,
                                 thread_q, tpb>
    heap(faiss::gpu::Limits<value_t>::getMax(),
         faiss::gpu::KeyValuePair<value_t, value_idx>(
           faiss::gpu::Limits<value_t>::getMax(), -1),
         k);

  value_t min_R_dist = R_knn_dists[row * k];

  /**
   * First add distances for k closest neighbors of R
   * to the heap
   */
  // Start iterating through elements of each set from closest R elements,
  // determining if the distance could even potentially be in the heap.

  // index and distance to current row's closest landmark
  value_t cur_R_dist = R_knn_dists[row * k + cur_k];
  value_idx cur_R_ind = R_knn_inds[row * k + cur_k];

  // Equation (2) in Cayton's paper- prune out R's which are > 3 * p(q, r_q)
  if(cur_R_dist > min_R_dist + R_radius[cur_R_ind]) return;
//  if (cur_R_dist > 3 * min_R_dist) return;

  // The whole warp should iterate through the elements in the current R
  value_idx R_start_offset = R_indptr[cur_R_ind];
  value_idx R_stop_offset = R_indptr[cur_R_ind + 1];

  value_idx R_size = R_stop_offset - R_start_offset;

  int limit = faiss::gpu::utils::roundDown(R_size, faiss::gpu::kWarpSize);

  int n_dists_computed = 0;
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

    if(row == 326622 && (cur_candidate_ind == 326622 || cur_candidate_ind == 326720 || cur_candidate_ind == 326623))
      printf("cur_candidate_ind=%ld, z=%f, heap.warKTop=%f, heap.warpKTopRDist=%f, cur_candidate_dist=%f\n",
             cur_candidate_ind, z, heap.warpKTop, heap.warpKTopRDist, cur_candidate_dist);

    value_t dist = std::numeric_limits<value_t>::max();
    if (i < k || z <= heap.warpKTop) {
      const value_t *y_ptr = X + (n_cols * cur_candidate_ind);
      value_t y1 = y_ptr[0];
      value_t y2 = y_ptr[1];

      dist = compute_haversine(x1, y1, x2, y2);
      n_dists_computed++;
    }
    if(row == 326622 && (cur_candidate_ind == 326622 || cur_candidate_ind == 326720 || cur_candidate_ind == 326623))
      printf("dist=%f\n", dist);
    heap.add(
      dist, faiss::gpu::KeyValuePair(cur_candidate_dist, cur_candidate_ind));
  }

  __syncthreads();

  if (i < R_size) {
    value_idx cur_candidate_ind = R_1nn_cols[R_start_offset + i];
    value_t cur_candidate_dist = R_1nn_dists[R_start_offset + i];
    value_t z = heap.warpKTopRDist == 0.0
                  ? 0.0
                  : (abs(heap.warpKTop - heap.warpKTopRDist) *
                       abs(heap.warpKTopRDist - cur_candidate_dist) -
                     heap.warpKTop * cur_candidate_dist) /
                      heap.warpKTopRDist;

    z = isnan(z) ? 0.0 : z;

    if(row == 326622 && (cur_candidate_ind == 326622 || cur_candidate_ind == 326720 || cur_candidate_ind == 326623))
      printf("cur_candidate_ind=%ld, z=%f, heap.warKTop=%f, heap.warpKTopRDist=%f, cur_candidate_dist=%f\n",
             cur_candidate_ind, z, heap.warpKTop, heap.warpKTopRDist, cur_candidate_dist);

    value_t dist = std::numeric_limits<value_t>::max();
    if (i < k || z <= heap.warpKTop) {
      const value_t *y_ptr = X + (n_cols * cur_candidate_ind);
      value_t y1 = y_ptr[0];
      value_t y2 = y_ptr[1];

      dist = compute_haversine(x1, y1, x2, y2);

      n_dists_computed++;
    }
    if(row == 326622 && (cur_candidate_ind == 326622 || cur_candidate_ind == 326720 || cur_candidate_ind == 326623))
      printf("dist=%f\n", dist);
    heap.addThreadQ(
      dist, faiss::gpu::KeyValuePair(cur_candidate_dist, cur_candidate_ind));
  }

  heap.reduce();

  value_idx cur_idx = (row * k * k) + (cur_k * k);
  heap.writeOut(out_dists + cur_idx, out_inds + cur_idx, k);

  atomicAdd(dist_counter + row, n_dists_computed);
}

//template<typename value_idx, typename value_t>
//void compute_landmark_radius()

/**
 * Random ball cover algorithm uses the triangle inequality
 * (which can be used for any valid metric or distance
 * that follows the triangle inequality)
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx = int64_t, typename value_t,
          typename value_int = int>
void random_ball_cover(const raft::handle_t &handle, const value_t *X,
                       value_int m, value_int n, int k, value_idx *inds,
                       value_t *dists, value_int n_samples = -1) {
  auto exec_policy = rmm::exec_policy(handle.get_stream());
  /**
   * 1. Randomly sample sqrt(n) points from X
   */
  rmm::device_uvector<value_idx> R_knn_inds(k * m, handle.get_stream());
  rmm::device_uvector<value_t> R_knn_dists(k * m, handle.get_stream());

  rmm::device_uvector<value_idx> out_inds_full(k * k * m, handle.get_stream());
  rmm::device_uvector<value_t> out_dists_full(k * k * m, handle.get_stream());

  thrust::fill(exec_policy, out_dists_full.data(),
               out_dists_full.data() + out_dists_full.size(),
               std::numeric_limits<value_t>::max());
  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  /**
   * the sqrt() here makes the sqrt(m)^2 a linear-time lower bound
   */
  n_samples = n_samples < 1 ? int(sqrt(m)) : n_samples;

  ASSERT(n_samples >= k, "number of landmark samples must be >= k");
  rmm::device_uvector<value_idx> R_indices(n_samples, handle.get_stream());
  rmm::device_uvector<value_t> R(n_samples * n, handle.get_stream());

  rmm::device_uvector<value_idx> R_1nn_cols(m, handle.get_stream());
  rmm::device_uvector<value_t> R_1nn_ones(m, handle.get_stream());

  thrust::fill(exec_policy, R_1nn_ones.data(),
               R_1nn_ones.data() + R_1nn_ones.size(), 1.0);

  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  rmm::device_uvector<value_idx> R_1nn_cols2(n_samples, handle.get_stream());

  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  thrust::sequence(thrust::cuda::par.on(handle.get_stream()), R_1nn_cols.data(),
                   R_1nn_cols.data() + m, (value_idx)0);

  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  auto rng = raft::random::Rng(12345);
  rng.sampleWithoutReplacement(
    handle, R_indices.data(), R_1nn_cols2.data(), R_1nn_cols.data(),
    R_1nn_ones.data(), (value_idx)n_samples, (value_idx)m, handle.get_stream());

  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  raft::matrix::copyRows<value_t, value_idx, size_t>(
    X, m, n, R.data(), R_1nn_cols2.data(), n_samples, handle.get_stream(),
    true);

  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));


  raft::print_device_vector("R sampled indices", R_indices.data(),
                            R_indices.size(), std::cout);

  /**
   * 2. Perform knn = bfknn(X, R, k)
   */
  std::vector<value_t *> input = {R.data()};
  std::vector<int> sizes = {n_samples};

  brute_force_knn_impl<int, int64_t>(
    input, sizes, n, const_cast<value_t *>(X), m, R_knn_inds.data(),
    R_knn_dists.data(), k, handle.get_device_allocator(), handle.get_stream(),
    nullptr, 0, (bool)true, (bool)true, nullptr,
    raft::distance::DistanceType::Haversine);

  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  /**
   * 3. Create L_r = knn[:,0].T (CSR)
   *
   * Slice closest neighboring R
   * Secondary sort by (R_knn_inds, R_knn_dists)
   */

  rmm::device_uvector<value_idx> R_1nn_inds(m, handle.get_stream());
  rmm::device_uvector<value_t> R_1nn_dists(m, handle.get_stream());

  value_idx *R_1nn_inds_ptr = R_1nn_inds.data();
  value_t *R_1nn_dists_ptr = R_1nn_dists.data();
  value_idx *R_knn_inds_ptr = R_knn_inds.data();
  value_t *R_knn_dists_ptr = R_knn_dists.data();

  auto idxs = thrust::make_counting_iterator<value_idx>(0);
  thrust::for_each(exec_policy, idxs, idxs + m, [=] __device__(value_idx i) {
    R_1nn_inds_ptr[i] = R_knn_inds_ptr[i * k];
    R_1nn_dists_ptr[i] = R_knn_dists_ptr[i * k];
  });

  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));


  auto keys = thrust::make_zip_iterator(
    thrust::make_tuple(R_1nn_inds.data(), R_1nn_dists.data()));
  auto vals = thrust::make_zip_iterator(thrust::make_tuple(R_1nn_cols.data()));

  // group neighborhoods for each reference landmark and sort each group by distance
  thrust::sort_by_key(thrust::cuda::par.on(handle.get_stream()), keys,
                      keys + R_1nn_inds.size(), vals, NNComp());

  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  // convert to CSR for fast lookup
  rmm::device_uvector<value_idx> R_indptr(n_samples + 1, handle.get_stream());
  raft::sparse::convert::sorted_coo_to_csr(
    R_1nn_inds.data(), m, R_indptr.data(), n_samples + 1,
    handle.get_device_allocator(), handle.get_stream());

  raft::print_device_vector("R_indptr", R_indptr.data(), R_indptr.size(),
                            std::cout);
  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  /**
   * Compute radius of each R for filtering: p(q, r) <= p(q, q_r) + radius(r)
   * (need to take the
   */
  rmm::device_uvector<value_t> R_radius(n_samples, handle.get_stream());

  value_idx *R_indptr_ptr = R_indptr.data();
  value_t *R_radius_ptr = R_radius.data();
  auto entries = thrust::make_counting_iterator<value_idx>(0);

  thrust::for_each(exec_policy, entries, entries+n_samples,
    [=] __device__ (value_idx input) {
     value_idx last_row_idx = R_indptr_ptr[input+1]-1;
     R_radius_ptr[input] = R_1nn_dists_ptr[last_row_idx];
    });

  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));


  /**
   * 4. Perform k-select over original KNN, using L_r to filter distances
   *
   * a. Map 1 row to each warp/block
   * b. Add closest k R points to heap
   * c. Iterate through batches of R, having each thread in the warp load a set
   * of distances y from R (only if d(q, r) < 3 * distance to closest r) and
   * marking the distance to be computed between x, y only
   * if knn[k].distance >= d(x_i, R_k) + d(R_k, y)
   */

  rmm::device_uvector<int> dists_counter(m, handle.get_stream());

  // Compute nearest k for each neighborhood in each closest R
  rbc_kernel<<<m * k, 32, 0, handle.get_stream()>>>(
    X, n, R_knn_inds.data(), R_knn_dists.data(), m, k, R_indptr.data(),
    R_1nn_cols.data(), R_1nn_dists.data(), out_inds_full.data(),
    out_dists_full.data(), R_indices.data(), dists_counter.data(), R_radius.data(), 0);

  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  raft::print_device_vector("dists_counter", dists_counter.data(), 15,
                            std::cout);
//
  raft::print_device_vector("closest R 2736511",
                            R_knn_inds.data() + (326622 * k), k,
                            std::cout);
  raft::print_device_vector("closest R dists 2736511",
                            R_knn_dists.data() + (326622 * k), k,
                            std::cout);

  raft::print_device_vector("closest R 2736503",
                            R_knn_inds.data() + (326720 * k), k,
                            std::cout);
  raft::print_device_vector("closest R dists 2736503",
                            R_knn_dists.data() + (326720 * k), k,
                            std::cout);

  raft::print_device_vector("closest R 326623",
                            R_knn_inds.data() + (326623 * k), k,
                            std::cout);
  raft::print_device_vector("closest R dists 326623",
                            R_knn_dists.data() + (326623 * k), k,
                            std::cout);


//  raft::print_device_vector(
//    "2547454 R knn: ", R_knn_inds.data() + (2547454 * k), k, std::cout);
//  raft::print_device_vector(
//    "2547454 R knn dists: ", R_knn_dists.data() + (2547454 * k), k, std::cout);
//  raft::print_device_vector(
//    "2547455 R knn: ", R_knn_inds.data() + (2547455 * k), k, std::cout);
//  raft::print_device_vector(
//    "2547455 R knn dists: ", R_knn_dists.data() + (2547455 * k), k, std::cout);

  // Reduce k * k to final k
  select_k(out_dists_full.data(), out_inds_full.data(), m, k * k, dists, inds,
           true, k, handle.get_stream());

//  // perform k-select of remaining landmarks
  int bitset_size = ceil(n_samples / 32.0);

  printf("bitset_size=%d\n", bitset_size);

  rmm::device_uvector<uint32_t> bitset(bitset_size * m, handle.get_stream());
  perform_post_filter<<<m, 32, bitset_size * sizeof(uint32_t), handle.get_stream()>>>(
    X, n, R_knn_inds.data(), R_knn_dists.data(), R_radius.data(),
    R.data(), n_samples, bitset_size, k, bitset.data());

  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  raft::print_device_vector("bitset", bitset.data(), 500, std::cout);

  rmm::device_uvector<int> post_dists_counter(m, handle.get_stream());

  // Compute any distances from the landmarks that remain in the bitset
  compute_final_dists<<<m, 32, 0, handle.get_stream()>>>(
                      X, n, bitset.data(), bitset_size,
                      R_knn_dists.data(),  R_indptr.data(), R_1nn_inds.data(),
                      R_1nn_dists.data(), inds,
                      dists, n_samples, k, post_dists_counter.data());

  raft::print_device_vector("dists_counter 2nd phase", post_dists_counter.data(), 15,
                            std::cout);

  int additional_dists = thrust::reduce(exec_policy, post_dists_counter.data(), post_dists_counter.data()+m, 0, thrust::plus<int>());
  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  printf("total post_dists: %d\n", additional_dists);

  //   Thoughts:
  //   For n_cols < 32, we could probably just have each thread compute the distance
  //   For n_cols >= 32, we could probably have full warps compute the distances
}

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft