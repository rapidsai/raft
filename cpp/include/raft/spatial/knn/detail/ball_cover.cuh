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
template <typename value_idx, typename value_t, int warp_q = 1024,
          int thread_q = 1, int tpb = 32>
__global__ void rbc_kernel(const value_t *X, const value_idx n_cols,
                           const value_idx *R_knn_inds,
                           const value_t *R_knn_dists, value_idx m, int k,
                           const value_idx *R_indptr,
                           const value_idx *R_1nn_cols,
                           const value_t *R_1nn_dists, value_idx *out_inds,
                           value_t *out_dists, value_idx *sampled_inds_map) {
  int row = blockIdx.x / k;
  int cur_k = blockIdx.x % k;

  const value_t *x_ptr = X + (n_cols * row);
  value_t x1 = x_ptr[0];
  value_t x2 = x_ptr[1];

  // Each warp works on 1 R
  faiss::gpu::WarpSelect<value_t, value_idx, false,
                         faiss::gpu::Comparator<value_t>, warp_q, thread_q, tpb>
    heap(faiss::gpu::Limits<value_t>::getMax(), -1, k);

  // Grid is exactly sized to rows available
  value_t min_R_dist = R_knn_dists[row * k];
  value_idx min_R_ind = R_knn_inds[row * k];

  /**
   * First add distances for k closest neighbors of R
   * to the heap
   */
  // Start iterating through elements of each set from closest R elements,
  // determining if the distance could even potentially be in the heap.

  // just doing Rs for the closest k for now
  value_t cur_R_dist = R_knn_dists[row * k + cur_k];
  value_idx cur_R_ind = R_knn_inds[row * k + cur_k];

  // The whole warp should iterate through the elements in the current R
  value_idx R_start_offset = R_indptr[cur_R_ind];
  value_idx R_stop_offset = R_indptr[cur_R_ind + 1];

  value_idx R_size = R_stop_offset - R_start_offset;

  int limit = faiss::gpu::utils::roundDown(R_size, faiss::gpu::kWarpSize);

  int i = threadIdx.x;
  for (; i < limit; i += tpb) {
    value_idx cur_candidate_ind = R_1nn_cols[R_start_offset + i];
    value_idx cur_candidate_dist = R_1nn_dists[R_start_offset + i];

//    printf("row=%d, warpKTopDist=%f, numVals=%d\n", row, heap.warpKTop,
//           heap.numVals);
//
    if (i < k || heap.warpKTop >= cur_candidate_dist + cur_R_dist) {
      const value_t *y_ptr = X + (n_cols * cur_candidate_ind);
      value_t y1 = y_ptr[0];
      value_t y2 = y_ptr[1];

      value_t dist = compute_haversine(x1, y1, x2, y2);

//      printf(
//        "row=%d, R=%d, R_stop_offset=%d, R_start_offset=%d, candidate_ind=%d, "
//        "dist=%f\n",
//        row, cur_R_ind, R_stop_offset, R_start_offset, cur_candidate_ind, dist);
//
      heap.addThreadQ(dist, cur_candidate_ind);
    }

    heap.checkThreadQ();
  }

  if (i < R_size) {
    value_idx cur_candidate_ind = R_1nn_cols[R_start_offset + i];
    value_idx cur_candidate_dist = R_1nn_dists[R_start_offset + i];
//
//    printf("row=%d, warpKTopDist=%f, numVals=%d\n", row, heap.warpKTop,
//           heap.numVals);

    if (i < k || heap.warpKTop >= cur_candidate_dist + cur_R_dist) {
      const value_t *y_ptr = X + (n_cols * cur_candidate_ind);
      value_t y1 = y_ptr[0];
      value_t y2 = y_ptr[1];

      value_t dist = compute_haversine(x1, y1, x2, y2);
//
//      printf(
//        "row=%d, R=%d, R_stop_offset=%d, R_start_offset=%d, candidate_ind=%d, "
//        "dist=%f\n",
//        row, cur_R_ind, R_stop_offset, R_start_offset, cur_candidate_ind, dist);

      heap.addThreadQ(dist, cur_candidate_ind);
    }
  }

  heap.reduce();

  value_idx cur_idx = (row * k * k) + (cur_k * k);
  heap.writeOut(out_dists + cur_idx, out_inds + cur_idx, k);
}

/**
 * Random ball cover algorithm uses the triangle inequality
 * (which can be used for any valid metric or distance
 * that follows the triangle inequality)
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx = int, typename value_t>
void random_ball_cover(const raft::handle_t &handle, const value_t *X,
                       value_idx m, value_idx n, int k, value_idx *inds,
                       value_t *dists, value_idx n_samples = -1) {
  auto exec_policy = rmm::exec_policy(handle.get_stream());
  /**
   * 1. Randomly sample sqrt(n) points from X
   */

//  printf("k=%d\n", k);
  rmm::device_uvector<int64_t> R_knn_inds_64b(k * m, handle.get_stream());

  rmm::device_uvector<value_idx> R_knn_inds(k * m, handle.get_stream());
  rmm::device_uvector<value_t> R_knn_dists(k * m, handle.get_stream());

  rmm::device_uvector<value_idx> out_inds_full(k * k * m, handle.get_stream());
  rmm::device_uvector<value_t> out_dists_full(k * k * m, handle.get_stream());


  n_samples = n_samples < 1 ? int(sqrt(m)) : n_samples;

  ASSERT(n_samples >= k, "number of landmark samples must be >= k");

//  printf("Sampling %d points\n", n_samples);

  rmm::device_uvector<value_idx> R_indices(n_samples, handle.get_stream());
  rmm::device_uvector<value_t> R(n_samples * n, handle.get_stream());

  rmm::device_uvector<value_idx> R_1nn_cols(m, handle.get_stream());
  rmm::device_uvector<value_t> R_1nn_ones(m, handle.get_stream());

  thrust::fill(exec_policy, R_1nn_ones.data(),
               R_1nn_ones.data() + R_1nn_ones.size(), 1.0);

  rmm::device_uvector<value_idx> R_1nn_cols2(m, handle.get_stream());

  thrust::sequence(thrust::cuda::par.on(handle.get_stream()), R_1nn_cols.data(),
                   R_1nn_cols.data() + m, 0);

  auto rng = raft::random::Rng(12345);
  rng.sampleWithoutReplacement(handle, R_indices.data(), R_1nn_cols2.data(),
                               R_1nn_cols.data(), R_1nn_ones.data(), n_samples,
                               m, handle.get_stream());

  raft::matrix::copyRows(X, m, n, R.data(), R_indices.data(), n_samples,
                         handle.get_stream(), true);
//
//  raft::print_device_vector("sampled indices", R_indices.data(),
//                            R_indices.size(), std::cout);
//  raft::print_device_vector("sampled rows", R.data(), R.size(), std::cout);

  /**
   * 2. Perform knn = bfknn(X, R, k)
   */
  printf("Performing bfknn of landmarks\n");
  std::vector<value_t *> input = {R.data()};
  std::vector<value_idx> sizes = {n_samples};

  brute_force_knn_impl<int, int64_t>(
    input, sizes, n, const_cast<value_t *>(X), m, R_knn_inds_64b.data(),
    R_knn_dists.data(), k, handle.get_device_allocator(), handle.get_stream(),
    nullptr, 0, (bool)true, (bool)true, nullptr,
    raft::distance::DistanceType::Haversine);

  thrust::transform(exec_policy, R_knn_inds_64b.data(),
                    R_knn_inds_64b.data() + R_knn_inds_64b.size(),
                    R_knn_inds.data(), [] __device__(int64_t i) { return i; });

//  raft::print_device_vector("R_knn_inds", R_knn_inds.data(), R_knn_inds.size(),
//                            std::cout);
//  raft::print_device_vector("R_knn_dists", R_knn_dists.data(),
//                            R_knn_dists.size(), std::cout);

  /**
   * 3. Create L_r = knn[:,0].T (CSR)
   *
   * Slice closest neighboring R
   * Secondary sort by (R_knn_inds, R_knn_dists)
   */

//  printf("Building representative lists for landmarks\n");
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

  auto keys = thrust::make_zip_iterator(
    thrust::make_tuple(R_1nn_inds.data(), R_1nn_dists.data()));
  auto vals = thrust::make_zip_iterator(thrust::make_tuple(R_1nn_cols.data()));

//  raft::print_device_vector("R_1nn_inds", R_1nn_inds.data(), R_1nn_inds.size(),
//                            std::cout);
//  raft::print_device_vector("R_1nn_cols", R_1nn_cols.data(), R_1nn_cols.size(),
//                            std::cout);
//
//  raft::print_device_vector("R_1nn_dists", R_1nn_dists.data(),
//                            R_1nn_dists.size(), std::cout);

  // group neighborhoods for each reference landmark and sort each group by distance
  thrust::sort_by_key(thrust::cuda::par.on(handle.get_stream()), keys,
                      keys + R_1nn_inds.size(), vals, NNComp());

//  raft::print_device_vector("R_1nn_inds_sorted", R_1nn_inds.data(),
//                            R_1nn_inds.size(), std::cout);
//  raft::print_device_vector("R_1nn_cols_sorted", R_1nn_cols.data(),
//                            R_1nn_cols.size(), std::cout);
//
//  raft::print_device_vector("R_1nn_dists_sorted", R_1nn_dists.data(),
//                            R_1nn_dists.size(), std::cout);

  // convert to CSR for fast lookup
  rmm::device_uvector<value_idx> R_indptr(n_samples, handle.get_stream());
  raft::sparse::convert::sorted_coo_to_csr(
    R_1nn_inds.data(), m, R_indptr.data(), n_samples + 1,
    handle.get_device_allocator(), handle.get_stream());
//
//  raft::print_device_vector("R_1nn_indptr", R_indptr.data(), R_indptr.size(),
//                            std::cout);

  /**
   * 4. Perform k-select over original KNN, using L_r to filter distances
   *
   * a. Map 1 row to each warp/block
   * b. Add closest k R points to heap
   * c. Iterate through batches of R, having each thread in the warp load a set
   * of distances y from R (only if d(q, r) < 3 * distance to closest r) and marking the distance to be computed between x, y only
   * if knn[k].distance >= d(x_i, R_k) + d(R_k, y)
   */

  printf("Performing final bfknn\n");

  /**
   * Compute nearest k for each nearest landmark R and reduce them to a final
   * k
   */
  rbc_kernel<<<m * k, 32, 0, handle.get_stream()>>>(
    X, n, R_knn_inds.data(), R_knn_dists.data(), m, k, R_indptr.data(),
    R_1nn_cols.data(), R_1nn_dists.data(), out_inds_full.data(), out_dists_full.data(), R_indices.data());
//
//  raft::print_device_vector("out_inds", out_inds_full.data(), out_inds_full.size(), std::cout);
//  raft::print_device_vector("out_dists", out_dists_full.data(), out_dists_full.size(), std::cout);

  //   Thoughts:
  //   For n_cols < 32, we could probably just have each thread compute the distance
  //   For n_cols >= 32, we could probably have full warps compute the distances
}

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft