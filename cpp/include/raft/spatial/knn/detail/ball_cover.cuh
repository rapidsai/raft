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

#include "ball_cover/common.cuh"
#include "ball_cover/registers.cuh"
#include "ball_cover/shared_mem.cuh"
#include "block_select_faiss.cuh"
#include "haversine_distance.cuh"
#include "knn_brute_force_faiss.cuh"
#include "selection_faiss.cuh"

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

#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

/**
 * Random ball cover algorithm uses the triangle inequality
 * (which can be used for any valid metric or distance
 * that follows the triangle inequality)
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx = int64_t, typename value_t,
          typename value_int = int, typename distance_func>
void random_ball_cover_all_neigh_knn(const raft::handle_t &handle,
                                     const value_t *X, value_int m, value_int n,
                                     int k, value_idx *inds, value_t *dists,
                                     distance_func dfunc,
                                     value_int n_samples = -1,
                                     bool perform_post_filtering = true,
                                     float weight = 1.0) {
  auto exec_policy = rmm::exec_policy(handle.get_stream());

  /**
   * 1. Randomly sample sqrt(n) points from X
   */
  rmm::device_uvector<value_idx> R_knn_inds(k * m, handle.get_stream());
  rmm::device_uvector<value_t> R_knn_dists(k * m, handle.get_stream());

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

  rmm::device_uvector<value_idx> R_1nn_cols2(n_samples, handle.get_stream());

  thrust::sequence(thrust::cuda::par.on(handle.get_stream()), R_1nn_cols.data(),
                   R_1nn_cols.data() + m, (value_idx)0);

  auto rng = raft::random::Rng(12345);
  rng.sampleWithoutReplacement(
    handle, R_indices.data(), R_1nn_cols2.data(), R_1nn_cols.data(),
    R_1nn_ones.data(), (value_idx)n_samples, (value_idx)m, handle.get_stream());

  raft::matrix::copyRows<value_t, value_idx, size_t>(
    X, m, n, R.data(), R_1nn_cols2.data(), n_samples, handle.get_stream(),
    true);

  /**
   * 2. Perform knn = bfknn(X, R, k)
   */
  std::vector<value_t *> input = {R.data()};
  std::vector<int> sizes = {n_samples};

  brute_force_knn_impl<int, int64_t>(
    input, sizes, n, const_cast<value_t *>(X), m, R_knn_inds.data(),
    R_knn_dists.data(), k, handle.get_device_allocator(), handle.get_stream(),
    nullptr, 0, (bool)true, (bool)true, nullptr,
    raft::distance::DistanceType::L2Expanded);

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

  auto keys = thrust::make_zip_iterator(
    thrust::make_tuple(R_1nn_inds.data(), R_1nn_dists.data()));
  auto vals = thrust::make_zip_iterator(thrust::make_tuple(R_1nn_cols.data()));

  // group neighborhoods for each reference landmark and sort each group by distance
  thrust::sort_by_key(thrust::cuda::par.on(handle.get_stream()), keys,
                      keys + R_1nn_inds.size(), vals, NNComp());

  // convert to CSR for fast lookup
  rmm::device_uvector<value_idx> R_indptr(n_samples + 1, handle.get_stream());
  raft::sparse::convert::sorted_coo_to_csr(
    R_1nn_inds.data(), m, R_indptr.data(), n_samples + 1,
    handle.get_device_allocator(), handle.get_stream());

  /**
   * Compute radius of each R for filtering: p(q, r) <= p(q, q_r) + radius(r)
   * (need to take the
   */
  rmm::device_uvector<value_t> R_radius(n_samples, handle.get_stream());

  value_idx *R_indptr_ptr = R_indptr.data();
  value_t *R_radius_ptr = R_radius.data();
  auto entries = thrust::make_counting_iterator<value_idx>(0);

  thrust::for_each(exec_policy, entries, entries + n_samples,
                   [=] __device__(value_idx input) {
                     value_idx last_row_idx = R_indptr_ptr[input + 1] - 1;
                     R_radius_ptr[input] = R_1nn_dists_ptr[last_row_idx];
                   });

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

  constexpr int max_vals = 328;
  if (n <= 3) {
    // Compute nearest k for each neighborhood in each closest R
    block_rbc_kernel_registers<value_idx, value_t, 32, 2, 128, 2>
      <<<m, 128, 0, handle.get_stream()>>>(
        X, n, R_knn_inds.data(), R_knn_dists.data(), m, k, R_indptr.data(),
        R_1nn_cols.data(), R_1nn_dists.data(), inds, dists, R_indices.data(),
        dists_counter.data(), R_radius.data(), dfunc, weight);
  } else if (n <= max_vals) {
//    CUDA_CHECK(cudaFuncSetCacheConfig(
//      block_rbc_kernel_smem<value_idx, value_t, 32, 2, 128, max_vals>,
//      cudaFuncCachePreferShared));

    // Compute nearest k for each neighborhood in each closest R
    block_rbc_kernel_smem<value_idx, value_t, 32, 2, 128, max_vals>
      <<<m, 128, 0, handle.get_stream()>>>(
        X, n, R_knn_inds.data(), R_knn_dists.data(), m, k, R_indptr.data(),
        R_1nn_cols.data(), R_1nn_dists.data(), inds, dists, R_indices.data(),
        dists_counter.data(), R_radius.data(), dfunc, weight);
  }
  //

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
  //  raft::print_device_vector("dists", dists_counter.data(), 500, std::cout);
  //
  if (perform_post_filtering) {
    int bitset_size = ceil(n_samples / 32.0);

    rmm::device_uvector<uint32_t> bitset(bitset_size * m, handle.get_stream());

    printf("performing post filter\n");

    perform_post_filter<value_idx, value_t, int, 128>
      <<<m, 128, bitset_size * sizeof(uint32_t), handle.get_stream()>>>(
        X, n, R_knn_inds.data(), R_knn_dists.data(), R_radius.data(), R.data(),
        n_samples, bitset_size, k, bitset.data(), weight);

        CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
    //
    printf("computing final dists\n");

    rmm::device_uvector<int> post_dists_counter(m, handle.get_stream());
    thrust::fill(exec_policy, post_dists_counter.data(),
                 post_dists_counter.data() + m, 0);

    if (n <= 3) {
      // Compute any distances from the landmarks that remain in the bitset
      compute_final_dists_registers<<<m, 128, 0, handle.get_stream()>>>(
        X, n, bitset.data(), bitset_size, R_knn_dists.data(), R_indptr.data(),
        R_1nn_cols.data(), R_1nn_dists.data(), inds, dists, n_samples, k, dfunc,
        post_dists_counter.data());
    } else if (n <= max_vals) {

//      CUDA_CHECK(cudaFuncSetCacheConfig(
//        compute_final_dists_smem<value_idx, value_t, 32, 2, 128, max_vals>,
//        cudaFuncCachePreferShared));

      // Compute any distances from the landmarks that remain in the bitset
      compute_final_dists_smem<value_idx, value_t, 32, 2, 128, max_vals>
        <<<m, 128, 0, handle.get_stream()>>>(
          X, n, bitset.data(), bitset_size, R_knn_dists.data(), R_indptr.data(),
          R_1nn_cols.data(), R_1nn_dists.data(), inds, dists, n_samples, k,
          dfunc, post_dists_counter.data());
    }
    //
        CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
    //    raft::print_device_vector("dists", post_dists_counter.data(), 500, std::cout);

    printf("Done.\n");
    //
    //    printf("total post_dists: %d\n", additional_dists);
  }
}

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft