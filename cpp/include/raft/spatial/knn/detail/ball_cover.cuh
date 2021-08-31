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

#include "../ball_cover_common.h"
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
#include <raft/sparse/distance/operators.cuh>

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
 * Given a set of points in row-major order which are to be
 * used as a set of index points, uniformly samples a subset
 * of points to be used as landmarks.
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param index
 */
template <typename value_idx, typename value_t>
void sample_landmarks(const raft::handle_t &handle,
                      BallCoverIndex<value_idx, value_t> &index) {
  auto exec_policy = rmm::exec_policy(handle.get_stream());

  rmm::device_uvector<value_idx> R_1nn_cols2(index.n_landmarks,
                                             handle.get_stream());
  rmm::device_uvector<value_t> R_1nn_ones(index.m, handle.get_stream());

  rmm::device_uvector<value_idx> R_indices(index.n_landmarks,
                                           handle.get_stream());

  thrust::sequence(thrust::cuda::par.on(handle.get_stream()),
                   index.get_R_1nn_cols(), index.get_R_1nn_cols() + index.m,
                   (value_idx)0);

  thrust::fill(exec_policy, R_1nn_ones.data(),
               R_1nn_ones.data() + R_1nn_ones.size(), 1.0);

  /**
 * 1. Randomly sample sqrt(n) points from X
 */
  auto rng = raft::random::Rng(12345);
  rng.sampleWithoutReplacement(handle, R_indices.data(), R_1nn_cols2.data(),
                               index.get_R_1nn_cols(), R_1nn_ones.data(),
                               (value_idx)index.n_landmarks, (value_idx)index.m,
                               handle.get_stream());

  raft::matrix::copyRows<value_t, value_idx, size_t>(
    index.get_X(), index.m, index.n, index.get_R(), R_1nn_cols2.data(),
    index.n_landmarks, handle.get_stream(), true);
}

/**
 * Constructs a 1-nn index mapping each landmark to their closest points.
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param R_knn_inds_ptr
 * @param R_knn_dists_ptr
 * @param k
 * @param index
 */
template <typename value_idx, typename value_t>
void construct_landmark_1nn(const raft::handle_t &handle,
                            const value_idx *R_knn_inds_ptr,
                            const value_t *R_knn_dists_ptr, int k,
                            BallCoverIndex<value_idx, value_t> &index) {
  auto exec_policy = rmm::exec_policy(handle.get_stream());

  rmm::device_uvector<value_idx> R_1nn_inds(index.m, handle.get_stream());

  value_idx *R_1nn_inds_ptr = R_1nn_inds.data();
  value_t *R_1nn_dists_ptr = index.get_R_1nn_dists();

  auto idxs = thrust::make_counting_iterator<value_idx>(0);
  thrust::for_each(exec_policy, idxs, idxs + index.m,
                   [=] __device__(value_idx i) {
                     R_1nn_inds_ptr[i] = R_knn_inds_ptr[i * k];
                     R_1nn_dists_ptr[i] = R_knn_dists_ptr[i * k];
                   });

  auto keys = thrust::make_zip_iterator(
    thrust::make_tuple(R_1nn_inds.data(), index.get_R_1nn_dists()));
  auto vals =
    thrust::make_zip_iterator(thrust::make_tuple(index.get_R_1nn_cols()));

  // group neighborhoods for each reference landmark and sort each group by distance
  thrust::sort_by_key(thrust::cuda::par.on(handle.get_stream()), keys,
                      keys + index.m, vals, NNComp());

  // convert to CSR for fast lookup
  raft::sparse::convert::sorted_coo_to_csr(
    R_1nn_inds.data(), index.m, index.get_R_indptr(), index.n_landmarks + 1,
    handle.get_stream());
}

/**
 * Computes the k closest landmarks to a set of query points.
 * @tparam value_idx
 * @tparam value_t
 * @tparam value_int
 * @param handle
 * @param index
 * @param query_pts
 * @param n_query_pts
 * @param k
 * @param R_knn_inds
 * @param R_knn_dists
 */
template <typename value_idx, typename value_t, typename value_int = int>
void k_closest_landmarks(const raft::handle_t &handle,
                         BallCoverIndex<value_idx, value_t> &index,
                         const value_t *query_pts, value_int n_query_pts, int k,
                         value_idx *R_knn_inds, value_t *R_knn_dists) {
  std::vector<value_t *> input = {index.get_R()};
  std::vector<int> sizes = {index.n_landmarks};

  brute_force_knn_impl<int, int64_t>(
    input, sizes, index.n, const_cast<value_t *>(query_pts), n_query_pts,
    R_knn_inds, R_knn_dists, k, handle.get_stream(), nullptr, 0, (bool)true,
    (bool)true, nullptr, index.metric);
}

/**
 * Uses the sorted data points in the 1-nn landmark index to compute
 * an array of radii for each landmark.
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param index
 */
template <typename value_idx, typename value_t>
void compute_landmark_radii(const raft::handle_t &handle,
                            BallCoverIndex<value_idx, value_t> &index) {
  auto exec_policy = rmm::exec_policy(handle.get_stream());

  auto entries = thrust::make_counting_iterator<value_idx>(0);

  const value_idx *R_indptr_ptr = index.get_R_indptr();
  const value_t *R_1nn_dists_ptr = index.get_R_1nn_dists();
  value_t *R_radius_ptr = index.get_R_radius();
  thrust::for_each(exec_policy, entries, entries + index.n_landmarks,
                   [=] __device__(value_idx input) {
                     value_idx last_row_idx = R_indptr_ptr[input + 1] - 1;
                     R_radius_ptr[input] = R_1nn_dists_ptr[last_row_idx];
                   });
}

/**
 * Similar to a ball tree, the random ball cover algorithm
 * uses the triangle inequality to prune distance computations
 * in any metric space with a guarantee of sqrt(n) * c^{3/2}
 * where `c` is an expansion constant based on the distance
 * metric.
 *
 * This function variant performs an all nearest neighbors
 * query which is useful for algorithms that need to perform
 * A * A.T.
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam value_int
 * @tparam distance_func TODO: Remove
 * @param[in] handle raft handle for resource management
 * @param[inout] index previously untrained index
 * @param[in] k number neighbors to return
 * @param[out] inds output indices
 * @param[out] dists output distances
 * @param[in] dfunc TODO: Remove
 * @param[in] perform_post_filtering turn off computing distances for
 *            additional landmarks outside of the closest k, if necessary.
 *            This can save a little computation time for approximate
 *            nearest neighbors and will generally return great recall.
 * @param[in] weight a weight for overlap between the closest landmark and
 *                   the radius of other landmarks when pruning distances.
 *                   Setting this value below 1 can effectively turn off
 *                   computing distances against many other balls, enabling
 *                   approximate nearest neighbors. Recall can be adjusted
 *                   based on how many relevant balls are ignored. Note that
 *                   many datasets can still have great recall even by only
 *                   looking in the closest landmark.
 */
template <typename value_idx = int64_t, typename value_t,
          typename value_int = int, typename distance_func>
void rbc_all_knn_query(const raft::handle_t &handle,
                       BallCoverIndex<value_idx, value_t> &index, int k,
                       value_idx *inds, value_t *dists,
                       // TODO: Remove this from user-facing API
                       distance_func dfunc,
                       // approximate nn options
                       bool perform_post_filtering = true, float weight = 1.0) {
  auto exec_policy = rmm::exec_policy(handle.get_stream());
  ASSERT(index.n_landmarks >= k, "number of landmark samples must be >= k");
  ASSERT(!index.is_index_trained(), "index cannot be previously trained");

  const int bitset_size = ceil(index.n_landmarks / 32.0);

  // Allocate all device memory upfront to increase asynchronicity

  rmm::device_uvector<value_idx> R_knn_inds(k * index.m, handle.get_stream());
  rmm::device_uvector<value_t> R_knn_dists(k * index.m, handle.get_stream());

  rmm::device_uvector<uint32_t> bitset(bitset_size * index.m,
                                       handle.get_stream());

  // For debugging / verification. Remove before releasing
  rmm::device_uvector<int> dists_counter(index.m, handle.get_stream());
  rmm::device_uvector<int> post_dists_counter(index.m, handle.get_stream());

  /**
   * 1. Randomly sample sqrt(n) points from X
   */
  sample_landmarks<value_idx, value_t>(handle, index);

  /**
   * 2. Perform knn = bfknn(X, R, k)
   */
  k_closest_landmarks(handle, index, index.get_X(), index.m, k,
                      R_knn_inds.data(), R_knn_dists.data());

  /**
   * 3. Create L_r = knn[:,0].T (CSR)
   *
   * Slice closest neighboring R
   * Secondary sort by (R_knn_inds, R_knn_dists)
   */
  construct_landmark_1nn(handle, R_knn_inds.data(), R_knn_dists.data(), k,
                         index);

  /**
   * Compute radius of each R for filtering: p(q, r) <= p(q, q_r) + radius(r)
   * (need to take the
   */
  compute_landmark_radii(handle, index);

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

  /**
   * TODO: Separate out the kernel invocations
   */

  // TODO: This will be a power of 2 cutoff for the number of dimensions
  // that will in smem in different buckets
  constexpr int rbc_tpb = 64;
  constexpr int max_vals = 300;

  if (index.n <= 2) {
    // Compute nearest k for each neighborhood in each closest R
    block_rbc_kernel_registers<value_idx, value_t, 32, 2, 128, 2>
      <<<index.m, 128, 0, handle.get_stream()>>>(
        index.get_X(), index.n, R_knn_inds.data(), R_knn_dists.data(), index.m,
        k, index.get_R_indptr(), index.get_R_1nn_cols(),
        index.get_R_1nn_dists(), inds, dists, dists_counter.data(),
        index.get_R_radius(), dfunc, weight);
  } else if (index.n <= max_vals) {
    printf("Calling smem rbc kernel\n");
    // Compute nearest k for each neighborhood in each closest R
    block_rbc_kernel_smem<value_idx, value_t, 32, 2, rbc_tpb, max_vals>
      <<<index.m, rbc_tpb, 0, handle.get_stream()>>>(
        index.get_X(), index.n, R_knn_inds.data(), R_knn_dists.data(), index.m,
        k, index.get_R_indptr(), index.get_R_1nn_cols(),
        index.get_R_1nn_dists(), inds, dists, dists_counter.data(),
        index.get_R_radius(), dfunc, raft::sparse::distance::SqDiff(),
        raft::sparse::distance::Sum(), weight);
  }

  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  raft::print_device_vector("dists", dists_counter.data(), 500, std::cout);

  // TODO: pull this out into "post_process()" function
  if (perform_post_filtering) {
    thrust::fill(exec_policy, post_dists_counter.data(),
                 post_dists_counter.data() + index.m, 0);

    if (index.n <= 2) {
      perform_post_filter_registers<value_idx, value_t, int, 128>
        <<<index.m, 128, bitset_size * sizeof(uint32_t), handle.get_stream()>>>(
          index.get_X(), index.n, R_knn_inds.data(), R_knn_dists.data(),
          index.get_R_radius(), index.get_R(), index.n_landmarks, bitset_size,
          k, dfunc, bitset.data(), weight);

      // Compute any distances from the landmarks that remain in the bitset
      compute_final_dists_registers<<<index.m, 128, 0, handle.get_stream()>>>(
        index.get_X(), index.n, bitset.data(), bitset_size, R_knn_dists.data(),
        index.get_R_indptr(), index.get_R_1nn_cols(), index.get_R_1nn_dists(),
        inds, dists, index.n_landmarks, k, dfunc, post_dists_counter.data());
    } else if (index.n <= max_vals) {
      printf("Calling smem post processing kernels\n");
      perform_post_filter<value_idx, value_t, int, 128>
        <<<index.m, 128, bitset_size * sizeof(uint32_t), handle.get_stream()>>>(
          index.get_X(), index.n, R_knn_inds.data(), R_knn_dists.data(),
          index.get_R_radius(), index.get_R(), dists, index.n_landmarks,
          bitset_size, k, bitset.data(), raft::sparse::distance::SqDiff(),
          raft::sparse::distance::Sum(), weight);

      CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

      raft::print_device_vector("bitset", bitset.data(), bitset_size * 5,
                                std::cout);

      printf("Computing final dists\n");
      // Compute any distances from the landmarks that remain in the bitset
      compute_final_dists_smem<value_idx, value_t, 32, 2, rbc_tpb, max_vals>
        <<<index.m, rbc_tpb, 0, handle.get_stream()>>>(
          index.get_X(), index.n, bitset.data(), bitset_size,
          R_knn_dists.data(), index.get_R_indptr(), index.get_R_1nn_cols(),
          index.get_R_1nn_dists(), inds, dists, index.n_landmarks, k, dfunc,
          raft::sparse::distance::SqDiff(), raft::sparse::distance::Sum(),
          post_dists_counter.data());

      CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
    }

    raft::print_device_vector("post dists", post_dists_counter.data(), 500,
                              std::cout);

    printf("Done.\n");
    //
    //    printf("total post_dists: %d\n", additional_dists);
  }

  //  raft::print_device_vector("R_knn 39077", R_knn_inds.data() + (k * 39077), k, std::cout);
  //  raft::print_device_vector("R_knn 39077 dists", R_knn_dists.data() + (k * 39077), k, std::cout);
  //  raft::print_device_vector("R_knn 35468", R_knn_inds.data() + (k * 35468), k, std::cout);
  //  raft::print_device_vector("R_knn 29384", R_knn_inds.data() + (k * 29384), k, std::cout);
  //  raft::print_device_vector("R_knn 29384 dists", R_knn_dists.data() + (k * 29384), k, std::cout);
  //  raft::print_device_vector("R 8", index.get_R() + (index.n * 8), index.n, std::cout);
  //  raft::print_device_vector("R 120", index.get_R() + (index.n * 120), index.n, std::cout);
}

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft
