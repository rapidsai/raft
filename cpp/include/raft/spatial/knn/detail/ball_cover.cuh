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
#include "ball_cover/plan.cuh"
#include "ball_cover/registers.cuh"
#include "block_select_faiss.cuh"
#include "haversine_distance.cuh"
#include "knn_brute_force_faiss.cuh"
#include "selection_faiss.cuh"

#include <limits.h>
#include <cstdint>

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
template <typename value_idx, typename value_t,
          typename value_int = std::uint32_t>
void sample_landmarks(const raft::handle_t &handle,
                      BallCoverIndex<value_idx, value_t, value_int> &index) {
  rmm::device_uvector<value_idx> R_1nn_cols2(index.n_landmarks,
                                             handle.get_stream());
  rmm::device_uvector<value_t> R_1nn_ones(index.m, handle.get_stream());
  rmm::device_uvector<value_idx> R_indices(index.n_landmarks,
                                           handle.get_stream());

  thrust::sequence(handle.get_thrust_policy(), index.get_R_1nn_cols(),
                   index.get_R_1nn_cols() + index.m, (value_idx)0);

  thrust::fill(handle.get_thrust_policy(), R_1nn_ones.data(),
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
template <typename value_idx, typename value_t,
          typename value_int = std::uint32_t>
void construct_landmark_1nn(
  const raft::handle_t &handle, const value_idx *R_knn_inds_ptr,
  const value_t *R_knn_dists_ptr, value_int k,
  BallCoverIndex<value_idx, value_t, value_int> &index) {
  rmm::device_uvector<value_idx> R_1nn_inds(index.m, handle.get_stream());

  value_idx *R_1nn_inds_ptr = R_1nn_inds.data();
  value_t *R_1nn_dists_ptr = index.get_R_1nn_dists();

  auto idxs = thrust::make_counting_iterator<value_idx>(0);
  thrust::for_each(handle.get_thrust_policy(), idxs, idxs + index.m,
                   [=] __device__(value_idx i) {
                     R_1nn_inds_ptr[i] = R_knn_inds_ptr[i * k];
                     R_1nn_dists_ptr[i] = R_knn_dists_ptr[i * k];
                   });

  auto keys = thrust::make_zip_iterator(
    thrust::make_tuple(R_1nn_inds.data(), index.get_R_1nn_dists()));

  // group neighborhoods for each reference landmark and sort each group by distance
  thrust::sort_by_key(handle.get_thrust_policy(), keys, keys + index.m,
                      index.get_R_1nn_cols(), NNComp());

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
template <typename value_idx, typename value_t,
          typename value_int = std::uint32_t>
void k_closest_landmarks(const raft::handle_t &handle,
                         BallCoverIndex<value_idx, value_t, value_int> &index,
                         const value_t *query_pts, value_int n_query_pts,
                         value_int k, value_idx *R_knn_inds,
                         value_t *R_knn_dists) {
  std::vector<value_t *> input = {index.get_R()};
  std::vector<std::uint32_t> sizes = {index.n_landmarks};

  brute_force_knn_impl<std::uint32_t, std::int64_t>(
    input, sizes, index.n, const_cast<value_t *>(query_pts), n_query_pts,
    R_knn_inds, R_knn_dists, k, handle.get_stream(), nullptr, 0, true, true,
    nullptr, index.metric);
}

/**
 * Uses the sorted data points in the 1-nn landmark index to compute
 * an array of radii for each landmark.
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param index
 */
template <typename value_idx, typename value_t,
          typename value_int = std::uint32_t>
void compute_landmark_radii(
  const raft::handle_t &handle,
  BallCoverIndex<value_idx, value_t, value_int> &index) {
  auto entries = thrust::make_counting_iterator<value_idx>(0);

  const value_idx *R_indptr_ptr = index.get_R_indptr();
  const value_t *R_1nn_dists_ptr = index.get_R_1nn_dists();
  value_t *R_radius_ptr = index.get_R_radius();
  thrust::for_each(handle.get_thrust_policy(), entries,
                   entries + index.n_landmarks,
                   [=] __device__(value_idx input) {
                     value_idx last_row_idx = R_indptr_ptr[input + 1] - 1;
                     R_radius_ptr[input] = R_1nn_dists_ptr[last_row_idx];
                   });
}

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
template <typename value_idx, typename value_t,
          typename value_int = std::uint32_t, typename dist_func>
void perform_rbc_query(const raft::handle_t &handle,
                       BallCoverIndex<value_idx, value_t, value_int> &index,
                       const value_t *query, value_int n_query_pts,
                       std::uint32_t k, const value_idx *R_knn_inds,
                       const value_t *R_knn_dists, dist_func dfunc,
                       value_idx *inds, value_t *dists,
                       value_int *dists_counter, value_int *post_dists_counter,
                       float weight = 1.0, bool perform_post_filtering = true) {
  // Compute nearest k for each neighborhood in each closest R
  rbc_low_dim_pass_one(handle, index, query, n_query_pts, k, R_knn_inds,
                       R_knn_dists, dfunc, inds, dists, weight, dists_counter);

  if (perform_post_filtering) {
    rbc_low_dim_pass_two(handle, index, query, n_query_pts, k, R_knn_inds,
                         R_knn_dists, dfunc, inds, dists, weight,
                         post_dists_counter);
  }
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
 */
template <typename value_idx = std::int64_t, typename value_t,
          typename value_int = std::uint32_t, typename distance_func>
void rbc_build_index(const raft::handle_t &handle,
                     BallCoverIndex<value_idx, value_t, value_int> &index,
                     distance_func dfunc) {
  ASSERT(!index.is_index_trained(), "index cannot be previously trained");

  rmm::device_uvector<value_idx> R_knn_inds(index.m, handle.get_stream());
  rmm::device_uvector<value_t> R_knn_dists(index.m, handle.get_stream());

  /**
   * 1. Randomly sample sqrt(n) points from X
   */
  sample_landmarks<value_idx, value_t>(handle, index);

  /**
   * 2. Perform knn = bfknn(X, R, k)
   */
  value_int k = 1;
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
}

/**
 * Performs an all neighbors knn query (e.g. index == query)
 */
template <typename value_idx = std::int64_t, typename value_t,
          typename value_int = std::uint32_t, typename distance_func>
void rbc_all_knn_query(const raft::handle_t &handle,
                       BallCoverIndex<value_idx, value_t, value_int> &index,
                       value_int k, value_idx *inds, value_t *dists,
                       distance_func dfunc,
                       // approximate nn options
                       bool perform_post_filtering = true, float weight = 1.0) {
  ASSERT(index.n_landmarks >= k, "number of landmark samples must be >= k");
  ASSERT(!index.is_index_trained(), "index cannot be previously trained");

  rmm::device_uvector<value_idx> R_knn_inds(k * index.m, handle.get_stream());
  rmm::device_uvector<value_t> R_knn_dists(k * index.m, handle.get_stream());

  // For debugging / verification. Remove before releasing
  rmm::device_uvector<value_int> dists_counter(index.m, handle.get_stream());
  rmm::device_uvector<value_int> post_dists_counter(index.m,
                                                    handle.get_stream());

  sample_landmarks<value_idx, value_t>(handle, index);

  k_closest_landmarks(handle, index, index.get_X(), index.m, k,
                      R_knn_inds.data(), R_knn_dists.data());

  construct_landmark_1nn(handle, R_knn_inds.data(), R_knn_dists.data(), k,
                         index);

  compute_landmark_radii(handle, index);

  if (index.n == 2) {
    //      perform_rbc_query(handle, index, index.get_X(), index.m, k, R_knn_inds.data(),
    //                        R_knn_dists.data(), dfunc, inds, dists,
    //                        dists_counter.data(), post_dists_counter.data(), weight,
    //                        perform_post_filtering);
  } else {
    thrust::fill(handle.get_thrust_policy(), dists, dists,
                 std::numeric_limits<value_t>::max());
    raft::sparse::COO<value_idx, value_idx> plan_coo(handle.get_stream());

    rbc_build_index(handle, index, EuclideanFunc());
    compute_plan(handle, index, k, index.get_X(), index.m, inds, dists,
                 plan_coo, weight);
    execute_plan(handle, index, plan_coo, k, index.get_X(), index.m, inds,
                 dists, weight);
  }
}

/**
 * Performs a knn query against an index. This assumes the index has
 * already been built.
 */
template <typename value_idx = std::int64_t, typename value_t,
          typename value_int = std::uint32_t, typename distance_func>
void rbc_knn_query(const raft::handle_t &handle,
                   BallCoverIndex<value_idx, value_t, value_int> &index,
                   value_int k, const value_t *query, value_int n_query_pts,
                   value_idx *inds, value_t *dists, distance_func dfunc,
                   // approximate nn options
                   bool perform_post_filtering = true, float weight = 1.0) {
  ASSERT(index.n_landmarks >= k, "number of landmark samples must be >= k");
  ASSERT(index.is_index_trained(), "index must be previously trained");

  rmm::device_uvector<value_idx> R_knn_inds(k * index.m, handle.get_stream());
  rmm::device_uvector<value_t> R_knn_dists(k * index.m, handle.get_stream());

  k_closest_landmarks(handle, index, query, n_query_pts, k, R_knn_inds.data(),
                      R_knn_dists.data());

  // For debugging / verification. Remove before releasing
  rmm::device_uvector<value_int> dists_counter(index.m, handle.get_stream());
  rmm::device_uvector<value_int> post_dists_counter(index.m,
                                                    handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), post_dists_counter.data(),
               post_dists_counter.data() + index.m, 0);

  if (index.n == 2) {
    //      perform_rbc_query(handle, index, query, n_query_pts, k, R_knn_inds.data(),
    //                        R_knn_dists.data(), dfunc, inds, dists,
    //                        dists_counter.data(), post_dists_counter.data(), weight,
    //                        perform_post_filtering);
  } else {
    thrust::fill(handle.get_thrust_policy(), dists, dists,
                 std::numeric_limits<value_t>::max());
    raft::sparse::COO<value_idx, value_idx> plan_coo(handle.get_stream());
    compute_plan(handle, index, k, query, n_query_pts, inds, dists, plan_coo,
                 weight);
    execute_plan(handle, index, plan_coo, k, query, n_query_pts, inds, dists,
                 weight);
  }
}

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft
