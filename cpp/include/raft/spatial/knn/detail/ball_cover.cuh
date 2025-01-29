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

#include "../ball_cover_types.hpp"
#include "ball_cover/common.cuh"
#include "ball_cover/registers.cuh"
#include "haversine_distance.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/neighbors/brute_force.cuh>
#include <raft/neighbors/detail/faiss_select/key_value_block_select.cuh>
#include <raft/random/rng.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <limits.h>

#include <cstdint>

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
template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t>
void sample_landmarks(raft::resources const& handle,
                      BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index)
{
  rmm::device_uvector<value_idx> R_1nn_cols2(index.n_landmarks, resource::get_cuda_stream(handle));
  rmm::device_uvector<value_t> R_1nn_ones(index.m, resource::get_cuda_stream(handle));
  rmm::device_uvector<value_idx> R_indices(index.n_landmarks, resource::get_cuda_stream(handle));

  thrust::sequence(resource::get_thrust_policy(handle),
                   index.get_R_1nn_cols().data_handle(),
                   index.get_R_1nn_cols().data_handle() + index.m,
                   (value_idx)0);

  thrust::fill(resource::get_thrust_policy(handle),
               R_1nn_ones.data(),
               R_1nn_ones.data() + R_1nn_ones.size(),
               1.0);

  thrust::fill(resource::get_thrust_policy(handle),
               R_indices.data(),
               R_indices.data() + R_indices.size(),
               0.0);

  /**
   * 1. Randomly sample sqrt(n) points from X
   */
  raft::random::RngState rng_state(12345);
  raft::random::sampleWithoutReplacement(handle,
                                         rng_state,
                                         R_indices.data(),
                                         R_1nn_cols2.data(),
                                         index.get_R_1nn_cols().data_handle(),
                                         R_1nn_ones.data(),
                                         (value_idx)index.n_landmarks,
                                         (value_idx)index.m);

  auto x = index.get_X();
  auto r = index.get_R();

  raft::matrix::copy_rows<value_t, value_idx>(
    handle,
    make_device_matrix_view<const value_t, value_idx>(x.data_handle(), x.extent(0), x.extent(1)),
    make_device_matrix_view<value_t, value_idx>(r.data_handle(), r.extent(0), r.extent(1)),
    make_device_vector_view(R_1nn_cols2.data(), index.n_landmarks));
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
template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t>
void construct_landmark_1nn(raft::resources const& handle,
                            const value_idx* R_knn_inds_ptr,
                            const value_t* R_knn_dists_ptr,
                            value_int k,
                            BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index)
{
  rmm::device_uvector<value_idx> R_1nn_inds(index.m, resource::get_cuda_stream(handle));

  thrust::fill(resource::get_thrust_policy(handle),
               R_1nn_inds.data(),
               R_1nn_inds.data() + index.m,
               std::numeric_limits<value_idx>::max());

  value_idx* R_1nn_inds_ptr = R_1nn_inds.data();
  value_t* R_1nn_dists_ptr  = index.get_R_1nn_dists().data_handle();

  auto idxs = thrust::make_counting_iterator<value_idx>(0);
  thrust::for_each(
    resource::get_thrust_policy(handle), idxs, idxs + index.m, [=] __device__(value_idx i) {
      R_1nn_inds_ptr[i]  = R_knn_inds_ptr[i * k];
      R_1nn_dists_ptr[i] = R_knn_dists_ptr[i * k];
    });

  auto keys = thrust::make_zip_iterator(
    thrust::make_tuple(R_1nn_inds.data(), index.get_R_1nn_dists().data_handle()));

  // group neighborhoods for each reference landmark and sort each group by distance
  thrust::sort_by_key(resource::get_thrust_policy(handle),
                      keys,
                      keys + index.m,
                      index.get_R_1nn_cols().data_handle(),
                      NNComp());

  // convert to CSR for fast lookup
  raft::sparse::convert::sorted_coo_to_csr(R_1nn_inds.data(),
                                           index.m,
                                           index.get_R_indptr().data_handle(),
                                           index.n_landmarks + 1,
                                           resource::get_cuda_stream(handle));

  // reorder X to allow aligned access
  raft::matrix::copy_rows<value_t, value_idx>(
    handle, index.get_X(), index.get_X_reordered(), index.get_R_1nn_cols());
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
template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t>
void k_closest_landmarks(raft::resources const& handle,
                         const BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
                         const value_t* query_pts,
                         value_int n_query_pts,
                         value_int k,
                         value_idx* R_knn_inds,
                         value_t* R_knn_dists)
{
  std::vector<raft::device_matrix_view<const value_t, value_int>> inputs = {index.get_R()};

  raft::neighbors::brute_force::knn<value_idx, value_t, value_int>(
    handle,
    inputs,
    make_device_matrix_view(query_pts, n_query_pts, inputs[0].extent(1)),
    make_device_matrix_view(R_knn_inds, n_query_pts, k),
    make_device_matrix_view(R_knn_dists, n_query_pts, k),
    index.get_metric());
}

/**
 * Uses the sorted data points in the 1-nn landmark index to compute
 * an array of radii for each landmark.
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param index
 */
template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t>
void compute_landmark_radii(raft::resources const& handle,
                            BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index)
{
  auto entries = thrust::make_counting_iterator<value_idx>(0);

  const value_idx* R_indptr_ptr  = index.get_R_indptr().data_handle();
  const value_t* R_1nn_dists_ptr = index.get_R_1nn_dists().data_handle();
  value_t* R_radius_ptr          = index.get_R_radius().data_handle();
  thrust::for_each(resource::get_thrust_policy(handle),
                   entries,
                   entries + index.n_landmarks,
                   [=] __device__(value_idx input) {
                     value_idx last_row_idx = R_indptr_ptr[input + 1] - 1;
                     R_radius_ptr[input]    = R_1nn_dists_ptr[last_row_idx];
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
template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          typename dist_func>
void perform_rbc_query(raft::resources const& handle,
                       const BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
                       const value_t* query,
                       value_int n_query_pts,
                       value_int k,
                       const value_idx* R_knn_inds,
                       const value_t* R_knn_dists,
                       dist_func dfunc,
                       value_idx* inds,
                       value_t* dists,
                       value_int* dists_counter,
                       value_int* post_dists_counter,
                       float weight                = 1.0,
                       bool perform_post_filtering = true)
{
  // initialize output inds and dists
  thrust::fill(resource::get_thrust_policy(handle),
               inds,
               inds + (k * n_query_pts),
               std::numeric_limits<value_idx>::max());
  thrust::fill(resource::get_thrust_policy(handle),
               dists,
               dists + (k * n_query_pts),
               std::numeric_limits<value_t>::max());

  if (index.n == 2) {
    // Compute nearest k for each neighborhood in each closest R
    rbc_low_dim_pass_one<value_idx, value_t, value_int, matrix_idx, 2>(handle,
                                                                       index,
                                                                       query,
                                                                       n_query_pts,
                                                                       k,
                                                                       R_knn_inds,
                                                                       R_knn_dists,
                                                                       dfunc,
                                                                       inds,
                                                                       dists,
                                                                       weight,
                                                                       dists_counter);

    if (perform_post_filtering) {
      rbc_low_dim_pass_two<value_idx, value_t, value_int, matrix_idx, 2>(handle,
                                                                         index,
                                                                         query,
                                                                         n_query_pts,
                                                                         k,
                                                                         R_knn_inds,
                                                                         R_knn_dists,
                                                                         dfunc,
                                                                         inds,
                                                                         dists,
                                                                         weight,
                                                                         post_dists_counter);
    }

  } else if (index.n == 3) {
    // Compute nearest k for each neighborhood in each closest R
    rbc_low_dim_pass_one<value_idx, value_t, value_int, matrix_idx, 3>(handle,
                                                                       index,
                                                                       query,
                                                                       n_query_pts,
                                                                       k,
                                                                       R_knn_inds,
                                                                       R_knn_dists,
                                                                       dfunc,
                                                                       inds,
                                                                       dists,
                                                                       weight,
                                                                       dists_counter);

    if (perform_post_filtering) {
      rbc_low_dim_pass_two<value_idx, value_t, value_int, matrix_idx, 3>(handle,
                                                                         index,
                                                                         query,
                                                                         n_query_pts,
                                                                         k,
                                                                         R_knn_inds,
                                                                         R_knn_dists,
                                                                         dfunc,
                                                                         inds,
                                                                         dists,
                                                                         weight,
                                                                         post_dists_counter);
    }
  }
}

/**
 * Perform eps-select
 *
 */
template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          typename dist_func>
void perform_rbc_eps_nn_query(
  raft::resources const& handle,
  const BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
  const value_t* query,
  value_int n_query_pts,
  value_t eps,
  const value_t* landmarks,
  dist_func dfunc,
  bool* adj,
  value_idx* vd)
{
  // initialize output
  RAFT_CUDA_TRY(cudaMemsetAsync(
    adj, 0, index.m * n_query_pts * sizeof(bool), resource::get_cuda_stream(handle)));

  resource::sync_stream(handle);

  rbc_eps_pass<value_idx, value_t, value_int, matrix_idx>(
    handle, index, query, n_query_pts, eps, landmarks, dfunc, adj, vd);

  resource::sync_stream(handle);
}

template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          typename dist_func>
void perform_rbc_eps_nn_query(
  raft::resources const& handle,
  const BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
  const value_t* query,
  value_int n_query_pts,
  value_t eps,
  value_int* max_k,
  const value_t* landmarks,
  dist_func dfunc,
  value_idx* adj_ia,
  value_idx* adj_ja,
  value_idx* vd)
{
  rbc_eps_pass<value_idx, value_t, value_int, matrix_idx>(
    handle, index, query, n_query_pts, eps, max_k, landmarks, dfunc, adj_ia, adj_ja, vd);

  resource::sync_stream(handle);
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
template <typename value_idx = std::int64_t,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          typename distance_func>
void rbc_build_index(raft::resources const& handle,
                     BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
                     distance_func dfunc)
{
  ASSERT(!index.is_index_trained(), "index cannot be previously trained");

  rmm::device_uvector<value_idx> R_knn_inds(index.m, resource::get_cuda_stream(handle));

  // Initialize the uvectors
  thrust::fill(resource::get_thrust_policy(handle),
               R_knn_inds.begin(),
               R_knn_inds.end(),
               std::numeric_limits<value_idx>::max());
  thrust::fill(resource::get_thrust_policy(handle),
               index.get_R_closest_landmark_dists().data_handle(),
               index.get_R_closest_landmark_dists().data_handle() + index.m,
               std::numeric_limits<value_t>::max());

  /**
   * 1. Randomly sample sqrt(n) points from X
   */
  sample_landmarks<value_idx, value_t>(handle, index);

  /**
   * 2. Perform knn = bfknn(X, R, k)
   */
  value_int k = 1;
  k_closest_landmarks(handle,
                      index,
                      index.get_X().data_handle(),
                      index.m,
                      k,
                      R_knn_inds.data(),
                      index.get_R_closest_landmark_dists().data_handle());

  /**
   * 3. Create L_r = knn[:,0].T (CSR)
   *
   * Slice closest neighboring R
   * Secondary sort by (R_knn_inds, R_knn_dists)
   */
  construct_landmark_1nn(
    handle, R_knn_inds.data(), index.get_R_closest_landmark_dists().data_handle(), k, index);

  /**
   * Compute radius of each R for filtering: p(q, r) <= p(q, q_r) + radius(r)
   * (need to take the
   */
  compute_landmark_radii(handle, index);
}

/**
 * Performs an all neighbors knn query (e.g. index == query)
 */
template <typename value_idx = std::int64_t,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          typename distance_func>
void rbc_all_knn_query(raft::resources const& handle,
                       BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
                       value_int k,
                       value_idx* inds,
                       value_t* dists,
                       distance_func dfunc,
                       // approximate nn options
                       bool perform_post_filtering = true,
                       float weight                = 1.0)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  ASSERT(index.n_landmarks >= k, "number of landmark samples must be >= k");
  ASSERT(!index.is_index_trained(), "index cannot be previously trained");

  rmm::device_uvector<value_idx> R_knn_inds(k * index.m, resource::get_cuda_stream(handle));
  rmm::device_uvector<value_t> R_knn_dists(k * index.m, resource::get_cuda_stream(handle));

  // Initialize the uvectors
  thrust::fill(resource::get_thrust_policy(handle),
               R_knn_inds.begin(),
               R_knn_inds.end(),
               std::numeric_limits<value_idx>::max());
  thrust::fill(resource::get_thrust_policy(handle),
               R_knn_dists.begin(),
               R_knn_dists.end(),
               std::numeric_limits<value_t>::max());

  thrust::fill(resource::get_thrust_policy(handle),
               inds,
               inds + (k * index.m),
               std::numeric_limits<value_idx>::max());
  thrust::fill(resource::get_thrust_policy(handle),
               dists,
               dists + (k * index.m),
               std::numeric_limits<value_t>::max());

  // For debugging / verification. Remove before releasing
  rmm::device_uvector<value_int> dists_counter(index.m, resource::get_cuda_stream(handle));
  rmm::device_uvector<value_int> post_dists_counter(index.m, resource::get_cuda_stream(handle));

  sample_landmarks<value_idx, value_t>(handle, index);

  k_closest_landmarks(
    handle, index, index.get_X().data_handle(), index.m, k, R_knn_inds.data(), R_knn_dists.data());

  construct_landmark_1nn(handle, R_knn_inds.data(), R_knn_dists.data(), k, index);

  compute_landmark_radii(handle, index);

  perform_rbc_query(handle,
                    index,
                    index.get_X().data_handle(),
                    index.m,
                    k,
                    R_knn_inds.data(),
                    R_knn_dists.data(),
                    dfunc,
                    inds,
                    dists,
                    dists_counter.data(),
                    post_dists_counter.data(),
                    weight,
                    perform_post_filtering);
}

/**
 * Performs a knn query against an index. This assumes the index has
 * already been built.
 */
template <typename value_idx = std::int64_t,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          typename distance_func>
void rbc_knn_query(raft::resources const& handle,
                   const BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
                   value_int k,
                   const value_t* query,
                   value_int n_query_pts,
                   value_idx* inds,
                   value_t* dists,
                   distance_func dfunc,
                   // approximate nn options
                   bool perform_post_filtering = true,
                   float weight                = 1.0)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  ASSERT(index.n_landmarks >= k, "number of landmark samples must be >= k");
  ASSERT(index.is_index_trained(), "index must be previously trained");

  rmm::device_uvector<value_idx> R_knn_inds(k * n_query_pts, resource::get_cuda_stream(handle));
  rmm::device_uvector<value_t> R_knn_dists(k * n_query_pts, resource::get_cuda_stream(handle));

  // Initialize the uvectors
  thrust::fill(resource::get_thrust_policy(handle),
               R_knn_inds.begin(),
               R_knn_inds.end(),
               std::numeric_limits<value_idx>::max());
  thrust::fill(resource::get_thrust_policy(handle),
               R_knn_dists.begin(),
               R_knn_dists.end(),
               std::numeric_limits<value_t>::max());

  thrust::fill(resource::get_thrust_policy(handle),
               inds,
               inds + (k * n_query_pts),
               std::numeric_limits<value_idx>::max());
  thrust::fill(resource::get_thrust_policy(handle),
               dists,
               dists + (k * n_query_pts),
               std::numeric_limits<value_t>::max());

  k_closest_landmarks(handle, index, query, n_query_pts, k, R_knn_inds.data(), R_knn_dists.data());

  // For debugging / verification. Remove before releasing
  rmm::device_uvector<value_int> dists_counter(index.m, resource::get_cuda_stream(handle));
  rmm::device_uvector<value_int> post_dists_counter(index.m, resource::get_cuda_stream(handle));
  thrust::fill(resource::get_thrust_policy(handle),
               post_dists_counter.data(),
               post_dists_counter.data() + post_dists_counter.size(),
               0);
  thrust::fill(resource::get_thrust_policy(handle),
               dists_counter.data(),
               dists_counter.data() + dists_counter.size(),
               0);

  perform_rbc_query(handle,
                    index,
                    query,
                    n_query_pts,
                    k,
                    R_knn_inds.data(),
                    R_knn_dists.data(),
                    dfunc,
                    inds,
                    dists,
                    dists_counter.data(),
                    post_dists_counter.data(),
                    weight,
                    perform_post_filtering);
}

template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t>
void compute_landmark_dists(raft::resources const& handle,
                            const BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
                            const value_t* query_pts,
                            value_int n_query_pts,
                            value_t* R_dists)
{
  // compute distances for all queries against all landmarks
  // index.get_R() -- landmark points in row order (index.n_landmarks x index.k)
  // query_pts     -- query points in row order (n_query_pts x index.k)
  RAFT_EXPECTS(std::max<size_t>(index.n_landmarks, n_query_pts) * index.n <
                 static_cast<size_t>(std::numeric_limits<int>::max()),
               "Too large input for pairwise_distance with `int` index.");
  RAFT_EXPECTS(n_query_pts * static_cast<size_t>(index.n_landmarks) <
                 static_cast<size_t>(std::numeric_limits<int>::max()),
               "Too large input for pairwise_distance with `int` index.");
  raft::distance::pairwise_distance<value_t, int>(handle,
                                                  query_pts,
                                                  index.get_R().data_handle(),
                                                  R_dists,
                                                  n_query_pts,
                                                  index.n_landmarks,
                                                  index.n,
                                                  index.get_metric());
}

/**
 * Performs a knn query against an index. This assumes the index has
 * already been built.
 * Modified version that takes an eps as threshold and outputs to a dense adj matrix (row-major)
 * we are assuming that there are sufficiently many landmarks
 */
template <typename value_idx = std::int64_t,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          typename distance_func>
void rbc_eps_nn_query(raft::resources const& handle,
                      const BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
                      const value_t eps,
                      const value_t* query,
                      value_int n_query_pts,
                      bool* adj,
                      value_idx* vd,
                      distance_func dfunc)
{
  ASSERT(index.is_index_trained(), "index must be previously trained");

  // query all points and write to adj
  perform_rbc_eps_nn_query(
    handle, index, query, n_query_pts, eps, index.get_R().data_handle(), dfunc, adj, vd);
}

template <typename value_idx = std::int64_t,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          typename distance_func>
void rbc_eps_nn_query(raft::resources const& handle,
                      const BallCoverIndex<value_idx, value_t, value_int, matrix_idx>& index,
                      const value_t eps,
                      value_int* max_k,
                      const value_t* query,
                      value_int n_query_pts,
                      value_idx* adj_ia,
                      value_idx* adj_ja,
                      value_idx* vd,
                      distance_func dfunc)
{
  ASSERT(index.is_index_trained(), "index must be previously trained");

  // query all points and write to adj
  perform_rbc_eps_nn_query(handle,
                           index,
                           query,
                           n_query_pts,
                           eps,
                           max_k,
                           index.get_R().data_handle(),
                           dfunc,
                           adj_ia,
                           adj_ja,
                           vd);
}

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft
