/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

/**
 * Note: This file is deprecated and will be removed in a future release
 * Please use include/raft/cluster/kmeans.cuh instead
 */

#pragma once

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/spectral/detail/warn_dbg.hpp>
#include <raft/spectral/matrix_wrappers.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_atomics.cuh>

#include <cuda.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/memory.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>

namespace raft {
namespace cluster {
namespace detail {
// =========================================================
// Useful grid settings
// =========================================================

constexpr unsigned int BLOCK_SIZE      = 1024;
constexpr unsigned int WARP_SIZE       = 32;
constexpr unsigned int BSIZE_DIV_WSIZE = (BLOCK_SIZE / WARP_SIZE);

// =========================================================
// CUDA kernels
// =========================================================

/**
 *  @brief Compute distances between observation vectors and centroids
 *    Block dimensions should be (warpSize, 1,
 *    blockSize/warpSize). Ideally, the grid is large enough so there
 *    are d threads in the x-direction, k threads in the y-direction,
 *    and n threads in the z-direction.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param n Number of observation vectors.
 *  @param d Dimension of observation vectors.
 *  @param k Number of clusters.
 *  @param obs (Input, d*n entries) Observation matrix. Matrix is
 *    stored column-major and each column is an observation
 *    vector. Matrix dimensions are d x n.
 *  @param centroids (Input, d*k entries) Centroid matrix. Matrix is
 *    stored column-major and each column is a centroid. Matrix
 *    dimensions are d x k.
 *  @param dists (Output, n*k entries) Distance matrix. Matrix is
 *    stored column-major and the (i,j)-entry is the square of the
 *    Euclidean distance between the ith observation vector and jth
 *    centroid. Matrix dimensions are n x k. Entries must be
 *    initialized to zero.
 */
template <typename index_type_t, typename value_type_t>
RAFT_KERNEL computeDistances(index_type_t n,
                             index_type_t d,
                             index_type_t k,
                             const value_type_t* __restrict__ obs,
                             const value_type_t* __restrict__ centroids,
                             value_type_t* __restrict__ dists)
{
  // Loop index
  index_type_t i;

  // Block indices
  index_type_t bidx;
  // Global indices
  index_type_t gidx, gidy, gidz;

  // Private memory
  value_type_t centroid_private, dist_private;

  // Global x-index indicates index of vector entry
  bidx = blockIdx.x;
  while (bidx * blockDim.x < d) {
    gidx = threadIdx.x + bidx * blockDim.x;

    // Global y-index indicates centroid
    gidy = threadIdx.y + blockIdx.y * blockDim.y;
    while (gidy < k) {
      // Load centroid coordinate from global memory
      centroid_private = (gidx < d) ? centroids[IDX(gidx, gidy, d)] : 0;

      // Global z-index indicates observation vector
      gidz = threadIdx.z + blockIdx.z * blockDim.z;
      while (gidz < n) {
        // Load observation vector coordinate from global memory
        dist_private = (gidx < d) ? obs[IDX(gidx, gidz, d)] : 0;

        // Compute contribution of current entry to distance
        dist_private = centroid_private - dist_private;
        dist_private = dist_private * dist_private;

        // Perform reduction on warp
        for (i = WARP_SIZE / 2; i > 0; i /= 2)
          dist_private += __shfl_down_sync(warp_full_mask(), dist_private, i, 2 * i);

        // Write result to global memory
        if (threadIdx.x == 0) atomicAdd(dists + IDX(gidz, gidy, n), dist_private);

        // Move to another observation vector
        gidz += blockDim.z * gridDim.z;
      }

      // Move to another centroid
      gidy += blockDim.y * gridDim.y;
    }

    // Move to another vector entry
    bidx += gridDim.x;
  }
}

/**
 *  @brief Find closest centroid to observation vectors.
 *    Block and grid dimensions should be 1-dimensional. Ideally the
 *    grid is large enough so there are n threads.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param n Number of observation vectors.
 *  @param k Number of clusters.
 *  @param centroids (Input, d*k entries) Centroid matrix. Matrix is
 *    stored column-major and each column is a centroid. Matrix
 *    dimensions are d x k.
 *  @param dists (Input/output, n*k entries) Distance matrix. Matrix
 *    is stored column-major and the (i,j)-entry is the square of
 *    the Euclidean distance between the ith observation vector and
 *    jth centroid. Matrix dimensions are n x k. On exit, the first
 *    n entries give the square of the Euclidean distance between
 *    observation vectors and closest centroids.
 *  @param codes (Output, n entries) Cluster assignments.
 *  @param clusterSizes (Output, k entries) Number of points in each
 *    cluster. Entries must be initialized to zero.
 */
template <typename index_type_t, typename value_type_t>
RAFT_KERNEL minDistances(index_type_t n,
                         index_type_t k,
                         value_type_t* __restrict__ dists,
                         index_type_t* __restrict__ codes,
                         index_type_t* __restrict__ clusterSizes)
{
  // Loop index
  index_type_t i, j;

  // Current matrix entry
  value_type_t dist_curr;

  // Smallest entry in row
  value_type_t dist_min;
  index_type_t code_min;

  // Each row in observation matrix is processed by a thread
  i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    // Find minimum entry in row
    code_min = 0;
    dist_min = dists[IDX(i, 0, n)];
    for (j = 1; j < k; ++j) {
      dist_curr = dists[IDX(i, j, n)];
      code_min  = (dist_curr < dist_min) ? j : code_min;
      dist_min  = (dist_curr < dist_min) ? dist_curr : dist_min;
    }

    // Transfer result to global memory
    dists[i] = dist_min;
    codes[i] = code_min;

    // Increment cluster sizes
    atomicAdd(clusterSizes + code_min, 1);

    // Move to another row
    i += blockDim.x * gridDim.x;
  }
}

/**
 *  @brief Check if newly computed distances are smaller than old distances.
 *    Block and grid dimensions should be 1-dimensional. Ideally the
 *    grid is large enough so there are n threads.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param n Number of observation vectors.
 *  @param dists_old (Input/output, n entries) Distances between
 *    observation vectors and closest centroids. On exit, entries
 *    are replaced by entries in 'dists_new' if the corresponding
 *    observation vectors are closest to the new centroid.
 *  @param dists_new (Input, n entries) Distance between observation
 *    vectors and new centroid.
 *  @param codes_old (Input/output, n entries) Cluster
 *    assignments. On exit, entries are replaced with 'code_new' if
 *    the corresponding observation vectors are closest to the new
 *    centroid.
 *  @param code_new Index associated with new centroid.
 */
template <typename index_type_t, typename value_type_t>
RAFT_KERNEL minDistances2(index_type_t n,
                          value_type_t* __restrict__ dists_old,
                          const value_type_t* __restrict__ dists_new,
                          index_type_t* __restrict__ codes_old,
                          index_type_t code_new)
{
  // Loop index
  index_type_t i = threadIdx.x + blockIdx.x * blockDim.x;

  // Distances
  value_type_t dist_old_private;
  value_type_t dist_new_private;

  // Each row is processed by a thread
  while (i < n) {
    // Get old and new distances
    dist_old_private = dists_old[i];
    dist_new_private = dists_new[i];

    // Update if new distance is smaller than old distance
    if (dist_new_private < dist_old_private) {
      dists_old[i] = dist_new_private;
      codes_old[i] = code_new;
    }

    // Move to another row
    i += blockDim.x * gridDim.x;
  }
}

/**
 *  @brief Compute size of k-means clusters.
 *    Block and grid dimensions should be 1-dimensional. Ideally the
 *    grid is large enough so there are n threads.
 *  @tparam index_type_t the type of data used for indexing.
 *  @param n Number of observation vectors.
 *  @param k Number of clusters.
 *  @param codes (Input, n entries) Cluster assignments.
 *  @param clusterSizes (Output, k entries) Number of points in each
 *    cluster. Entries must be initialized to zero.
 */
template <typename index_type_t>
RAFT_KERNEL computeClusterSizes(index_type_t n,
                                const index_type_t* __restrict__ codes,
                                index_type_t* __restrict__ clusterSizes)
{
  index_type_t i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    atomicAdd(clusterSizes + codes[i], 1);
    i += blockDim.x * gridDim.x;
  }
}

/**
 *  @brief Divide rows of centroid matrix by cluster sizes.
 *    Divides the ith column of the sum matrix by the size of the ith
 *    cluster. If the sum matrix has been initialized so that the ith
 *    row is the sum of all observation vectors in the ith cluster,
 *    this kernel produces cluster centroids. The grid and block
 *    dimensions should be 2-dimensional. Ideally the grid is large
 *    enough so there are d threads in the x-direction and k threads
 *    in the y-direction.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param d Dimension of observation vectors.
 *  @param k Number of clusters.
 *  @param clusterSizes (Input, k entries) Number of points in each
 *    cluster.
 *  @param centroids (Input/output, d*k entries) Sum matrix. Matrix
 *    is stored column-major and matrix dimensions are d x k. The
 *    ith column is the sum of all observation vectors in the ith
 *    cluster. On exit, the matrix is the centroid matrix (each
 *    column is the mean position of a cluster).
 */
template <typename index_type_t, typename value_type_t>
RAFT_KERNEL divideCentroids(index_type_t d,
                            index_type_t k,
                            const index_type_t* __restrict__ clusterSizes,
                            value_type_t* __restrict__ centroids)
{
  // Global indices
  index_type_t gidx, gidy;

  // Current cluster size
  index_type_t clusterSize_private;

  // Observation vector is determined by global y-index
  gidy = threadIdx.y + blockIdx.y * blockDim.y;
  while (gidy < k) {
    // Get cluster size from global memory
    clusterSize_private = clusterSizes[gidy];

    // Add vector entries to centroid matrix
    //   vector entris are determined by global x-index
    gidx = threadIdx.x + blockIdx.x * blockDim.x;
    while (gidx < d) {
      centroids[IDX(gidx, gidy, d)] /= clusterSize_private;
      gidx += blockDim.x * gridDim.x;
    }

    // Move to another centroid
    gidy += blockDim.y * gridDim.y;
  }
}

// =========================================================
// Helper functions
// =========================================================

/**
 *  @brief Randomly choose new centroids.
 *    Centroid is randomly chosen with k-means++ algorithm.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param n Number of observation vectors.
 *  @param d Dimension of observation vectors.
 *  @param k Number of clusters.
 *  @param rand Random number drawn uniformly from [0,1).
 *  @param obs (Input, device memory, d*n entries) Observation
 *    matrix. Matrix is stored column-major and each column is an
 *    observation vector. Matrix dimensions are n x d.
 *  @param dists (Input, device memory, 2*n entries) Workspace. The
 *    first n entries should be the distance between observation
 *    vectors and the closest centroid.
 *  @param centroid (Output, device memory, d entries) Centroid
 *    coordinates.
 *  @return Zero if successful. Otherwise non-zero.
 */
template <typename index_type_t, typename value_type_t>
static int chooseNewCentroid(raft::resources const& handle,
                             index_type_t n,
                             index_type_t d,
                             value_type_t rand,
                             const value_type_t* __restrict__ obs,
                             value_type_t* __restrict__ dists,
                             value_type_t* __restrict__ centroid)
{
  // Cumulative sum of distances
  value_type_t* distsCumSum = dists + n;
  // Residual sum of squares
  value_type_t distsSum{0};
  // Observation vector that is chosen as new centroid
  index_type_t obsIndex;

  auto stream             = resource::get_cuda_stream(handle);
  auto thrust_exec_policy = resource::get_thrust_policy(handle);

  // Compute cumulative sum of distances
  thrust::inclusive_scan(thrust_exec_policy,
                         thrust::device_pointer_cast(dists),
                         thrust::device_pointer_cast(dists + n),
                         thrust::device_pointer_cast(distsCumSum));
  RAFT_CHECK_CUDA(stream);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    &distsSum, distsCumSum + n - 1, sizeof(value_type_t), cudaMemcpyDeviceToHost, stream));

  // Randomly choose observation vector
  //   Probabilities are proportional to square of distance to closest
  //   centroid (see k-means++ algorithm)
  //
  // seg-faults due to Thrust bug
  // on binary-search-like algorithms
  // when run with stream dependent
  // execution policies; fixed on Thrust GitHub
  // hence replace w/ linear interpolation,
  // until the Thrust issue gets resolved:
  //
  // obsIndex = (thrust::lower_bound(
  //               thrust_exec_policy, thrust::device_pointer_cast(distsCumSum),
  //               thrust::device_pointer_cast(distsCumSum + n), distsSum * rand) -
  //             thrust::device_pointer_cast(distsCumSum));
  //
  // linear interpolation logic:
  //{
  value_type_t minSum{0};
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(&minSum, distsCumSum, sizeof(value_type_t), cudaMemcpyDeviceToHost, stream));
  RAFT_CHECK_CUDA(stream);

  if (distsSum > minSum) {
    value_type_t vIndex = static_cast<value_type_t>(n - 1);
    obsIndex = static_cast<index_type_t>(vIndex * (distsSum * rand - minSum) / (distsSum - minSum));
  } else {
    obsIndex = 0;
  }
  //}

  RAFT_CHECK_CUDA(stream);
  obsIndex = std::max(obsIndex, static_cast<index_type_t>(0));
  obsIndex = std::min(obsIndex, n - 1);

  // Record new centroid position
  RAFT_CUDA_TRY(cudaMemcpyAsync(centroid,
                                obs + IDX(0, obsIndex, d),
                                d * sizeof(value_type_t),
                                cudaMemcpyDeviceToDevice,
                                stream));

  return 0;
}

/**
 *  @brief Choose initial cluster centroids for k-means algorithm.
 *    Centroids are randomly chosen with k-means++ algorithm
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param n Number of observation vectors.
 *  @param d Dimension of observation vectors.
 *  @param k Number of clusters.
 *  @param obs (Input, device memory, d*n entries) Observation
 *    matrix. Matrix is stored column-major and each column is an
 *    observation vector. Matrix dimensions are d x n.
 *  @param centroids (Output, device memory, d*k entries) Centroid
 *    matrix. Matrix is stored column-major and each column is a
 *    centroid. Matrix dimensions are d x k.
 *  @param codes (Output, device memory, n entries) Cluster
 *    assignments.
 *  @param clusterSizes (Output, device memory, k entries) Number of
 *    points in each cluster.
 *  @param dists (Output, device memory, 2*n entries) Workspace. On
 *    exit, the first n entries give the square of the Euclidean
 *    distance between observation vectors and the closest centroid.
 *  @return Zero if successful. Otherwise non-zero.
 */
template <typename index_type_t, typename value_type_t>
static int initializeCentroids(raft::resources const& handle,
                               index_type_t n,
                               index_type_t d,
                               index_type_t k,
                               const value_type_t* __restrict__ obs,
                               value_type_t* __restrict__ centroids,
                               index_type_t* __restrict__ codes,
                               index_type_t* __restrict__ clusterSizes,
                               value_type_t* __restrict__ dists,
                               unsigned long long seed)
{
  // -------------------------------------------------------
  // Variable declarations
  // -------------------------------------------------------

  // Loop index
  index_type_t i;

  // Random number generator
  thrust::default_random_engine rng(seed);
  thrust::uniform_real_distribution<value_type_t> uniformDist(0, 1);

  auto stream             = resource::get_cuda_stream(handle);
  auto thrust_exec_policy = resource::get_thrust_policy(handle);

  constexpr unsigned grid_lower_bound{65535};

  // -------------------------------------------------------
  // Implementation
  // -------------------------------------------------------

  // Initialize grid dimensions
  dim3 blockDim_warp{WARP_SIZE, 1, BSIZE_DIV_WSIZE};

  // CUDA grid dimensions
  dim3 gridDim_warp{std::min(ceildiv<unsigned>(d, WARP_SIZE), grid_lower_bound),
                    1,
                    std::min(ceildiv<unsigned>(n, BSIZE_DIV_WSIZE), grid_lower_bound)};

  // CUDA grid dimensions
  dim3 gridDim_block{std::min(ceildiv<unsigned>(n, BLOCK_SIZE), grid_lower_bound), 1, 1};

  // Assign observation vectors to code 0
  RAFT_CUDA_TRY(cudaMemsetAsync(codes, 0, n * sizeof(index_type_t), stream));

  // Choose first centroid
  thrust::fill(thrust_exec_policy,
               thrust::device_pointer_cast(dists),
               thrust::device_pointer_cast(dists + n),
               1);
  RAFT_CHECK_CUDA(stream);
  if (chooseNewCentroid(handle, n, d, uniformDist(rng), obs, dists, centroids))
    WARNING("error in k-means++ (could not pick centroid)");

  // Compute distances from first centroid
  RAFT_CUDA_TRY(cudaMemsetAsync(dists, 0, n * sizeof(value_type_t), stream));
  computeDistances<<<gridDim_warp, blockDim_warp, 0, stream>>>(n, d, 1, obs, centroids, dists);
  RAFT_CHECK_CUDA(stream);

  // Choose remaining centroids
  for (i = 1; i < k; ++i) {
    // Choose ith centroid
    if (chooseNewCentroid(handle, n, d, uniformDist(rng), obs, dists, centroids + IDX(0, i, d)))
      WARNING("error in k-means++ (could not pick centroid)");

    // Compute distances from ith centroid
    RAFT_CUDA_TRY(cudaMemsetAsync(dists + n, 0, n * sizeof(value_type_t), stream));
    computeDistances<<<gridDim_warp, blockDim_warp, 0, stream>>>(
      n, d, 1, obs, centroids + IDX(0, i, d), dists + n);
    RAFT_CHECK_CUDA(stream);

    // Recompute minimum distances
    minDistances2<<<gridDim_block, BLOCK_SIZE, 0, stream>>>(n, dists, dists + n, codes, i);
    RAFT_CHECK_CUDA(stream);
  }

  // Compute cluster sizes
  RAFT_CUDA_TRY(cudaMemsetAsync(clusterSizes, 0, k * sizeof(index_type_t), stream));
  computeClusterSizes<<<gridDim_block, BLOCK_SIZE, 0, stream>>>(n, codes, clusterSizes);
  RAFT_CHECK_CUDA(stream);

  return 0;
}

/**
 *  @brief Find cluster centroids closest to observation vectors.
 *    Distance is measured with Euclidean norm.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param n Number of observation vectors.
 *  @param d Dimension of observation vectors.
 *  @param k Number of clusters.
 *  @param obs (Input, device memory, d*n entries) Observation
 *    matrix. Matrix is stored column-major and each column is an
 *    observation vector. Matrix dimensions are d x n.
 *  @param centroids (Input, device memory, d*k entries) Centroid
 *    matrix. Matrix is stored column-major and each column is a
 *    centroid. Matrix dimensions are d x k.
 *  @param dists (Output, device memory, n*k entries) Workspace. On
 *    exit, the first n entries give the square of the Euclidean
 *    distance between observation vectors and the closest centroid.
 *  @param codes (Output, device memory, n entries) Cluster
 *    assignments.
 *  @param clusterSizes (Output, device memory, k entries) Number of
 *    points in each cluster.
 *  @param residual_host (Output, host memory, 1 entry) Residual sum
 *    of squares of assignment.
 *  @return Zero if successful. Otherwise non-zero.
 */
template <typename index_type_t, typename value_type_t>
static int assignCentroids(raft::resources const& handle,
                           index_type_t n,
                           index_type_t d,
                           index_type_t k,
                           const value_type_t* __restrict__ obs,
                           const value_type_t* __restrict__ centroids,
                           value_type_t* __restrict__ dists,
                           index_type_t* __restrict__ codes,
                           index_type_t* __restrict__ clusterSizes,
                           value_type_t* residual_host)
{
  auto stream             = resource::get_cuda_stream(handle);
  auto thrust_exec_policy = resource::get_thrust_policy(handle);

  // Compute distance between centroids and observation vectors
  RAFT_CUDA_TRY(cudaMemsetAsync(dists, 0, n * k * sizeof(value_type_t), stream));

  // CUDA grid dimensions
  dim3 blockDim{WARP_SIZE, 1, BLOCK_SIZE / WARP_SIZE};

  dim3 gridDim;
  constexpr unsigned grid_lower_bound{65535};
  gridDim.x = std::min(ceildiv<unsigned>(d, WARP_SIZE), grid_lower_bound);
  gridDim.y = std::min(static_cast<unsigned>(k), grid_lower_bound);
  gridDim.z = std::min(ceildiv<unsigned>(n, BSIZE_DIV_WSIZE), grid_lower_bound);

  computeDistances<<<gridDim, blockDim, 0, stream>>>(n, d, k, obs, centroids, dists);
  RAFT_CHECK_CUDA(stream);

  // Find centroid closest to each observation vector
  RAFT_CUDA_TRY(cudaMemsetAsync(clusterSizes, 0, k * sizeof(index_type_t), stream));
  blockDim.x = BLOCK_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x  = std::min(ceildiv<unsigned>(n, BLOCK_SIZE), grid_lower_bound);
  gridDim.y  = 1;
  gridDim.z  = 1;
  minDistances<<<gridDim, blockDim, 0, stream>>>(n, k, dists, codes, clusterSizes);
  RAFT_CHECK_CUDA(stream);

  // Compute residual sum of squares
  *residual_host = thrust::reduce(
    thrust_exec_policy, thrust::device_pointer_cast(dists), thrust::device_pointer_cast(dists + n));

  return 0;
}

/**
 *  @brief Update cluster centroids for k-means algorithm.
 *    All clusters are assumed to be non-empty.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param n Number of observation vectors.
 *  @param d Dimension of observation vectors.
 *  @param k Number of clusters.
 *  @param obs (Input, device memory, d*n entries) Observation
 *    matrix. Matrix is stored column-major and each column is an
 *    observation vector. Matrix dimensions are d x n.
 *  @param codes (Input, device memory, n entries) Cluster
 *    assignments.
 *  @param clusterSizes (Input, device memory, k entries) Number of
 *    points in each cluster.
 *  @param centroids (Output, device memory, d*k entries) Centroid
 *    matrix. Matrix is stored column-major and each column is a
 *    centroid. Matrix dimensions are d x k.
 *  @param work (Output, device memory, n*d entries) Workspace.
 *  @param work_int (Output, device memory, 2*d*n entries)
 *    Workspace.
 *  @return Zero if successful. Otherwise non-zero.
 */
template <typename index_type_t, typename value_type_t>
static int updateCentroids(raft::resources const& handle,
                           index_type_t n,
                           index_type_t d,
                           index_type_t k,
                           const value_type_t* __restrict__ obs,
                           const index_type_t* __restrict__ codes,
                           const index_type_t* __restrict__ clusterSizes,
                           value_type_t* __restrict__ centroids,
                           value_type_t* __restrict__ work,
                           index_type_t* __restrict__ work_int)
{
  // -------------------------------------------------------
  // Variable declarations
  // -------------------------------------------------------

  // Useful constants
  const value_type_t one  = 1;
  const value_type_t zero = 0;

  constexpr unsigned grid_lower_bound{65535};

  auto stream             = resource::get_cuda_stream(handle);
  auto cublas_h           = resource::get_cublas_handle(handle);
  auto thrust_exec_policy = resource::get_thrust_policy(handle);

  // Device memory
  thrust::device_ptr<value_type_t> obs_copy(work);
  thrust::device_ptr<index_type_t> codes_copy(work_int);
  thrust::device_ptr<index_type_t> rows(work_int + d * n);

  // Take transpose of observation matrix
  // #TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgeam(cublas_h,
                                                   CUBLAS_OP_T,
                                                   CUBLAS_OP_N,
                                                   n,
                                                   d,
                                                   &one,
                                                   obs,
                                                   d,
                                                   &zero,
                                                   (value_type_t*)NULL,
                                                   n,
                                                   thrust::raw_pointer_cast(obs_copy),
                                                   n,
                                                   stream));

  // Cluster assigned to each observation matrix entry
  thrust::sequence(thrust_exec_policy, rows, rows + d * n);
  RAFT_CHECK_CUDA(stream);
  thrust::transform(thrust_exec_policy,
                    rows,
                    rows + d * n,
                    thrust::make_constant_iterator<index_type_t>(n),
                    rows,
                    thrust::modulus<index_type_t>());
  RAFT_CHECK_CUDA(stream);
  thrust::gather(
    thrust_exec_policy, rows, rows + d * n, thrust::device_pointer_cast(codes), codes_copy);
  RAFT_CHECK_CUDA(stream);

  // Row associated with each observation matrix entry
  thrust::sequence(thrust_exec_policy, rows, rows + d * n);
  RAFT_CHECK_CUDA(stream);
  thrust::transform(thrust_exec_policy,
                    rows,
                    rows + d * n,
                    thrust::make_constant_iterator<index_type_t>(n),
                    rows,
                    thrust::divides<index_type_t>());
  RAFT_CHECK_CUDA(stream);

  // Sort and reduce to add observation vectors in same cluster
  thrust::stable_sort_by_key(thrust_exec_policy,
                             codes_copy,
                             codes_copy + d * n,
                             make_zip_iterator(make_tuple(obs_copy, rows)));
  RAFT_CHECK_CUDA(stream);
  thrust::reduce_by_key(thrust_exec_policy,
                        rows,
                        rows + d * n,
                        obs_copy,
                        codes_copy,  // Output to codes_copy is ignored
                        thrust::device_pointer_cast(centroids));
  RAFT_CHECK_CUDA(stream);

  // Divide sums by cluster size to get centroid matrix
  //
  // CUDA grid dimensions
  dim3 blockDim{WARP_SIZE, BLOCK_SIZE / WARP_SIZE, 1};

  // CUDA grid dimensions
  dim3 gridDim{std::min(ceildiv<unsigned>(d, WARP_SIZE), grid_lower_bound),
               std::min(ceildiv<unsigned>(k, BSIZE_DIV_WSIZE), grid_lower_bound),
               1};

  divideCentroids<<<gridDim, blockDim, 0, stream>>>(d, k, clusterSizes, centroids);
  RAFT_CHECK_CUDA(stream);

  return 0;
}

// =========================================================
// k-means algorithm
// =========================================================

/**
 *  @brief Find clusters with k-means algorithm.
 *    Initial centroids are chosen with k-means++ algorithm. Empty
 *    clusters are reinitialized by choosing new centroids with
 *    k-means++ algorithm.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param n Number of observation vectors.
 *  @param d Dimension of observation vectors.
 *  @param k Number of clusters.
 *  @param tol Tolerance for convergence. k-means stops when the
 *    change in residual divided by n is less than tol.
 *  @param maxiter Maximum number of k-means iterations.
 *  @param obs (Input, device memory, d*n entries) Observation
 *    matrix. Matrix is stored column-major and each column is an
 *    observation vector. Matrix dimensions are d x n.
 *  @param codes (Output, device memory, n entries) Cluster
 *    assignments.
 *  @param clusterSizes (Output, device memory, k entries) Number of
 *    points in each cluster.
 *  @param centroids (Output, device memory, d*k entries) Centroid
 *    matrix. Matrix is stored column-major and each column is a
 *    centroid. Matrix dimensions are d x k.
 *  @param work (Output, device memory, n*max(k,d) entries)
 *    Workspace.
 *  @param work_int (Output, device memory, 2*d*n entries)
 *    Workspace.
 *  @param residual_host (Output, host memory, 1 entry) Residual sum
 *    of squares (sum of squares of distances between observation
 *    vectors and centroids).
 *  @param iters_host (Output, host memory, 1 entry) Number of
 *    k-means iterations.
 *  @param seed random seed to be used.
 *  @return error flag.
 */
template <typename index_type_t, typename value_type_t>
int kmeans(raft::resources const& handle,
           index_type_t n,
           index_type_t d,
           index_type_t k,
           value_type_t tol,
           index_type_t maxiter,
           const value_type_t* __restrict__ obs,
           index_type_t* __restrict__ codes,
           index_type_t* __restrict__ clusterSizes,
           value_type_t* __restrict__ centroids,
           value_type_t* __restrict__ work,
           index_type_t* __restrict__ work_int,
           value_type_t* residual_host,
           index_type_t* iters_host,
           unsigned long long seed)
{
  // -------------------------------------------------------
  // Variable declarations
  // -------------------------------------------------------

  // Current iteration
  index_type_t iter;

  constexpr unsigned grid_lower_bound{65535};

  // Residual sum of squares at previous iteration
  value_type_t residualPrev = 0;

  // Random number generator
  thrust::default_random_engine rng(seed);
  thrust::uniform_real_distribution<value_type_t> uniformDist(0, 1);

  // -------------------------------------------------------
  // Initialization
  // -------------------------------------------------------

  auto stream             = resource::get_cuda_stream(handle);
  auto cublas_h           = resource::get_cublas_handle(handle);
  auto thrust_exec_policy = resource::get_thrust_policy(handle);

  // Trivial cases
  if (k == 1) {
    RAFT_CUDA_TRY(cudaMemsetAsync(codes, 0, n * sizeof(index_type_t), stream));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(clusterSizes, &n, sizeof(index_type_t), cudaMemcpyHostToDevice, stream));
    if (updateCentroids(handle, n, d, k, obs, codes, clusterSizes, centroids, work, work_int))
      WARNING("could not compute k-means centroids");

    dim3 blockDim{WARP_SIZE, 1, BLOCK_SIZE / WARP_SIZE};

    dim3 gridDim{std::min(ceildiv<unsigned>(d, WARP_SIZE), grid_lower_bound),
                 1,
                 std::min(ceildiv<unsigned>(n, BLOCK_SIZE / WARP_SIZE), grid_lower_bound)};

    RAFT_CUDA_TRY(cudaMemsetAsync(work, 0, n * k * sizeof(value_type_t), stream));
    computeDistances<<<gridDim, blockDim, 0, stream>>>(n, d, 1, obs, centroids, work);
    RAFT_CHECK_CUDA(stream);
    *residual_host = thrust::reduce(
      thrust_exec_policy, thrust::device_pointer_cast(work), thrust::device_pointer_cast(work + n));
    RAFT_CHECK_CUDA(stream);
    return 0;
  }
  if (n <= k) {
    thrust::sequence(thrust_exec_policy,
                     thrust::device_pointer_cast(codes),
                     thrust::device_pointer_cast(codes + n));
    RAFT_CHECK_CUDA(stream);
    thrust::fill_n(thrust_exec_policy, thrust::device_pointer_cast(clusterSizes), n, 1);
    RAFT_CHECK_CUDA(stream);

    if (n < k)
      RAFT_CUDA_TRY(cudaMemsetAsync(clusterSizes + n, 0, (k - n) * sizeof(index_type_t), stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      centroids, obs, d * n * sizeof(value_type_t), cudaMemcpyDeviceToDevice, stream));
    *residual_host = 0;
    return 0;
  }

  // Initialize cuBLAS
  // #TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST, stream));

  // -------------------------------------------------------
  // k-means++ algorithm
  // -------------------------------------------------------

  // Choose initial cluster centroids
  if (initializeCentroids(handle, n, d, k, obs, centroids, codes, clusterSizes, work, seed))
    WARNING("could not initialize k-means centroids");

  // Apply k-means iteration until convergence
  for (iter = 0; iter < maxiter; ++iter) {
    // Update cluster centroids
    if (updateCentroids(handle, n, d, k, obs, codes, clusterSizes, centroids, work, work_int))
      WARNING("could not update k-means centroids");

    // Determine centroid closest to each observation
    residualPrev = *residual_host;
    if (assignCentroids(handle, n, d, k, obs, centroids, work, codes, clusterSizes, residual_host))
      WARNING("could not assign observation vectors to k-means clusters");

    // Reinitialize empty clusters with new centroids
    index_type_t emptyCentroid = (thrust::find(thrust_exec_policy,
                                               thrust::device_pointer_cast(clusterSizes),
                                               thrust::device_pointer_cast(clusterSizes + k),
                                               0) -
                                  thrust::device_pointer_cast(clusterSizes));

    // FIXME: emptyCentroid never reaches k (infinite loop) under certain
    // conditions, such as if obs is corrupt (as seen as a result of a
    // DataFrame column of NULL edge vals used to create the Graph)
    while (emptyCentroid < k) {
      if (chooseNewCentroid(
            handle, n, d, uniformDist(rng), obs, work, centroids + IDX(0, emptyCentroid, d)))
        WARNING("could not replace empty centroid");
      if (assignCentroids(
            handle, n, d, k, obs, centroids, work, codes, clusterSizes, residual_host))
        WARNING("could not assign observation vectors to k-means clusters");
      emptyCentroid = (thrust::find(thrust_exec_policy,
                                    thrust::device_pointer_cast(clusterSizes),
                                    thrust::device_pointer_cast(clusterSizes + k),
                                    0) -
                       thrust::device_pointer_cast(clusterSizes));
      RAFT_CHECK_CUDA(stream);
    }

    // Check for convergence
    if (std::fabs(residualPrev - (*residual_host)) / n < tol) {
      ++iter;
      break;
    }
  }

  // Warning if k-means has failed to converge
  if (std::fabs(residualPrev - (*residual_host)) / n >= tol) WARNING("k-means failed to converge");

  *iters_host = iter;
  return 0;
}

/**
 *  @brief Find clusters with k-means algorithm.
 *    Initial centroids are chosen with k-means++ algorithm. Empty
 *    clusters are reinitialized by choosing new centroids with
 *    k-means++ algorithm.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param n Number of observation vectors.
 *  @param d Dimension of observation vectors.
 *  @param k Number of clusters.
 *  @param tol Tolerance for convergence. k-means stops when the
 *    change in residual divided by n is less than tol.
 *  @param maxiter Maximum number of k-means iterations.
 *  @param obs (Input, device memory, d*n entries) Observation
 *    matrix. Matrix is stored column-major and each column is an
 *    observation vector. Matrix dimensions are d x n.
 *  @param codes (Output, device memory, n entries) Cluster
 *    assignments.
 *  @param residual On exit, residual sum of squares (sum of squares
 *    of distances between observation vectors and centroids).
 *  @param iters on exit, number of k-means iterations.
 *  @param seed random seed to be used.
 *  @return error flag
 */
template <typename index_type_t, typename value_type_t>
int kmeans(raft::resources const& handle,
           index_type_t n,
           index_type_t d,
           index_type_t k,
           value_type_t tol,
           index_type_t maxiter,
           const value_type_t* __restrict__ obs,
           index_type_t* __restrict__ codes,
           value_type_t& residual,
           index_type_t& iters,
           unsigned long long seed = 123456)
{
  // Check that parameters are valid
  RAFT_EXPECTS(n > 0, "invalid parameter (n<1)");
  RAFT_EXPECTS(d > 0, "invalid parameter (d<1)");
  RAFT_EXPECTS(k > 0, "invalid parameter (k<1)");
  RAFT_EXPECTS(tol > 0, "invalid parameter (tol<=0)");
  RAFT_EXPECTS(maxiter >= 0, "invalid parameter (maxiter<0)");

  // Allocate memory
  raft::spectral::matrix::vector_t<index_type_t> clusterSizes(handle, k);
  raft::spectral::matrix::vector_t<value_type_t> centroids(handle, d * k);
  raft::spectral::matrix::vector_t<value_type_t> work(handle, n * std::max(k, d));
  raft::spectral::matrix::vector_t<index_type_t> work_int(handle, 2 * d * n);

  // Perform k-means
  return kmeans<index_type_t, value_type_t>(handle,
                                            n,
                                            d,
                                            k,
                                            tol,
                                            maxiter,
                                            obs,
                                            codes,
                                            clusterSizes.raw(),
                                            centroids.raw(),
                                            work.raw(),
                                            work_int.raw(),
                                            &residual,
                                            &iters,
                                            seed);
}

}  // namespace detail
}  // namespace cluster
}  // namespace raft
