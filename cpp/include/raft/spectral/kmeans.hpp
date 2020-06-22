/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cmath>
#include <cstdio>
#include <ctime>

#include <cuda.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <raft/cudart_utils.h>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/handle.hpp>
#include <raft/spectral/matrix_wrappers.hpp>
#include <raft/spectral/sm_utils.hpp>
#include <raft/spectral/warn_dbg.hpp>

namespace {

using namespace raft;
using namespace raft::linalg;
// =========================================================
// Useful grid settings
// =========================================================

constexpr unsigned int BLOCK_SIZE = 1024;
constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int BSIZE_DIV_WSIZE = (BLOCK_SIZE / WARP_SIZE);

// =========================================================
// CUDA kernels
// =========================================================

/// Compute distances between observation vectors and centroids
/** Block dimensions should be (warpSize, 1,
 *  blockSize/warpSize). Ideally, the grid is large enough so there
 *  are d threads in the x-direction, k threads in the y-direction,
 *  and n threads in the z-direction.
 *
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
template <typename IndexType_, typename ValueType_>
static __global__ void computeDistances(
  IndexType_ n, IndexType_ d, IndexType_ k, const ValueType_* __restrict__ obs,
  const ValueType_* __restrict__ centroids, ValueType_* __restrict__ dists) {
  // Loop index
  IndexType_ i;

  // Block indices
  IndexType_ bidx;
  // Global indices
  IndexType_ gidx, gidy, gidz;

  // Private memory
  ValueType_ centroid_private, dist_private;

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
          dist_private += utils::shfl_down(dist_private, i, 2 * i);

        // Write result to global memory
        if (threadIdx.x == 0)
          utils::atomicFPAdd(dists + IDX(gidz, gidy, n), dist_private);

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

/// Find closest centroid to observation vectors
/** Block and grid dimensions should be 1-dimensional. Ideally the
 *  grid is large enough so there are n threads.
 *
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
template <typename IndexType_, typename ValueType_>
static __global__ void minDistances(IndexType_ n, IndexType_ k,
                                    ValueType_* __restrict__ dists,
                                    IndexType_* __restrict__ codes,
                                    IndexType_* __restrict__ clusterSizes) {
  // Loop index
  IndexType_ i, j;

  // Current matrix entry
  ValueType_ dist_curr;

  // Smallest entry in row
  ValueType_ dist_min;
  IndexType_ code_min;

  // Each row in observation matrix is processed by a thread
  i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    // Find minimum entry in row
    code_min = 0;
    dist_min = dists[IDX(i, 0, n)];
    for (j = 1; j < k; ++j) {
      dist_curr = dists[IDX(i, j, n)];
      code_min = (dist_curr < dist_min) ? j : code_min;
      dist_min = (dist_curr < dist_min) ? dist_curr : dist_min;
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

/// Check if newly computed distances are smaller than old distances
/** Block and grid dimensions should be 1-dimensional. Ideally the
 *  grid is large enough so there are n threads.
 *
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
template <typename IndexType_, typename ValueType_>
static __global__ void minDistances2(IndexType_ n,
                                     ValueType_* __restrict__ dists_old,
                                     const ValueType_* __restrict__ dists_new,
                                     IndexType_* __restrict__ codes_old,
                                     IndexType_ code_new) {
  // Loop index
  IndexType_ i;

  // Distances
  ValueType_ dist_old_private;
  ValueType_ dist_new_private;

  // Each row is processed by a thread
  i = threadIdx.x + blockIdx.x * blockDim.x;
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

/// Compute size of k-means clusters
/** Block and grid dimensions should be 1-dimensional. Ideally the
 *  grid is large enough so there are n threads.
 *
 *  @param n Number of observation vectors.
 *  @param k Number of clusters.
 *  @param codes (Input, n entries) Cluster assignments.
 *  @param clusterSizes (Output, k entries) Number of points in each
 *    cluster. Entries must be initialized to zero.
 */
template <typename IndexType_>
static __global__ void computeClusterSizes(
  IndexType_ n, IndexType_ k, const IndexType_* __restrict__ codes,
  IndexType_* __restrict__ clusterSizes) {
  IndexType_ i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    atomicAdd(clusterSizes + codes[i], 1);
    i += blockDim.x * gridDim.x;
  }
}

/// Divide rows of centroid matrix by cluster sizes
/** Divides the ith column of the sum matrix by the size of the ith
 *  cluster. If the sum matrix has been initialized so that the ith
 *  row is the sum of all observation vectors in the ith cluster,
 *  this kernel produces cluster centroids. The grid and block
 *  dimensions should be 2-dimensional. Ideally the grid is large
 *  enough so there are d threads in the x-direction and k threads
 *  in the y-direction.
 *
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
template <typename IndexType_, typename ValueType_>
static __global__ void divideCentroids(
  IndexType_ d, IndexType_ k, const IndexType_* __restrict__ clusterSizes,
  ValueType_* __restrict__ centroids) {
  // Global indices
  IndexType_ gidx, gidy;

  // Current cluster size
  IndexType_ clusterSize_private;

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

/// Randomly choose new centroids
/** Centroid is randomly chosen with k-means++ algorithm.
 *
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
template <typename IndexType_, typename ValueType_, typename ThrustExePolicy>
static int chooseNewCentroid(handle_t const& handle,
                             ThrustExePolicy thrust_exec_policy, IndexType_ n,
                             IndexType_ d, IndexType_ k, ValueType_ rand,
                             const ValueType_* __restrict__ obs,
                             ValueType_* __restrict__ dists,
                             ValueType_* __restrict__ centroid) {
  // Cumulative sum of distances
  ValueType_* distsCumSum = dists + n;
  // Residual sum of squares
  ValueType_ distsSum{0};
  // Observation vector that is chosen as new centroid
  IndexType_ obsIndex;

  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  // Compute cumulative sum of distances
  thrust::inclusive_scan(thrust_exec_policy, thrust::device_pointer_cast(dists),
                         thrust::device_pointer_cast(dists + n),
                         thrust::device_pointer_cast(distsCumSum));
  CHECK_CUDA(stream);
  CUDA_TRY(cudaMemcpy(&distsSum, distsCumSum + n - 1, sizeof(ValueType_),
                      cudaMemcpyDeviceToHost));

  // Randomly choose observation vector
  //   Probabilities are proportional to square of distance to closest
  //   centroid (see k-means++ algorithm)
  obsIndex = (thrust::lower_bound(
                thrust_exec_policy, thrust::device_pointer_cast(distsCumSum),
                thrust::device_pointer_cast(distsCumSum + n), distsSum * rand) -
              thrust::device_pointer_cast(distsCumSum));
  CHECK_CUDA(stream);
  obsIndex = max(obsIndex, 0);
  obsIndex = min(obsIndex, n - 1);

  // Record new centroid position
  CUDA_TRY(cudaMemcpyAsync(centroid, obs + IDX(0, obsIndex, d),
                           d * sizeof(ValueType_), cudaMemcpyDeviceToDevice,
                           stream));

  return 0;
}

/// Choose initial cluster centroids for k-means algorithm
/** Centroids are randomly chosen with k-means++ algorithm
 *
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
template <typename IndexType_, typename ValueType_, typename ThrustExePolicy>
static int initializeCentroids(
  handle_t const& handle, ThrustExePolicy thrust_exec_policy, IndexType_ n,
  IndexType_ d, IndexType_ k, const ValueType_* __restrict__ obs,
  ValueType_* __restrict__ centroids, IndexType_* __restrict__ codes,
  IndexType_* __restrict__ clusterSizes, ValueType_* __restrict__ dists,
  unsigned long long seed) {
  // -------------------------------------------------------
  // Variable declarations
  // -------------------------------------------------------

  // Loop index
  IndexType_ i;

  // CUDA grid dimensions
  dim3 blockDim_warp, gridDim_warp, gridDim_block;

  // Random number generator
  thrust::default_random_engine rng(seed);
  thrust::uniform_real_distribution<ValueType_> uniformDist(0, 1);

  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  // -------------------------------------------------------
  // Implementation
  // -------------------------------------------------------

  // Initialize grid dimensions
  blockDim_warp.x = WARP_SIZE;
  blockDim_warp.y = 1;
  blockDim_warp.z = BSIZE_DIV_WSIZE;
  gridDim_warp.x = min((d + WARP_SIZE - 1) / WARP_SIZE, 65535);
  gridDim_warp.y = 1;
  gridDim_warp.z = min((n + BSIZE_DIV_WSIZE - 1) / BSIZE_DIV_WSIZE, 65535);
  gridDim_block.x = min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);
  gridDim_block.y = 1;
  gridDim_block.z = 1;

  // Assign observation vectors to code 0
  CUDA_TRY(cudaMemsetAsync(codes, 0, n * sizeof(IndexType_), stream));

  // Choose first centroid
  thrust::fill(thrust_exec_policy, thrust::device_pointer_cast(dists),
               thrust::device_pointer_cast(dists + n), 1);
  CHECK_CUDA(stream);
  if (chooseNewCentroid(handle, thrust_exec_policy, n, d, k, uniformDist(rng),
                        obs, dists, centroids))
    WARNING("error in k-means++ (could not pick centroid)");

  // Compute distances from first centroid
  CUDA_TRY(cudaMemsetAsync(dists, 0, n * sizeof(ValueType_), stream));
  computeDistances<<<gridDim_warp, blockDim_warp, 0, stream>>>(
    n, d, 1, obs, centroids, dists);
  CHECK_CUDA(stream);

  // Choose remaining centroids
  for (i = 1; i < k; ++i) {
    // Choose ith centroid
    if (chooseNewCentroid(handle, thrust_exec_policy, n, d, k, uniformDist(rng),
                          obs, dists, centroids + IDX(0, i, d)))
      WARNING("error in k-means++ (could not pick centroid)");

    // Compute distances from ith centroid
    CUDA_TRY(cudaMemsetAsync(dists + n, 0, n * sizeof(ValueType_), stream));
    computeDistances<<<gridDim_warp, blockDim_warp, 0, stream>>>(
      n, d, 1, obs, centroids + IDX(0, i, d), dists + n);
    CHECK_CUDA(stream);

    // Recompute minimum distances
    minDistances2<<<gridDim_block, BLOCK_SIZE, 0, stream>>>(n, dists, dists + n,
                                                            codes, i);
    CHECK_CUDA(stream);
  }

  // Compute cluster sizes
  CUDA_TRY(cudaMemsetAsync(clusterSizes, 0, k * sizeof(IndexType_), stream));
  computeClusterSizes<<<gridDim_block, BLOCK_SIZE, 0, stream>>>(n, k, codes,
                                                                clusterSizes);
  CHECK_CUDA(stream);

  return 0;
}

/// Find cluster centroids closest to observation vectors
/** Distance is measured with Euclidean norm.
 *
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
template <typename IndexType_, typename ValueType_, typename ThrustExePolicy>
static int assignCentroids(
  handle_t const& handle, ThrustExePolicy thrust_exec_policy, IndexType_ n,
  IndexType_ d, IndexType_ k, const ValueType_* __restrict__ obs,
  const ValueType_* __restrict__ centroids, ValueType_* __restrict__ dists,
  IndexType_* __restrict__ codes, IndexType_* __restrict__ clusterSizes,
  ValueType_* residual_host) {
  // CUDA grid dimensions
  dim3 blockDim, gridDim;

  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  // Compute distance between centroids and observation vectors
  CUDA_TRY(cudaMemsetAsync(dists, 0, n * k * sizeof(ValueType_), stream));
  blockDim.x = WARP_SIZE;
  blockDim.y = 1;
  blockDim.z = BLOCK_SIZE / WARP_SIZE;
  gridDim.x = min((d + WARP_SIZE - 1) / WARP_SIZE, 65535);
  gridDim.y = min(k, 65535);
  gridDim.z = min((n + BSIZE_DIV_WSIZE - 1) / BSIZE_DIV_WSIZE, 65535);
  computeDistances<<<gridDim, blockDim, 0, stream>>>(n, d, k, obs, centroids,
                                                     dists);
  CHECK_CUDA(stream);

  // Find centroid closest to each observation vector
  CUDA_TRY(cudaMemsetAsync(clusterSizes, 0, k * sizeof(IndexType_), stream));
  blockDim.x = BLOCK_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);
  gridDim.y = 1;
  gridDim.z = 1;
  minDistances<<<gridDim, blockDim, 0, stream>>>(n, k, dists, codes,
                                                 clusterSizes);
  CHECK_CUDA(stream);

  // Compute residual sum of squares
  *residual_host =
    thrust::reduce(thrust_exec_policy, thrust::device_pointer_cast(dists),
                   thrust::device_pointer_cast(dists + n));

  return 0;
}

/// Update cluster centroids for k-means algorithm
/** All clusters are assumed to be non-empty.
 *
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
template <typename IndexType_, typename ValueType_, typename ThrustExePolicy>
static int updateCentroids(handle_t const& handle,
                           ThrustExePolicy thrust_exec_policy, IndexType_ n,
                           IndexType_ d, IndexType_ k,
                           const ValueType_* __restrict__ obs,
                           const IndexType_* __restrict__ codes,
                           const IndexType_* __restrict__ clusterSizes,
                           ValueType_* __restrict__ centroids,
                           ValueType_* __restrict__ work,
                           IndexType_* __restrict__ work_int) {
  // -------------------------------------------------------
  // Variable declarations
  // -------------------------------------------------------

  // Useful constants
  const ValueType_ one = 1;
  const ValueType_ zero = 0;

  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  // CUDA grid dimensions
  dim3 blockDim, gridDim;

  // Device memory
  thrust::device_ptr<ValueType_> obs_copy(work);
  thrust::device_ptr<IndexType_> codes_copy(work_int);
  thrust::device_ptr<IndexType_> rows(work_int + d * n);

  // Take transpose of observation matrix
  CUBLAS_CHECK(cublasgeam(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, n, d, &one, obs,
                          d, &zero, (ValueType_*)NULL, n,
                          thrust::raw_pointer_cast(obs_copy), n, stream));

  // Cluster assigned to each observation matrix entry
  thrust::sequence(thrust_exec_policy, rows, rows + d * n);
  CHECK_CUDA(stream);
  thrust::transform(thrust_exec_policy, rows, rows + d * n,
                    thrust::make_constant_iterator<IndexType_>(n), rows,
                    thrust::modulus<IndexType_>());
  CHECK_CUDA(stream);
  thrust::gather(thrust_exec_policy, rows, rows + d * n,
                 thrust::device_pointer_cast(codes), codes_copy);
  CHECK_CUDA(stream);

  // Row associated with each observation matrix entry
  thrust::sequence(thrust_exec_policy, rows, rows + d * n);
  CHECK_CUDA(stream);
  thrust::transform(thrust_exec_policy, rows, rows + d * n,
                    thrust::make_constant_iterator<IndexType_>(n), rows,
                    thrust::divides<IndexType_>());
  CHECK_CUDA(stream);

  // Sort and reduce to add observation vectors in same cluster
  thrust::stable_sort_by_key(thrust_exec_policy, codes_copy, codes_copy + d * n,
                             make_zip_iterator(make_tuple(obs_copy, rows)));
  CHECK_CUDA(stream);
  thrust::reduce_by_key(thrust_exec_policy, rows, rows + d * n, obs_copy,
                        codes_copy,  // Output to codes_copy is ignored
                        thrust::device_pointer_cast(centroids));
  CHECK_CUDA(stream);

  // Divide sums by cluster size to get centroid matrix
  blockDim.x = WARP_SIZE;
  blockDim.y = BLOCK_SIZE / WARP_SIZE;
  blockDim.z = 1;
  gridDim.x = min((d + WARP_SIZE - 1) / WARP_SIZE, 65535);
  gridDim.y = min((k + BSIZE_DIV_WSIZE - 1) / BSIZE_DIV_WSIZE, 65535);
  gridDim.z = 1;
  divideCentroids<<<gridDim, blockDim, 0, stream>>>(d, k, clusterSizes,
                                                    centroids);
  CHECK_CUDA(stream);

  return 0;
}

}  // namespace

namespace raft {

// =========================================================
// k-means algorithm
// =========================================================

/// Find clusters with k-means algorithm
/** Initial centroids are chosen with k-means++ algorithm. Empty
 *  clusters are reinitialized by choosing new centroids with
 *  k-means++ algorithm.
 *
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
 *  @return error flag.
 */
template <typename IndexType_, typename ValueType_, typename ThrustExePolicy>
int kmeans(handle_t const& handle, ThrustExePolicy thrust_exec_policy,
           IndexType_ n, IndexType_ d, IndexType_ k, ValueType_ tol,
           IndexType_ maxiter, const ValueType_* __restrict__ obs,
           IndexType_* __restrict__ codes,
           IndexType_* __restrict__ clusterSizes,
           ValueType_* __restrict__ centroids, ValueType_* __restrict__ work,
           IndexType_* __restrict__ work_int, ValueType_* residual_host,
           IndexType_* iters_host, unsigned long long seed) {
  // -------------------------------------------------------
  // Variable declarations
  // -------------------------------------------------------

  // Current iteration
  IndexType_ iter;

  // Residual sum of squares at previous iteration
  ValueType_ residualPrev = 0;

  // Random number generator
  thrust::default_random_engine rng(seed);
  thrust::uniform_real_distribution<ValueType_> uniformDist(0, 1);

  // -------------------------------------------------------
  // Initialization
  // -------------------------------------------------------

  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  // Trivial cases
  if (k == 1) {
    CUDA_TRY(cudaMemsetAsync(codes, 0, n * sizeof(IndexType_), stream));
    CUDA_TRY(cudaMemcpyAsync(clusterSizes, &n, sizeof(IndexType_),
                             cudaMemcpyHostToDevice, stream));
    if (updateCentroids(handle, thrust_exec_policy, n, d, k, obs, codes,
                        clusterSizes, centroids, work, work_int))
      WARNING("could not compute k-means centroids");
    dim3 blockDim, gridDim;
    blockDim.x = WARP_SIZE;
    blockDim.y = 1;
    blockDim.z = BLOCK_SIZE / WARP_SIZE;
    gridDim.x = min((d + WARP_SIZE - 1) / WARP_SIZE, 65535);
    gridDim.y = 1;
    gridDim.z =
      min((n + BLOCK_SIZE / WARP_SIZE - 1) / (BLOCK_SIZE / WARP_SIZE), 65535);
    CUDA_TRY(cudaMemsetAsync(work, 0, n * k * sizeof(ValueType_), stream));
    computeDistances<<<gridDim, blockDim, 0, stream>>>(n, d, 1, obs, centroids,
                                                       work);
    CHECK_CUDA(stream);
    *residual_host =
      thrust::reduce(thrust_exec_policy, thrust::device_pointer_cast(work),
                     thrust::device_pointer_cast(work + n));
    CHECK_CUDA(stream);
    return 0;
  }
  if (n <= k) {
    thrust::sequence(thrust_exec_policy, thrust::device_pointer_cast(codes),
                     thrust::device_pointer_cast(codes + n));
    CHECK_CUDA(stream);
    thrust::fill_n(thrust_exec_policy,
                   thrust::device_pointer_cast(clusterSizes), n, 1);
    CHECK_CUDA(stream);

    if (n < k)
      CUDA_TRY(cudaMemsetAsync(clusterSizes + n, 0,
                               (k - n) * sizeof(IndexType_), stream));
    CUDA_TRY(cudaMemcpyAsync(centroids, obs, d * n * sizeof(ValueType_),
                             cudaMemcpyDeviceToDevice, stream));
    *residual_host = 0;
    return 0;
  }

  // Initialize cuBLAS
  CUBLAS_CHECK(
    linalg::cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST,
                                 stream));  // ????? TODO: check / remove

  // -------------------------------------------------------
  // k-means++ algorithm
  // -------------------------------------------------------

  // Choose initial cluster centroids
  if (initializeCentroids(handle, thrust_exec_policy, n, d, k, obs, centroids,
                          codes, clusterSizes, work, seed))
    WARNING("could not initialize k-means centroids");

  // Apply k-means iteration until convergence
  for (iter = 0; iter < maxiter; ++iter) {
    // Update cluster centroids
    if (updateCentroids(handle, thrust_exec_policy, n, d, k, obs, codes,
                        clusterSizes, centroids, work, work_int))
      WARNING("could not update k-means centroids");

    // Determine centroid closest to each observation
    residualPrev = *residual_host;
    if (assignCentroids(handle, thrust_exec_policy, n, d, k, obs, centroids,
                        work, codes, clusterSizes, residual_host))
      WARNING("could not assign observation vectors to k-means clusters");

    // Reinitialize empty clusters with new centroids
    IndexType_ emptyCentroid =
      (thrust::find(thrust_exec_policy,
                    thrust::device_pointer_cast(clusterSizes),
                    thrust::device_pointer_cast(clusterSizes + k), 0) -
       thrust::device_pointer_cast(clusterSizes));

    // FIXME: emptyCentroid never reaches k (infinite loop) under certain
    // conditions, such as if obs is corrupt (as seen as a result of a
    // DataFrame column of NULL edge vals used to create the Graph)
    while (emptyCentroid < k) {
      if (chooseNewCentroid(handle, thrust_exec_policy, n, d, k,
                            uniformDist(rng), obs, work,
                            centroids + IDX(0, emptyCentroid, d)))
        WARNING("could not replace empty centroid");
      if (assignCentroids(handle, thrust_exec_policy, n, d, k, obs, centroids,
                          work, codes, clusterSizes, residual_host))
        WARNING("could not assign observation vectors to k-means clusters");
      emptyCentroid =
        (thrust::find(thrust_exec_policy,
                      thrust::device_pointer_cast(clusterSizes),
                      thrust::device_pointer_cast(clusterSizes + k), 0) -
         thrust::device_pointer_cast(clusterSizes));
      CHECK_CUDA(stream);
    }

    // Check for convergence
    if (std::fabs(residualPrev - (*residual_host)) / n < tol) {
      ++iter;
      break;
    }
  }

  // Warning if k-means has failed to converge
  if (std::fabs(residualPrev - (*residual_host)) / n >= tol)
    WARNING("k-means failed to converge");

  *iters_host = iter;
  return 0;
}

/// Find clusters with k-means algorithm
/** Initial centroids are chosen with k-means++ algorithm. Empty
 *  clusters are reinitialized by choosing new centroids with
 *  k-means++ algorithm.
 *
 *  CNMEM must be initialized before calling this function.
 *
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
 *  @param On exit, number of k-means iterations.
 *  @return error flag
 */
template <typename IndexType_, typename ValueType_, typename ThrustExePolicy>
int kmeans(handle_t const& handle, ThrustExePolicy thrust_exec_policy,
           IndexType_ n, IndexType_ d, IndexType_ k, ValueType_ tol,
           IndexType_ maxiter, const ValueType_* __restrict__ obs,
           IndexType_* __restrict__ codes, ValueType_& residual,
           IndexType_& iters, unsigned long long seed = 123456) {
  using namespace matrix;

  // Check that parameters are valid
  RAFT_EXPECTS(n > 0, "invalid parameter (n<1)");
  RAFT_EXPECTS(d > 0, "invalid parameter (d<1)");
  RAFT_EXPECTS(k > 0, "invalid parameter (k<1)");
  RAFT_EXPECTS(tol > 0, "invalid parameter (tol<=0)");
  RAFT_EXPECTS(maxiter >= 0, "invalid parameter (maxiter<0)");

  // Allocate memory
  vector_t<IndexType_> clusterSizes(handle, k);
  vector_t<ValueType_> centroids(handle, d * k);
  vector_t<ValueType_> work(handle, n * max(k, d));
  vector_t<IndexType_> work_int(handle, 2 * d * n);

  // Perform k-means
  return kmeans<IndexType_, ValueType_>(
    handle, thrust_exec_policy, n, d, k, tol, maxiter, obs, codes,
    clusterSizes.raw(), centroids.raw(), work.raw(), work_int.raw(), &residual,
    &iters, seed);
}

}  // namespace raft
