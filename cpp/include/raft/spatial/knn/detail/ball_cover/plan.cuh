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

#include <cub/cub.cuh>

#include "../block_select_faiss.cuh"
#include "../knn_brute_force_faiss.cuh"
#include "../selection_faiss.cuh"

#include <limits.h>

#include <raft/cuda_utils.cuh>

#include <faiss/utils/Heap.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>

#include <raft/distance/distance.cuh>
#include <raft/selection/col_wise_sort.cuh>
#include <raft/sparse/coo.cuh>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

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
template <typename value_idx, typename value_t, typename value_int = uint32_t>
void k_closest_landmarks2(const raft::handle_t &handle,
                          BallCoverIndex<value_idx, value_t, value_int> &index,
                          const value_t *query_pts, value_int n_query_pts,
                          value_int k, value_idx *R_knn_inds,
                          value_t *R_knn_dists) {
  std::vector<value_t *> input = {index.get_R()};
  std::vector<value_int> sizes = {index.n_landmarks};

  brute_force_knn_impl<value_int, int64_t>(
    input, sizes, index.n, const_cast<value_t *>(query_pts), n_query_pts,
    R_knn_inds, R_knn_dists, k, handle.get_stream(), nullptr, 0, true, true,
    nullptr, index.metric);
}

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
template <typename value_idx, typename value_t, typename value_int = uint32_t,
          int tpb = 32>
__global__ void prune_landmarks(
  const value_t *landmark_dists, const value_int n_cols,
  const value_idx *R_knn_inds, const value_t *R_knn_dists,
  const value_t *R_radius, const value_t *landmarks, const value_t *knn_dists,
  const value_idx *landmarks_indptr, const int n_landmarks,
  const int bitset_size, const int k, const int batch_size,
  uint32_t *output_bitset, value_int *total_landmark_points,
  float weight = 1.0) {
  // allocate array of size n_landmarks / 32 ints
  extern __shared__ uint32_t sh_mem[];

  // Start with all bits on
#pragma unroll
  for (int i = threadIdx.x; i < bitset_size; i += tpb) sh_mem[i] = 0xffffffff;

  __syncthreads();

  value_t closest_R_dist = R_knn_dists[blockIdx.x * k + (k - 1)];

  value_idx n_points = 0;

  // Discard any landmarks where p(q, r) > p(q, r_q) + radius(r)
  // That is, the distance between the current point and the current
  // landmark is > the distance between the current point and
  // its closest landmark + the radius of the current landmark.
#pragma unroll
  for (int l = threadIdx.x; l < n_landmarks; l += tpb) {
    value_idx cardinality = landmarks_indptr[l + 1] - landmarks_indptr[l];
    value_t p_q_r = landmark_dists[blockIdx.x * n_landmarks + threadIdx.x];

    if (p_q_r > weight * (closest_R_dist + R_radius[l]) ||
        p_q_r > 3 * closest_R_dist ||
        knn_dists[blockIdx.x * k + (k - 1)] < p_q_r - R_radius[l]) {
      _zero_bit(sh_mem, l);
    } else {
      n_points += ceil(cardinality / static_cast<value_t>(batch_size));
    }
  }

  __syncthreads();

  /**
    * Output bitset
    */
#pragma unroll
  for (int l = threadIdx.x; l < bitset_size; l += tpb) {
    output_bitset[blockIdx.x * bitset_size + l] = sh_mem[l];
  }
  atomicAdd(total_landmark_points + blockIdx.x, n_points);
}

template <typename value_idx, typename value_int = uint32_t>
__global__ void write_plan_coo(
  const value_idx *landmark_indptr, const value_idx *coo_write_plan,
  const uint32_t *bitset, const value_int bitset_size,
  const value_int n_landmarks, const int batch_size,
  const value_int n_query_pts, value_idx *plan_query_ids_coo,
  value_idx *plan_landmark_ids_coo, value_idx *plan_offset_ids_coo) {
  int query_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (query_id >= n_query_pts) return;

  int cur_plan_offset = coo_write_plan[query_id];

  for (value_int cur_R_ind = 0; cur_R_ind < n_landmarks; cur_R_ind++) {
    // if cur R overlaps cur point's closest R, it could be a
    // candidate
    if (_get_val(bitset + (query_id * bitset_size), cur_R_ind)) {
      value_idx start_offset = landmark_indptr[cur_R_ind];
      value_idx stop_offset = landmark_indptr[cur_R_ind + 1];
      for (value_int batch_offset = start_offset; batch_offset < stop_offset;
           batch_offset += batch_size) {
        plan_query_ids_coo[cur_plan_offset] = query_id;
        plan_landmark_ids_coo[cur_plan_offset] = cur_R_ind;
        plan_offset_ids_coo[cur_plan_offset] = batch_offset;
        ++cur_plan_offset;
      }
    }
  }
}

template <typename value_idx, typename value_t, typename value_int = uint32_t>
void landmark_q_pw_dists(const raft::handle_t &handle,
                         BallCoverIndex<value_idx, value_t> &index,
                         const value_t *queries, value_int n_queries,
                         value_t *out_dists) {
  rmm::device_uvector<char> pw_workspace(0, handle.get_stream());

  // Compute pairwise dists between queries and landmarks.
  raft::distance::pairwise_distance(
    queries, index.get_R(), out_dists, n_queries, index.get_n_landmarks(),
    index.n, pw_workspace, index.get_metric(), handle.get_stream());
}

/**
 * Construct a "plan" for executing brute-force knn with random access loads
 * of index points. This plan constructs a edge list in COOrdinate format
 * where each tuple maps (query_id, landmark_id, batch_offset). Each
 * batch_offset is the start offset for the batch in the ball cover index 1nn.
 *
 * This plan allows distance computations to be spread more uniformly across
 * compute resources, increasing parallelism and lowering the potential for
 * stragglers. The plan guarantees  block will need to compute greater than
 * `batch_size` number of distances, though the actual number of distances
 * computed can be much smaller depending on additional pruning and
 * small landmarks.
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam value_int
 * @param handle
 * @param index
 * @param k
 * @param query
 * @param n_query_pts
 * @param knn_inds
 * @param knn_dists
 * @param weight
 */
template <typename value_idx, typename value_t, typename value_int = uint32_t,
          int batch_size = 2048>
void compute_plan(const raft::handle_t &handle,
                  BallCoverIndex<value_idx, value_t, value_int> &index,
                  value_int k, const value_t *query,
                  const value_int n_query_pts, const value_idx *knn_inds,
                  const value_t *knn_dists,
                  raft::sparse::COO<value_idx, value_idx> &plan_coo,
                  float weight = 1.0) {
  auto exec_policy = rmm::exec_policy(handle.get_stream());
  /**
     * Query Plan is a COO matrix mapping (query_point_id, landmark_id, landmark_index_start_offset)
     * for each query point. This is meant to be done in batches over a pairwise distance matrix
     * between the query points and landmarks and increase uniformity of distance computations.
     *
     * The query plan edge list is guaranteed to be ordered such that query_point_id is
     * increasing monotonically and within each query_id, the landmark_id should be ordered
     * by its distance from the query_id.
     *
     * Steps (this can be done in batches both by query points and landmarks):
     * 1. Compute pairwise_distances(query_points, landmarks)
     * 2. K-select to get radius for each query point
     * 3. Apply triangle inequality and bounds checking
     *      a) Compute cardinalities for each landmark set
     *      b) Compute n_batches
     *
     * 4. Create coo w/ nnz = n_batches.sum()
     * 5. Populate coo w/ batch information- rows=query_row_id, cols=landmark_id, vals=start_offset
     */
  rmm::device_uvector<value_t> ql_dists(n_query_pts * index.n_landmarks,
                                        handle.get_stream());

  // Compute pw dists between queries and landmarks
  landmark_q_pw_dists(handle, index, query, n_query_pts, ql_dists.data());

  // K-select to get radius bounds for each query point
  rmm::device_uvector<value_idx> R_knn_inds(k * index.m, handle.get_stream());
  rmm::device_uvector<value_t> R_knn_dists(k * index.m, handle.get_stream());

  k_closest_landmarks2(handle, index, query, n_query_pts, k, R_knn_inds.data(),
                       R_knn_dists.data());

  raft::print_device_vector("R_knn_inds", R_knn_inds.data(), 10, std::cout);

  // Compute filtered balls for current batch based on k found so far
  const value_int bitset_size = ceil(index.get_n_landmarks() / 32.0);
  rmm::device_uvector<uint32_t> bitset(bitset_size * index.m,
                                       handle.get_stream());

  rmm::device_uvector<value_int> landmark_batches(n_query_pts,
                                                  handle.get_stream());

  thrust::fill(handle.get_thrust_policy(), landmark_batches.data(),
               landmark_batches.data() + n_query_pts, 0);

  prune_landmarks<value_idx, value_t, value_int, 128>
    <<<n_query_pts, 128, bitset_size * sizeof(value_int),
       handle.get_stream()>>>(
      ql_dists.data(), index.n, R_knn_inds.data(), R_knn_dists.data(),
      index.get_R_radius(), index.get_R(), knn_dists, index.get_R_indptr(),
      index.get_n_landmarks(), bitset_size, k, batch_size, bitset.data(),
      landmark_batches.data(), weight);

  raft::print_device_vector("landmark_batches", landmark_batches.data(), 50,
                            std::cout);

  // Sum of cardinality array is nnz of plan
  value_int n_batches =
    thrust::reduce(exec_policy, landmark_batches.data(),
                   landmark_batches.data() + n_query_pts, 0);

  printf("n_batches=%d\n", n_batches);

  raft::print_device_vector("landmark_batches", landmark_batches.data(), 10,
                            std::cout);

  rmm::device_uvector<value_idx> coo_write_plan(n_query_pts,
                                                handle.get_stream());

  thrust::exclusive_scan(exec_policy, landmark_batches.data(),
                         landmark_batches.data() + n_query_pts,
                         coo_write_plan.data(), 0);

  // Construct COO where nnz=n_batches
  plan_coo.allocate(n_batches, 0, handle.get_stream());

  write_plan_coo<<<raft::ceildiv(n_query_pts, (value_int)256), 256, 0,
                   handle.get_stream()>>>(
    index.get_R_indptr(), coo_write_plan.data(), bitset.data(), bitset_size,
    index.get_n_landmarks(), batch_size, n_query_pts, plan_coo.rows(),
    plan_coo.cols(), plan_coo.vals());

  raft::print_device_vector("coo_write_plan", coo_write_plan.data(), 10,
                            std::cout);

  // TODO: Sort the COO plan so that the landmark_ids within each query_id are ordered by
  // their distances.
}

template <typename value_idx, typename value_t, int warp_q, int thread_q,
          int tpb>
__device__ void topk_merge(value_t *sh_memK, value_idx *sh_memV,
                           value_idx query_id, value_idx *batch_inds,
                           value_t *batch_dists, int batch_size, int k,
                           int *mutex, value_idx *knn_inds,
                           value_t *knn_dists) {
  faiss::gpu::BlockSelect<value_t, value_idx, false,
                          faiss::gpu::Comparator<value_t>, warp_q, thread_q,
                          tpb>
    heap(faiss::gpu::Limits<value_t>::getMax(),
         faiss::gpu::Limits<value_t>::getMax(), sh_memK, sh_memV, k);

  /**
   * First add batch
   */
  const int n_b =
    faiss::gpu::utils::roundDown(batch_size, faiss::gpu::kWarpSize);
  int i = threadIdx.x;
  for (; i < n_b; i += tpb) heap.add(sqrt(batch_dists[i]), batch_inds[i]);
  if (i < batch_size) heap.addThreadQ(sqrt(batch_dists[i]), batch_inds[i]);

  heap.checkThreadQ();

  if (threadIdx.x == 0) {
    bool isSet = false;
    while (!isSet) {
      isSet = atomicCAS(mutex + query_id, 0, 1) == 0;
    }
  }

  __syncthreads();

  const int n_k = faiss::gpu::utils::roundDown(k, faiss::gpu::kWarpSize);
  i = threadIdx.x;
  for (; i < n_k; i += tpb)
    heap.add(knn_dists[query_id * k + i], knn_inds[query_id * k + i]);

  if (i < k)
    heap.addThreadQ(knn_dists[query_id * k + i], knn_inds[query_id * k + i]);
  heap.checkThreadQ();
  heap.reduce();

  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    knn_dists[query_id * k + i] = sh_memK[i];
    knn_inds[query_id * k + i] = sh_memV[i];
  }

  if (threadIdx.x == 0) {
    mutex[query_id] = 0;
  }
}

/**
     * It is assumed that batch_size is larger than or equal to k but small enough that
     * it doesn't limit occupancy.
     *
     * When n_cols >= 32 but smaller than the block size (e.g. 128), it's going to be more efficient to
     * parallelize distance computation over warps. When n_cols >= 128, it's going to be more efficient
     * to
     * @tparam value_idx
     * @tparam value_t
     * @tparam value_int
     * @tparam warp_q
     * @tparam thread_q
     * @tparam tpb
     * @param X
     * @param query
     * @param k
     * @param batch_size
     * @param n_cols
     * @param R_indptr
     * @param R_1nn_cols
     * @param plan_query_ids_coo
     * @param plan_landmark_ids_coo
     * @param plan_offset_ids_coo
     */
template <typename value_idx, typename value_t, typename value_int = int,
          int warp_q, int thread_q, int tpb, int batch_size = 2048>
__global__ void compute_dists(const value_t *X, const value_t *query,
                              const int k, const int n_cols,
                              const value_idx *R_indptr,
                              const value_idx *R_1nn_cols,
                              const value_idx *plan_query_ids_coo,
                              const value_idx *plan_landmark_ids_coo,
                              const value_idx *plan_offset_ids_coo, int *mutex,
                              value_idx *knn_inds, value_t *knn_dists) {
  static constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

  value_idx query_id = plan_query_ids_coo[blockIdx.x];

  // Batch size is an upper bound
  __shared__ value_t batch_dists[batch_size];
  __shared__ value_idx batch_inds[batch_size];

  __shared__ value_t sh_memK[kNumWarps * warp_q];
  __shared__ value_idx sh_memV[kNumWarps * warp_q];

  for (int i = threadIdx.x; i < batch_size; i += blockDim.x)
    batch_dists[i] = 0.0;

  __syncthreads();

  value_idx offset_start = plan_offset_ids_coo[blockIdx.x];
  value_idx offset_stop = R_indptr[plan_landmark_ids_coo[blockIdx.x] + 1];

  int working_batch_size =
    min(offset_stop - offset_start, (value_idx)batch_size);

  for (int i = threadIdx.x; i < working_batch_size; i += blockDim.x)
    batch_inds[i] = R_1nn_cols[offset_start + i];

  __syncthreads();

  // in chunks of block_dims, compute distances, store to sh_mem / registers
  for (int i = 0; i < working_batch_size; ++i) {
    value_idx point_index = batch_inds[i];

    value_t dist = 0.0;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
      value_t d = query[query_id * n_cols + j] - X[point_index * n_cols + j];
      dist += d * d;
    }

    // TODO: Use warp-reduction to minimize atomics to smem
    atomicAdd(batch_dists + i, dist);
  }

  topk_merge<value_idx, value_t, warp_q, thread_q, tpb>(
    sh_memK, sh_memV, query_id, batch_inds, batch_dists, working_batch_size, k,
    mutex, knn_inds, knn_dists);
}

/**
 * Executes plan coo across thread-blocks, computing distances for the batches (each edge is a batch).
 * @tparam value_idx
 * @tparam value_t
 * @tparam value_int
 * @param handle
 * @param index
 * @param k
 * @param query
 * @param n_query_pts
 * @param knn_inds
 * @param knn_dists
 * @param batch_size
 * @param weight
 */
template <typename value_idx, typename value_t, typename value_int = uint32_t,
          int batch_size = 2048>
void execute_plan(const raft::handle_t &handle,
                  BallCoverIndex<value_idx, value_t, value_int> &index,
                  raft::sparse::COO<value_idx, value_idx> &plan_coo,
                  value_int k, const value_t *query, value_int n_query_pts,
                  value_idx *knn_inds, value_t *knn_dists, float weight = 1.0) {
  // Each block of the resulting distances kernel needs to look up the current knn to see if the distances
  // are still worth computing, then they need to compute
  rmm::device_uvector<int> mutex(n_query_pts, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), mutex.data(),
               mutex.data() + n_query_pts, 0);
  compute_dists<value_idx, value_t, value_int, 32, 2, 128, batch_size>
    <<<n_query_pts, 128, 0, handle.get_stream()>>>(
      index.get_X(), query, k, index.n, index.get_R_indptr(),
      index.get_R_1nn_cols(), plan_coo.rows(), plan_coo.cols(), plan_coo.vals(),
      mutex.data(), knn_inds, knn_dists);
}

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft