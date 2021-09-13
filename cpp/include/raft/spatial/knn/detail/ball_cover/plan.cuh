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

#include "../block_select_faiss.cuh"
#include "../selection_faiss.cuh"

#include <limits.h>

#include <raft/cuda_utils.cuh>

#include <faiss/utils/Heap.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>

#include <raft/distance/distance.cuh>
#include <raft/sparse/coo.cuh>
#include <raft/selection/col_wise_sort.cuh>

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
    template <typename value_idx, typename value_t, typename value_int = int,
            int tpb = 32, typename reduce_func, typename accum_func>
    __global__ void prune_landmarks(
            const value_t *landmark_dists, value_int n_cols,
            const value_idx *R_knn_inds,
            const value_t *R_knn_dists,
            const value_t *R_radius,
            const value_t *landmarks,
            const value_t *knn_dists,
            const value_idx *landmarks_indptr,
            int n_landmarks,
            int bitset_size, int k,
            int batch_size,
            uint32_t *output_bitset,
            value_idx *total_landmark_points,
            float weight = 1.0) {
        static constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

        // allocate array of size n_landmarks / 32 ints
        extern __shared__ uint32_t smem[];

        // Start with all bits on
        for (int i = threadIdx.x; i < bitset_size; i += tpb) smem[i] = 0xffffffff;

        __syncthreads();

        value_t closest_R_dist = R_knn_dists[blockIdx.x * k + (k-1)];

        value_idx n_points = 0;

        // Discard any landmarks where p(q, r) > p(q, r_q) + radius(r)
        // That is, the distance between the current point and the current
        // landmark is > the distance between the current point and
        // its closest landmark + the radius of the current landmark.
        for (int l = threadIdx.x; l < n_landmarks; l += kNumWarps) {
            value_idx cardinality = landmarks_indptr[l+1] - landmarks_indptr[l];
            value_t p_q_r = pw_dists[blockIdx.x * n_landmarks + threadIdx.x];

            if(p_q_r > weight * (closest_R_dist + R_radius[l]) ||
                                   p_q_r > 3 * closest_R_dist ||
                                   knn_dists[blockIdx.x * k + (k-1)] < p_q_r - R_radius[l]) {
                _zero_bit(smem, l);
                n_points += ceil(cardinality / batch_size);
            }
        }

        __syncthreads();

        /**
         * Output bitset
         */
        for (int l = threadIdx.x; l < bitset_size; l += tpb) {
            output[blockIdx.x * bitset_size + l] = smem[l];
        }

        atomicAdd(total_landmark_points+blockIdx.x, n_points);
    }

    template <typename value_idx, typename value_t, typename value_int = int,
            int tpb = 32, typename reduce_func, typename accum_func>
    __global__ void write_plan_coo(const value_idx *landmark_indptr,
                                   const value_idx *coo_write_plan,
                                   const uint32_t *bitset,
                                   int bitset_size,
                                   int n_landmarks,
                                   int batch_size,
                                   value_idx *plan_query_ids_coo,
                                   value_idx *plan_landmark_ids_coo,
                                   value_idx *plan_offset_ids_coo) {
        int query_id = blockIdx.x * blockDim.x + threadIdx.x;

        int cur_plan_offset = coo_write_plan[query_id];

        for (int cur_R_ind = 0; cur_R_ind < n_landmarks; cur_R_ind++) {
            // if cur R overlaps cur point's closest R, it could be a
            // candidate
            if (_get_val(bitset + (query_id * bitset_size), cur_R_ind)) {

                value_idx start_offset = landmark_indptr[cur_R_ind];
                value_idx stop_offset = landmark_indptr[cur_R_ind+1];
                for(int batch_offset = start_offset; batch_offset < stop_offset; batch+=batch_size) {
                    plan_query_ids_coo[cur_plan_offset] = query_id;
                    plan_landmark_ids_coo[cur_plan_offset] = cur_R_ind;
                    plan_offset_ids_coo[cur_plan_offset] = batch_offset;
                    ++cur_plan_offset;
                }
            }
        }
    }

    template<typename value_idx, typename value_t, typename value_int = int>
    void landmark_q_pw_dists(const raft::handle_t &handle,
                             BallCoverIndex<value_idx, value_t> &index,
                             const value_t *queries
                             value_int n_queries,
                             value_t *out_dists) {

        raft::device_uvector<char> pw_workspace(0, handle.get_stream());

        // Compute pairwise dists between queries and landmarks.
        raft::pairwise_distances(queries, index.get_landmarks(), out_dists,
                                 n_queries, index.get_n_landmarks(), index.n,
                                 pw_workspace, index.get_metric(), handle.get_stream());
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
    template<typename value_idx, typename value_t, typename value_int = int>
    void compute_plan(const raft::handle_t &handle,
                      BallCoverIndex<value_idx, value_t> &index,
                      int k,
                      const value_t *query,
                      value_int n_query_pts,
                      const value_idx *knn_inds,
                      const value_t *knn_dists,
                      raft::sparse::COO<value_t, value_idx> &plan_coo,
                      int batch_size = -1,
                      float weight = 1.0) {

        ASSERT(plan_coo.nnz == 0, "Plan COO is expecteo be uninitialized");

        // The average landmark cardinality tends to have a mean very close
        // to n_landmarks so we can guarantee batch_size will never be greater.
        if(batch_size < 0) batch_size = index.get_n_landmarks();

        auto exec_policy = rmm::exec_policy(handle.get_stream());
        /**
         * Query Plan is a COO matrix mapping (query_point_id, landmark_id, landmark_index_start_offset)
         * for each query point. This is meant to be done in batches over a pairwise distance matrix
         * between the query points and landmarks and increase uniformity of distance computations.
         */

        /**
         * Steps (this can be done in batches both by query points and landmarks:
         * 1. Compute pairwise_distances(query_points, landmarks)
         * 2. K-select to get radius for each query point
         * 3. Apply triangle inequality and bounds checking
         *      a) Compute cardinalities for each landmark set
         *      b) Compute n_batches
         *
         * 4. Create coo w/ nnz = n_batches.sum()
         * 5. Populate coo w/ batch information- rows=query_row_id, cols=landmark_id, vals=start_offset
         */

        rmm::device_uvector<value_t> ql_dists(n_queries * index.n_landmarks, handle.get_stream());

        // Compute pw dists between queries and landmarks
        landmark_q_pw_dists(handle, index, query, n_query_pts, ql_dists.data());

        // K-select to get radius for each query point
        rmm::device_uvector<value_idx> R_knn_inds(k * index.m, handle.get_stream());
        rmm::device_uvector<value_t> R_knn_dists(k * index.m, handle.get_stream());

        k_closest_landmarks(handle, index, query, n_query_pts, k, R_knn_inds.data(),
                            R_knn_dists.data());

        // Compute filtered balls for current batch based on k found so far
        const int bitset_size = ceil(index.n_landmarks / 32.0);
        rmm::device_uvector<uint32_t> bitset(bitset_size * index.m,
                                             handle.get_stream());

        rmm::device_uvector<value_int> landmark_batches(n_query_pts, handle.get_stream());

        prune_landmarks<<<n_query_pts, 128, 0, handle.get_stream()>>>(
                ql_dists.data(), index.n, R_knn_inds.data(),
                R_knn_dists.data(), R_radius.data(),
                index.get_R_radius(), index.get_R(),
                knn_dists, index.get_R_indptr(),
                index.get_n_landmarks(), bitset_size, k,
                bitset.data(), landmark_batches.data(),
                weight);

        // Sum of cardinality array is nnz of plan
        value_idx n_batches = thrust::reduce(exec_policy, landmark_batches.data(),
                                             landmark_batches.data()+n_query_pts, 0);

        rmm::device_uvector<value_int> coo_write_plan(n_query_pts, handle.get_stream());
        thrust::exclusive_scan(exec_policy, landmark_batches.data(), landmark_batches.data()+n_query_pts,
                               coo_write_plan.data(), 0);

        // Construct COO where nnz=n_batches
        plan_coo.allocate(n_batches);

        write_plan_coo<<<raft::ceildiv(n_query_pts, 256), 256, 0, handle.get_stream()>>>(
                index.get_R_indptr(), coo_write_plan.data(), bitset.data(),
                bitset_size, n_landmarks, batch_size,
                plan_coo.rows(), plan_coo.cols(), plan_coo.vals());
    }

    /**
     * Executes plan coo across thread-blocks, computing distances for the batches (each edge).
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
    template<typename value_idx, typename value_t, typename value_int = int>
    void execute_plan(const raft::handle_t &handle,
                      BallCoverIndex<value_idx, value_t> &index,
                      const raft::sparse::COO<value_t, value_idx> &plan_coo,
                      int k, const value_t *query,
                      value_int n_query_pts,
                      value_idx *knn_inds,
                      value_t *knn_dists,
                      int batch_size = -1,
                      float weight = 1.0) {

        
    }

};
};
};
};