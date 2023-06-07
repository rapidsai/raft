/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"
#include "./knn_utils.cuh"
#include <raft/core/resource/cuda_stream.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/brute_force.cuh>
#include <raft/random/rng.cuh>
#include <raft/spatial/knn/knn.cuh>

#include <raft/distance/distance.cuh>

#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <vector>

namespace raft {
namespace spatial {
namespace knn {
struct FusedL2KNNInputs {
  int num_queries;
  int num_db_vecs;
  int dim;
  int k;
  raft::distance::DistanceType metric_;
};

template <typename T>
class FusedL2KNNTest : public ::testing::TestWithParam<FusedL2KNNInputs> {
 public:
  FusedL2KNNTest()
    : stream_(resource::get_cuda_stream(handle_)),
      params_(::testing::TestWithParam<FusedL2KNNInputs>::GetParam()),
      database(params_.num_db_vecs * params_.dim, stream_),
      search_queries(params_.num_queries * params_.dim, stream_),
      raft_indices_(params_.num_queries * params_.k, stream_),
      raft_distances_(params_.num_queries * params_.k, stream_),
      ref_indices_(params_.num_queries * params_.k, stream_),
      ref_distances_(params_.num_queries * params_.k, stream_)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(database.data(), 0, database.size() * sizeof(T), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(search_queries.data(), 0, search_queries.size() * sizeof(T), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(raft_indices_.data(), 0, raft_indices_.size() * sizeof(int64_t), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(raft_distances_.data(), 0, raft_distances_.size() * sizeof(T), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(ref_indices_.data(), 0, ref_indices_.size() * sizeof(int64_t), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(ref_distances_.data(), 0, ref_distances_.size() * sizeof(T), stream_));
  }

 protected:
  void testBruteForce()
  {
    // calculate the naive knn, by calculating the full pairwise distances and doing a k-select
    rmm::device_uvector<T> temp_distances(num_db_vecs * num_queries, stream_);
    distance::pairwise_distance(
      handle_,
      raft::make_device_matrix_view<T, int32_t>(search_queries.data(), num_queries, dim),
      raft::make_device_matrix_view<T, int32_t>(database.data(), num_db_vecs, dim),
      raft::make_device_matrix_view<T, int32_t>(temp_distances.data(), num_queries, num_db_vecs),
      metric);

    spatial::knn::select_k<int64_t, T>(temp_distances.data(),
                                       nullptr,
                                       num_queries,
                                       num_db_vecs,
                                       ref_distances_.data(),
                                       ref_indices_.data(),
                                       true,
                                       k_,
                                       stream_);

    auto index_view =
      raft::make_device_matrix_view<const T, int64_t>(database.data(), num_db_vecs, dim);
    auto query_view =
      raft::make_device_matrix_view<const T, int64_t>(search_queries.data(), num_queries, dim);
    auto out_indices_view =
      raft::make_device_matrix_view<int64_t, int64_t>(raft_indices_.data(), num_queries, k_);
    auto out_dists_view =
      raft::make_device_matrix_view<T, int64_t>(raft_distances_.data(), num_queries, k_);
    raft::neighbors::brute_force::fused_l2_knn(
      handle_, index_view, query_view, out_indices_view, out_dists_view, metric);

    // verify.
    ASSERT_TRUE(devArrMatchKnnPair(ref_indices_.data(),
                                   raft_indices_.data(),
                                   ref_distances_.data(),
                                   raft_distances_.data(),
                                   num_queries,
                                   k_,
                                   float(0.001),
                                   stream_));
  }

  void SetUp() override
  {
    num_queries = params_.num_queries;
    num_db_vecs = params_.num_db_vecs;
    dim         = params_.dim;
    k_          = params_.k;
    metric      = params_.metric_;

    unsigned long long int seed = 1234ULL;
    raft::random::RngState r(seed);
    uniform(handle_, r, database.data(), num_db_vecs * dim, T(-1.0), T(1.0));
    uniform(handle_, r, search_queries.data(), num_queries * dim, T(-1.0), T(1.0));
  }

 private:
  raft::resources handle_;
  cudaStream_t stream_ = 0;
  FusedL2KNNInputs params_;
  int num_queries;
  int num_db_vecs;
  int dim;
  rmm::device_uvector<T> database;
  rmm::device_uvector<T> search_queries;
  rmm::device_uvector<int64_t> raft_indices_;
  rmm::device_uvector<T> raft_distances_;
  rmm::device_uvector<int64_t> ref_indices_;
  rmm::device_uvector<T> ref_distances_;
  int k_;
  raft::distance::DistanceType metric;
};

const std::vector<FusedL2KNNInputs> inputs = {
  {100, 1000, 16, 10, raft::distance::DistanceType::L2Expanded},
  {256, 256, 30, 10, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 10, raft::distance::DistanceType::L2Expanded},
  {100, 1000, 16, 50, raft::distance::DistanceType::L2Expanded},
  {20, 10000, 16, 10, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 50, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 32, 50, raft::distance::DistanceType::L2Expanded},
  {10000, 40000, 32, 30, raft::distance::DistanceType::L2Expanded},
  // L2 unexpanded
  {100, 1000, 16, 10, raft::distance::DistanceType::L2Unexpanded},
  {1000, 10000, 16, 10, raft::distance::DistanceType::L2Unexpanded},
  {100, 1000, 16, 50, raft::distance::DistanceType::L2Unexpanded},
  {20, 10000, 16, 50, raft::distance::DistanceType::L2Unexpanded},
  {1000, 10000, 16, 50, raft::distance::DistanceType::L2Unexpanded},
  {1000, 10000, 32, 50, raft::distance::DistanceType::L2Unexpanded},
  {10000, 40000, 32, 30, raft::distance::DistanceType::L2Unexpanded},
};

typedef FusedL2KNNTest<float> FusedL2KNNTestF;
TEST_P(FusedL2KNNTestF, FusedBruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(FusedL2KNNTest, FusedL2KNNTestF, ::testing::ValuesIn(inputs));

}  // namespace knn
}  // namespace spatial
}  // namespace raft
