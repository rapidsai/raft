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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/brute_force.cuh>

#if defined RAFT_NN_COMPILED
#include <raft/distance/specializations.cuh>
#include <raft/neighbors/specializations.cuh>
#endif

#include <raft/spatial/knn/detail/faiss_select/DistanceUtils.h>

#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <vector>

namespace raft::neighbors::brute_force {
struct TiledKNNInputs {
  int num_queries;
  int num_db_vecs;
  int dim;
  int k;
  int row_tiles;
  int col_tiles;
  raft::distance::DistanceType metric_;
};

template <typename T>
class TiledKNNTest : public ::testing::TestWithParam<TiledKNNInputs> {
 public:
  TiledKNNTest()
    : stream_(handle_.get_stream()),
      params_(::testing::TestWithParam<TiledKNNInputs>::GetParam()),
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
      cudaMemsetAsync(raft_indices_.data(), 0, raft_indices_.size() * sizeof(int), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(raft_distances_.data(), 0, raft_distances_.size() * sizeof(T), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(ref_indices_.data(), 0, ref_indices_.size() * sizeof(int), stream_));
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
      raft::make_device_matrix_view<T, int>(search_queries.data(), num_queries, dim),
      raft::make_device_matrix_view<T, int>(database.data(), num_db_vecs, dim),
      raft::make_device_matrix_view<T, int>(temp_distances.data(), num_queries, num_db_vecs),
      metric);

    using namespace raft::spatial;
    knn::select_k<int, T>(temp_distances.data(),
                          nullptr,
                          num_queries,
                          num_db_vecs,
                          ref_distances_.data(),
                          ref_indices_.data(),
                          true,
                          k_,
                          stream_);

    knn::detail::tiled_brute_force_knn(handle_,
                                       search_queries.data(),
                                       database.data(),
                                       num_queries,
                                       num_db_vecs,
                                       dim,
                                       k_,
                                       raft_distances_.data(),
                                       raft_indices_.data(),
                                       metric,
                                       params_.row_tiles,
                                       params_.col_tiles);

    // verify.
    ASSERT_TRUE(knn::devArrMatchKnnPair(ref_indices_.data(),
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
  raft::device_resources handle_;
  cudaStream_t stream_ = 0;
  TiledKNNInputs params_;
  int num_queries;
  int num_db_vecs;
  int dim;
  rmm::device_uvector<T> database;
  rmm::device_uvector<T> search_queries;
  rmm::device_uvector<int> raft_indices_;
  rmm::device_uvector<T> raft_distances_;
  rmm::device_uvector<int> ref_indices_;
  rmm::device_uvector<T> ref_distances_;
  int k_;
  raft::distance::DistanceType metric;
};

const std::vector<TiledKNNInputs> random_inputs = {
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::L2Expanded},
  {4, 12, 32, 6, 4, 8, raft::distance::DistanceType::L2Expanded},
  {10000, 40000, 32, 30, 512, 1024, raft::distance::DistanceType::L2Expanded},
};

typedef TiledKNNTest<float> TiledKNNTestF;
TEST_P(TiledKNNTestF, BruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(TiledKNNTest, TiledKNNTestF, ::testing::ValuesIn(random_inputs));
}  // namespace raft::neighbors::brute_force
