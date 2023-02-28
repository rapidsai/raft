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
#include "./ann_utils.cuh"
#include "./knn_utils.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/transpose.cuh>
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
  raft::distance::DistanceType metric;
  bool row_major;
};

std::ostream& operator<<(std::ostream& os, const TiledKNNInputs& input)
{
  return os << "num_queries:" << input.num_queries << " num_vecs:" << input.num_db_vecs
            << " dim:" << input.dim << " k:" << input.k << " row_tiles:" << input.row_tiles
            << " col_tiles:" << input.col_tiles << " metric:" << print_metric{input.metric}
            << " row_major:" << input.row_major;
}

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
    float metric_arg = 3.0;

    // calculate the naive knn, by calculating the full pairwise distances and doing a k-select
    rmm::device_uvector<T> temp_distances(num_db_vecs * num_queries, stream_);
    rmm::device_uvector<char> workspace(0, stream_);
    distance::pairwise_distance(handle_,
                                search_queries.data(),
                                database.data(),
                                temp_distances.data(),
                                num_queries,
                                num_db_vecs,
                                dim,
                                workspace,
                                metric,
                                params_.row_major,
                                metric_arg);

    // setting the 'isRowMajor' flag in the pairwise distances api, not only sets
    // the inputs as colmajor - but also the output. this means we have to transpose in this
    // case
    auto temp_dist = temp_distances.data();
    rmm::device_uvector<T> temp_row_major_dist(num_db_vecs * num_queries, stream_);
    if (!params_.row_major) {
      raft::linalg::transpose(
        handle_, temp_dist, temp_row_major_dist.data(), num_queries, num_db_vecs, stream_);
      temp_dist = temp_row_major_dist.data();
    }

    using namespace raft::spatial;
    knn::select_k<int, T>(temp_dist,
                          nullptr,
                          num_queries,
                          num_db_vecs,
                          ref_distances_.data(),
                          ref_indices_.data(),
                          raft::distance::is_min_close(metric),
                          k_,
                          stream_);

    if ((params_.row_tiles == 0) && (params_.col_tiles == 0)) {
      std::vector<T*> input{database.data()};
      std::vector<size_t> sizes{static_cast<size_t>(num_db_vecs)};
      raft::spatial::knn::brute_force_knn<int, T, size_t>(handle_,
                                                          input,
                                                          sizes,
                                                          dim,
                                                          const_cast<T*>(search_queries.data()),
                                                          num_queries,
                                                          raft_indices_.data(),
                                                          raft_distances_.data(),
                                                          k_,
                                                          params_.row_major,
                                                          params_.row_major,
                                                          nullptr,
                                                          metric,
                                                          metric_arg);
    } else {
      neighbors::detail::tiled_brute_force_knn(handle_,
                                               search_queries.data(),
                                               database.data(),
                                               num_queries,
                                               num_db_vecs,
                                               dim,
                                               k_,
                                               raft_distances_.data(),
                                               raft_indices_.data(),
                                               metric,
                                               metric_arg,
                                               params_.row_tiles,
                                               params_.col_tiles);
    }

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
    metric      = params_.metric;

    unsigned long long int seed = 1234ULL;
    raft::random::RngState r(seed);

    // JensenShannon distance requires positive values
    T min_val = metric == raft::distance::DistanceType::JensenShannon ? T(0.0) : T(-1.0);
    uniform(handle_, r, database.data(), num_db_vecs * dim, min_val, T(1.0));
    uniform(handle_, r, search_queries.data(), num_queries * dim, min_val, T(1.0));
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
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::L2Expanded, true},
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::L2Unexpanded, true},
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::L2SqrtExpanded, true},
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::L2SqrtUnexpanded, true},
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::L1, true},
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::Linf, true},
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::InnerProduct, true},
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::CorrelationExpanded, true},
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::CosineExpanded, true},
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::LpUnexpanded, true},
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::JensenShannon, true},
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::L2SqrtExpanded, true},
  // BrayCurtis isn't currently supported by pairwise_distance api
  // {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::BrayCurtis},
  {256, 512, 16, 8, 16, 8, raft::distance::DistanceType::Canberra, true},
  {10000, 40000, 32, 30, 512, 1024, raft::distance::DistanceType::L2Expanded, true},
  {345, 1023, 16, 128, 512, 1024, raft::distance::DistanceType::CosineExpanded, true},
  {789, 20516, 64, 256, 512, 4096, raft::distance::DistanceType::L2SqrtExpanded, true},
  // Test where the final column tile has < K items:
  {4, 12, 32, 6, 4, 8, raft::distance::DistanceType::L2Expanded, true},
  // Test where passing column_tiles < K
  {1, 40, 32, 30, 1, 8, raft::distance::DistanceType::L2Expanded, true},
  // Passing tile sizes of 0 means to use the public api (instead of the
  // detail api). Note that we can only test col_major in the public api
  {1000, 500000, 128, 128, 0, 0, raft::distance::DistanceType::L2Expanded, true},
  {1000, 500000, 128, 128, 0, 0, raft::distance::DistanceType::L2Expanded, false},
  {1000, 5000, 128, 128, 0, 0, raft::distance::DistanceType::LpUnexpanded, true},
  {1000, 5000, 128, 128, 0, 0, raft::distance::DistanceType::L2SqrtExpanded, false},
  {1000, 5000, 128, 128, 0, 0, raft::distance::DistanceType::InnerProduct, false}};

typedef TiledKNNTest<float> TiledKNNTestF;
TEST_P(TiledKNNTestF, BruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(TiledKNNTest, TiledKNNTestF, ::testing::ValuesIn(random_inputs));
}  // namespace raft::neighbors::brute_force
