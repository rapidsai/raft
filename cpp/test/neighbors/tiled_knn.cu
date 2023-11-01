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
#include <raft/core/resource/cuda_stream.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance.cuh>  // raft::distance::pairwise_distance
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>
#include <raft/neighbors/brute_force.cuh>
#include <raft/neighbors/brute_force_batch_k_query.cuh>
#include <raft/neighbors/detail/knn_brute_force.cuh>  // raft::neighbors::detail::brute_force_knn_impl
#include <raft/neighbors/detail/selection_faiss.cuh>  // raft::neighbors::detail::select_k

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
    : stream_(resource::get_cuda_stream(handle_)),
      params_(::testing::TestWithParam<TiledKNNInputs>::GetParam()),
      database(params_.num_db_vecs * params_.dim, stream_),
      search_queries(params_.num_queries * params_.dim, stream_),
      raft_indices_(params_.num_queries * params_.k, stream_),
      raft_distances_(params_.num_queries * params_.k, stream_),
      ref_indices_(params_.num_queries * params_.k, stream_),
      ref_distances_(params_.num_queries * params_.k, stream_)
  {
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(database.data(), params_.num_db_vecs, params_.dim),
      T{0.0});
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(search_queries.data(), params_.num_queries, params_.dim),
      T{0.0});
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(raft_indices_.data(), params_.num_queries, params_.k),
      0);
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(raft_distances_.data(), params_.num_queries, params_.k),
      T{0.0});
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(ref_indices_.data(), params_.num_queries, params_.k),
      0);
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(ref_distances_.data(), params_.num_queries, params_.k),
      T{0.0});
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

    raft::neighbors::detail::select_k<int, T>(temp_dist,
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
      neighbors::detail::brute_force_knn_impl<size_t, int, T>(handle_,
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
    ASSERT_TRUE(raft::spatial::knn::devArrMatchKnnPair(ref_indices_.data(),
                                                       raft_indices_.data(),
                                                       ref_distances_.data(),
                                                       raft_distances_.data(),
                                                       num_queries,
                                                       k_,
                                                       float(0.001),
                                                       stream_,
                                                       true));

    // Also test out the 'index' api - where we can use precomputed norms
    if (params_.row_major) {
      auto idx =
        raft::neighbors::brute_force::build<T>(handle_,
                                               raft::make_device_matrix_view<const T, int64_t>(
                                                 database.data(), params_.num_db_vecs, params_.dim),
                                               metric,
                                               metric_arg);

      auto query_view = raft::make_device_matrix_view<const T, int64_t>(
        search_queries.data(), params_.num_queries, params_.dim);

      raft::neighbors::brute_force::search<T, int>(
        handle_,
        idx,
        query_view,
        raft::make_device_matrix_view<int, int64_t>(
          raft_indices_.data(), params_.num_queries, params_.k),
        raft::make_device_matrix_view<T, int64_t>(
          raft_distances_.data(), params_.num_queries, params_.k));

      ASSERT_TRUE(raft::spatial::knn::devArrMatchKnnPair(ref_indices_.data(),
                                                         raft_indices_.data(),
                                                         ref_distances_.data(),
                                                         raft_distances_.data(),
                                                         num_queries,
                                                         k_,
                                                         float(0.001),
                                                         stream_,
                                                         true));
      // also test out the batch api. First get new reference results (all k, up to a certain
      // max size)
      auto all_size      = std::min(params_.num_db_vecs, 1024);
      auto all_indices   = raft::make_device_matrix<int, int64_t>(handle_, num_queries, all_size);
      auto all_distances = raft::make_device_matrix<T, int64_t>(handle_, num_queries, all_size);
      raft::neighbors::brute_force::search<T, int>(
        handle_, idx, query_view, all_indices.view(), all_distances.view());

      int64_t offset = 0;
      for (auto batch : batch_k_query<T, int>(handle_, idx, query_view, k_)) {
        auto batch_size = batch.indices().extent(1);
        auto indices    = raft::make_device_matrix<int, int64_t>(handle_, num_queries, batch_size);
        auto distances  = raft::make_device_matrix<T, int64_t>(handle_, num_queries, batch_size);

        matrix::slice_coordinates<int64_t> coords{0, offset, num_queries, offset + batch_size};

        matrix::slice(handle_, raft::make_const_mdspan(all_indices.view()), indices.view(), coords);
        matrix::slice(
          handle_, raft::make_const_mdspan(all_distances.view()), distances.view(), coords);

        ASSERT_TRUE(raft::spatial::knn::devArrMatchKnnPair(indices.data_handle(),
                                                           batch.indices().data_handle(),
                                                           distances.data_handle(),
                                                           batch.distances().data_handle(),
                                                           num_queries,
                                                           batch.indices().extent(1),
                                                           float(0.001),
                                                           stream_,
                                                           true));

        offset += batch_size;
        if (offset + batch_size > all_size) break;
      }
    }
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
  raft::resources handle_;
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
  // Passing tile sizes of 0 means to use brute_force_knn_impl (instead of the
  // tiled_brute_force_knn api).
  {1000, 500000, 128, 128, 0, 0, raft::distance::DistanceType::L2Expanded, true},
  {1000, 500000, 128, 128, 0, 0, raft::distance::DistanceType::L2Expanded, false},
  {1000, 5000, 128, 128, 0, 0, raft::distance::DistanceType::LpUnexpanded, true},
  {1000, 5000, 128, 128, 0, 0, raft::distance::DistanceType::L2SqrtExpanded, false},
  {1000, 5000, 128, 128, 0, 0, raft::distance::DistanceType::InnerProduct, false}};

typedef TiledKNNTest<float> TiledKNNTestF;
TEST_P(TiledKNNTestF, BruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(TiledKNNTest, TiledKNNTestF, ::testing::ValuesIn(random_inputs));
}  // namespace raft::neighbors::brute_force
