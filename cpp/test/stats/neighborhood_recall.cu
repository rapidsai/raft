/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../neighbors/ann_utils.cuh"
#include "../test_utils.h"

#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/neighborhood_recall.cuh>
#include <raft/util/itertools.hpp>

#include <raft_internal/neighbors/naive_knn.cuh>

#include <gtest/gtest.h>

namespace raft::stats {

struct NeighborhoodRecallInputs {
  int n_rows;
  int n_cols;
  int k;
};

template <typename DistanceT, typename IdxT>
class NeighborhoodRecallTest : public ::testing::TestWithParam<NeighborhoodRecallInputs> {
 public:
  NeighborhoodRecallTest()
    : ps{::testing::TestWithParam<NeighborhoodRecallInputs>::GetParam()},
      data_1{raft::make_device_matrix<DistanceT, IdxT>(res, ps.n_rows, ps.n_cols)},
      data_2{raft::make_device_matrix<DistanceT, IdxT>(res, ps.n_rows, ps.n_cols)}
  {
  }

 protected:
  void test_recall()
  {
    size_t queries_size = ps.n_rows * ps.k;

    // calculate nn for dataset 1
    auto distances_1 = raft::make_device_matrix<DistanceT, IdxT>(res, ps.n_rows, ps.k);
    auto indices_1   = raft::make_device_matrix<IdxT, IdxT>(res, ps.n_rows, ps.k);
    raft::neighbors::naive_knn<DistanceT, DistanceT, IdxT>(
      res,
      distances_1.data_handle(),
      indices_1.data_handle(),
      data_1.data_handle(),
      data_1.data_handle(),
      ps.n_rows,
      ps.n_rows,
      ps.n_cols,
      ps.k,
      raft::distance::DistanceType::L2Expanded);
    std::vector<DistanceT> distances_1_h(queries_size);
    std::vector<IdxT> indices_1_h(queries_size);
    raft::copy(distances_1_h.data(),
               distances_1.data_handle(),
               ps.n_rows * ps.k,
               raft::resource::get_cuda_stream(res));
    raft::copy(indices_1_h.data(),
               indices_1.data_handle(),
               ps.n_rows * ps.k,
               raft::resource::get_cuda_stream(res));

    // calculate nn for dataset 2
    auto distances_2 = raft::make_device_matrix<DistanceT, IdxT>(res, ps.n_rows, ps.k);
    auto indices_2   = raft::make_device_matrix<IdxT, IdxT>(res, ps.n_rows, ps.k);
    raft::neighbors::naive_knn<DistanceT, DistanceT, IdxT>(
      res,
      distances_2.data_handle(),
      indices_2.data_handle(),
      data_2.data_handle(),
      data_2.data_handle(),
      ps.n_rows,
      ps.n_rows,
      ps.n_cols,
      ps.k,
      raft::distance::DistanceType::L2Expanded);
    std::vector<DistanceT> distances_2_h(queries_size);
    std::vector<IdxT> indices_2_h(queries_size);
    raft::copy(distances_2_h.data(),
               distances_2.data_handle(),
               ps.n_rows * ps.k,
               raft::resource::get_cuda_stream(res));
    raft::copy(indices_2_h.data(),
               indices_2.data_handle(),
               ps.n_rows * ps.k,
               raft::resource::get_cuda_stream(res));

    raft::resource::sync_stream(res);

    // find CPU recall scores
    [[maybe_unused]] auto [indices_only_recall_h, mc1, tc1] =
      raft::neighbors::calc_recall(indices_1_h, indices_2_h, ps.n_rows, ps.k);
    [[maybe_unused]] auto [recall_h, mc2, tc2] = raft::neighbors::calc_recall(
      indices_1_h, indices_2_h, distances_1_h, distances_2_h, ps.n_rows, ps.k, 0.001);

    // find GPU recall scores
    auto s1                         = 0;
    auto indices_only_recall_scalar = raft::make_host_scalar<double>(s1);
    neighborhood_recall(res,
                        raft::make_const_mdspan(indices_1.view()),
                        raft::make_const_mdspan(indices_2.view()),
                        indices_only_recall_scalar.view());

    auto s2            = 0;
    auto recall_scalar = raft::make_host_scalar<double>(s2);
    DistanceT s3       = 0.001;
    auto eps_mda       = raft::make_host_scalar<DistanceT>(s3);

    neighborhood_recall<IdxT, IdxT, double, DistanceT>(res,
                                                       raft::make_const_mdspan(indices_1.view()),
                                                       raft::make_const_mdspan(indices_2.view()),
                                                       recall_scalar.view(),
                                                       raft::make_const_mdspan(distances_1.view()),
                                                       raft::make_const_mdspan(distances_2.view()));

    // assert correctness
    ASSERT_TRUE(raft::match(indices_only_recall_h,
                            *indices_only_recall_scalar.data_handle(),
                            raft::CompareApprox<double>(0.01)));
    ASSERT_TRUE(
      raft::match(recall_h, *recall_scalar.data_handle(), raft::CompareApprox<double>(0.01)));
  }

  void SetUp() override
  {
    // form two random datasets
    raft::random::Rng r1(1234ULL);
    r1.normal(data_1.data_handle(),
              ps.n_rows * ps.n_cols,
              DistanceT(0.1),
              DistanceT(2.0),
              raft::resource::get_cuda_stream(res));
    raft::random::Rng r2(21111ULL);
    r2.normal(data_2.data_handle(),
              ps.n_rows * ps.n_cols,
              DistanceT(0.1),
              DistanceT(2.0),
              raft::resource::get_cuda_stream(res));
    resource::sync_stream(res);
  }

 private:
  raft::resources res;
  NeighborhoodRecallInputs ps;
  raft::device_matrix<DistanceT, IdxT> data_1;
  raft::device_matrix<DistanceT, IdxT> data_2;
};

const std::vector<NeighborhoodRecallInputs> inputs =
  raft::util::itertools::product<NeighborhoodRecallInputs>({10, 50, 100},  // n_rows
                                                           {80, 100},      // n_cols
                                                           {32, 64});      // k

using NeighborhoodRecallTestF_U32 = NeighborhoodRecallTest<float, std::uint32_t>;
TEST_P(NeighborhoodRecallTestF_U32, AnnCagra) { this->test_recall(); }

INSTANTIATE_TEST_CASE_P(NeighborhoodRecallTest,
                        NeighborhoodRecallTestF_U32,
                        ::testing::ValuesIn(inputs));

}  // end namespace raft::stats
