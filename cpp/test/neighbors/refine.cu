/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "../test_utils.h"
#include "ann_utils.cuh"

#include "refine_helper.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/detail/refine.cuh>
#include <raft/neighbors/refine.cuh>
#include <raft/spatial/knn/ann.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <gtest/gtest.h>

#if defined RAFT_NN_COMPILED
#include <raft/neighbors/specializations.cuh>
#endif

#include <vector>

namespace raft::neighbors {

template <typename DataT, typename DistanceT, typename IdxT>
class RefineTest : public ::testing::TestWithParam<detail::RefineInputs<IdxT>> {
 public:
  RefineTest()
    : stream_(handle_.get_stream()),
      data(handle_, ::testing::TestWithParam<detail::RefineInputs<IdxT>>::GetParam())
  {
  }

 protected:
 public:  // tamas remove
  void testRefine()
  {
    std::vector<IdxT> indices(data.p.n_queries * data.p.k);
    std::vector<DistanceT> distances(data.p.n_queries * data.p.k);

    if (data.p.host_data) {
      raft::neighbors::refine_host<IdxT, DataT, DistanceT, IdxT>(handle_,
                                                                 data.dataset_host.view(),
                                                                 data.queries_host.view(),
                                                                 data.candidates_host.view(),
                                                                 data.refined_indices_host.view(),
                                                                 data.refined_distances_host.view(),
                                                                 data.p.metric);
      raft::copy(indices.data(),
                 data.refined_indices_host.data_handle(),
                 data.refined_indices_host.size(),
                 stream_);
      raft::copy(distances.data(),
                 data.refined_distances_host.data_handle(),
                 data.refined_distances_host.size(),
                 stream_);

    } else {
      raft::neighbors::refine<IdxT, DataT, DistanceT, IdxT>(handle_,
                                                            data.dataset.view(),
                                                            data.queries.view(),
                                                            data.candidates.view(),
                                                            data.refined_indices.view(),
                                                            data.refined_distances.view(),
                                                            data.p.metric);
      update_host(distances.data(),
                  data.refined_distances.data_handle(),
                  data.refined_distances.size(),
                  stream_);
      update_host(
        indices.data(), data.refined_indices.data_handle(), data.refined_indices.size(), stream_);
    }
    handle_.sync_stream(stream_);

    double min_recall = 1;

    ASSERT_TRUE(raft::neighbors::eval_neighbours(data.true_refined_indices_host,
                                                 indices,
                                                 data.true_refined_distances_host,
                                                 distances,
                                                 data.p.n_queries,
                                                 data.p.k,
                                                 0.001,
                                                 min_recall));
  }

 public:
  raft::handle_t handle_;
  rmm::cuda_stream_view stream_;
  detail::RefineHelper<DataT, DistanceT, IdxT> data;
};

const std::vector<detail::RefineInputs<int64_t>> inputs =
  raft::util::itertools::product<detail::RefineInputs<int64_t>>(
    {137},
    {1000},
    {16},
    {1, 10, 33},
    {33},
    {raft::distance::DistanceType::L2Expanded},
    {false, true});

typedef RefineTest<float, float, std::int64_t> RefineTestF;
TEST_P(RefineTestF, AnnRefine) { this->testRefine(); }

INSTANTIATE_TEST_CASE_P(RefineTest, RefineTestF, ::testing::ValuesIn(inputs));

typedef RefineTest<uint8_t, float, std::int64_t> RefineTestF_uint8;
TEST_P(RefineTestF_uint8, AnnRefine) { this->testRefine(); }
INSTANTIATE_TEST_CASE_P(RefineTest, RefineTestF_uint8, ::testing::ValuesIn(inputs));

}  // namespace raft::neighbors
