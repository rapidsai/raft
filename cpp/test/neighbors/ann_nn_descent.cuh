/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"
#include "ann_utils.cuh"

#include <raft_internal/neighbors/naive_knn.cuh>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/neighbors/nn_descent.cuh>
#include <raft/util/itertools.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace raft::neighbors::experimental::nn_descent {

struct AnnNNDescentInputs {
  int n_rows;
  int dim;
  int graph_degree;
  raft::distance::DistanceType metric;
  bool host_dataset;
  double min_recall;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnNNDescentInputs& p)
{
  os << "dataset shape=" << p.n_rows << "x" << p.dim << ", graph_degree=" << p.graph_degree
     << ", metric=" << static_cast<int>(p.metric) << (p.host_dataset ? ", host" : ", device")
     << std::endl;
  return os;
}

template <typename DistanceT, typename DataT, typename IdxT>
class AnnNNDescentTest : public ::testing::TestWithParam<AnnNNDescentInputs> {
 public:
  AnnNNDescentTest()
    : stream_(resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnNNDescentInputs>::GetParam()),
      database(0, stream_)
  {
  }

 protected:
  void testNNDescent()
  {
    size_t queries_size = ps.n_rows * ps.graph_degree;
    std::vector<IdxT> indices_NNDescent(queries_size);
    std::vector<IdxT> indices_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      naive_knn<DistanceT, DataT, IdxT>(handle_,
                                        distances_naive_dev.data(),
                                        indices_naive_dev.data(),
                                        database.data(),
                                        database.data(),
                                        ps.n_rows,
                                        ps.n_rows,
                                        ps.dim,
                                        ps.graph_degree,
                                        ps.metric);
      update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      resource::sync_stream(handle_);
    }

    {
      {
        nn_descent::index_params index_params;
        index_params.metric                    = ps.metric;
        index_params.graph_degree              = ps.graph_degree;
        index_params.intermediate_graph_degree = 2 * ps.graph_degree;
        index_params.max_iterations            = 100;

        auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data(), ps.n_rows, ps.dim);

        {
          if (ps.host_dataset) {
            auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
            raft::copy(database_host.data_handle(), database.data(), database.size(), stream_);
            auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
              (const DataT*)database_host.data_handle(), ps.n_rows, ps.dim);
            auto index = nn_descent::build<DataT, IdxT>(handle_, index_params, database_host_view);
            update_host(
              indices_NNDescent.data(), index.graph().data_handle(), queries_size, stream_);
          } else {
            auto index = nn_descent::build<DataT, IdxT>(handle_, index_params, database_view);
            update_host(
              indices_NNDescent.data(), index.graph().data_handle(), queries_size, stream_);
          };
        }
        resource::sync_stream(handle_);
      }

      double min_recall = ps.min_recall;
      EXPECT_TRUE(eval_recall(
        indices_naive, indices_NNDescent, ps.n_rows, ps.graph_degree, 0.001, min_recall));
    }
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::normal(handle_, r, database.data(), ps.n_rows * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.n_rows * ps.dim, DataT(1), DataT(20));
    }
    resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    resource::sync_stream(handle_);
    database.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnNNDescentInputs ps;
  rmm::device_uvector<DataT> database;
};

const std::vector<AnnNNDescentInputs> inputs = raft::util::itertools::product<AnnNNDescentInputs>(
  {1000, 2000},                                              // n_rows
  {3, 5, 7, 8, 17, 64, 128, 137, 192, 256, 512, 619, 1024},  // dim
  {32, 64},                                                  // graph_degree
  {raft::distance::DistanceType::L2Expanded},
  {false, true},
  {0.90});

}  // namespace raft::neighbors::experimental::nn_descent
