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
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/map.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/neighbors/ivf_list.hpp>
#include <raft/neighbors/sample_filter.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/fast_int_div.cuh>
#include <thrust/functional.h>

#include <raft_internal/neighbors/naive_knn.cuh>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/matrix/gather.cuh>
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/neighbors/ivf_flat_helpers.cuh>
#include <raft/random/rng.cuh>
#include <raft/spatial/knn/ann.cuh>
#include <raft/spatial/knn/knn.cuh>
#include <raft/stats/mean.cuh>
#include <raft/neighbors/ann_mg.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <rmm/device_uvector.hpp>
#include <thrust/sequence.h>

#include <cstddef>
#include <iostream>
#include <vector>

namespace raft::neighbors::mg {

template <typename IdxT>
struct AnnMGInputs {
  IdxT num_queries;
  IdxT num_db_vecs;
  IdxT dim;
  IdxT k;
  IdxT nprobe;
  IdxT nlist;
  raft::distance::DistanceType metric;
  bool adaptive_centers;
};

template <typename T, typename DataT, typename IdxT>
class AnnMGTest : public ::testing::TestWithParam<AnnMGInputs<IdxT>> {
 public:
  AnnMGTest()
    : stream_(resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnMGInputs<IdxT>>::GetParam()),
      d_index_dataset(0, stream_),
      d_query_dataset(0, stream_),
      h_index_dataset(0),
      h_query_dataset(0)
  {
  }

  void testAnnMG()
  {
    size_t queries_size = ps.num_queries * ps.k;
    std::vector<T> distances_ivfflat(queries_size);
    std::vector<T> distances_naive(queries_size);
    std::vector<IdxT> indices_ivfflat(queries_size);
    std::vector<IdxT> indices_naive(queries_size);

    {
      rmm::device_uvector<T> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      naive_knn<T, DataT, IdxT>(handle_,
                                distances_naive_dev.data(),
                                indices_naive_dev.data(),
                                d_query_dataset.data(),
                                d_index_dataset.data(),
                                ps.num_queries,
                                ps.num_db_vecs,
                                ps.dim,
                                ps.k,
                                ps.metric);
      update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      resource::sync_stream(handle_);
    }

    {
      rmm::device_uvector<T> distances_ivfflat_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_ivfflat_dev(queries_size, stream_);

      std::vector<int> device_ids{0, 1};
      raft::neighbors::mg::dist_mode mode = SHARDING;
      ivf_flat::index_params index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.adaptive_centers         = ps.adaptive_centers;
      index_params.add_data_on_build        = false;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;
      auto index_dataset_view = raft::make_host_matrix_view<const DataT, IdxT, row_major>(h_index_dataset.data(), ps.num_db_vecs, ps.dim);
      auto index = raft::neighbors::mg::build<DataT, IdxT>(device_ids, mode, index_params, index_dataset_view);

      update_host(distances_ivfflat.data(), distances_ivfflat_dev.data(), queries_size, stream_);
      update_host(indices_ivfflat.data(), indices_ivfflat_dev.data(), queries_size, stream_);
      resource::sync_stream(handle_);
    }

    double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
    ASSERT_TRUE(eval_neighbours(indices_naive,
                                indices_ivfflat,
                                distances_naive,
                                distances_ivfflat,
                                ps.num_queries,
                                ps.k,
                                0.001,
                                min_recall));
  }

  void SetUp() override
  {
    d_index_dataset.resize(ps.num_db_vecs * ps.dim, stream_);
    d_query_dataset.resize(ps.num_queries * ps.dim, stream_);
    h_index_dataset.resize(ps.num_db_vecs * ps.dim);
    h_query_dataset.resize(ps.num_queries * ps.dim);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::uniform(
        handle_, r, d_index_dataset.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::uniform(
        handle_, r, d_query_dataset.data(), ps.num_queries * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, d_index_dataset.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20));
      raft::random::uniformInt(
        handle_, r, d_query_dataset.data(), ps.num_queries * ps.dim, DataT(1), DataT(20));
    }

    raft::copy(h_index_dataset.data(),
               d_index_dataset.data(),
               d_index_dataset.size(),
               resource::get_cuda_stream(handle_));
    raft::copy(h_query_dataset.data(),
               d_query_dataset.data(),
               d_query_dataset.size(),
               resource::get_cuda_stream(handle_));
    resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    resource::sync_stream(handle_);
    h_index_dataset.clear();
    h_query_dataset.clear();
    d_index_dataset.resize(0, stream_);
    d_query_dataset.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnMGInputs<IdxT> ps;
  std::vector<DataT> h_index_dataset;
  std::vector<DataT> h_query_dataset;
  rmm::device_uvector<DataT> d_index_dataset;
  rmm::device_uvector<DataT> d_query_dataset;
};

const std::vector<AnnMGInputs<uint32_t>> inputs = {
  {1000, 10000, 1, 16, 40, 1024, raft::distance::DistanceType::L2Expanded, true},
};}