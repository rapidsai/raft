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
#include "knn_utils.cuh"
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/map.cuh>
#include <raft/neighbors/brute_force_types.hpp>
#include <raft/neighbors/ivf_list.hpp>
#include <raft/neighbors/sample_filter.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/fast_int_div.cuh>
#include <thrust/functional.h>

#include <raft_internal/neighbors/naive_knn.cuh>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/neighbors/brute_force.cuh>
#include <raft/neighbors/brute_force_serialize.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/mean.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <rmm/device_uvector.hpp>
#include <thrust/sequence.h>

#include <cstddef>
#include <iostream>
#include <vector>

namespace raft::neighbors::brute_force {

template <typename IdxT>
struct AnnBruteForceInputs {
  IdxT num_queries;
  IdxT num_db_vecs;
  IdxT dim;
  IdxT k;
  raft::distance::DistanceType metric;
  bool host_dataset;
};

template <typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const AnnBruteForceInputs<IdxT>& p)
{
  os << "{ " << p.num_queries << ", " << p.num_db_vecs << ", " << p.dim << ", " << p.k << ", "
     << static_cast<int>(p.metric) << ", " << p.host_dataset << '}' << std::endl;
  return os;
}

template <typename T, typename DataT, typename IdxT>
class AnnBruteForceTest : public ::testing::TestWithParam<AnnBruteForceInputs<IdxT>> {
 public:
  AnnBruteForceTest()
    : stream_(resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnBruteForceInputs<IdxT>>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

  void testBruteForce()
  {
    size_t queries_size = ps.num_queries * ps.k;

    rmm::device_uvector<T> distances_naive_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
    naive_knn<T, DataT, IdxT>(handle_,
                              distances_naive_dev.data(),
                              indices_naive_dev.data(),
                              search_queries.data(),
                              database.data(),
                              ps.num_queries,
                              ps.num_db_vecs,
                              ps.dim,
                              ps.k,
                              ps.metric);
    resource::sync_stream(handle_);

    {
      // Require exact result for brute force
      rmm::device_uvector<T> distances_bruteforce_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_bruteforce_dev(queries_size, stream_);
      brute_force::index_params index_params{};
      brute_force::search_params search_params{};
      index_params.metric     = ps.metric;
      index_params.metric_arg = 0;

      auto device_dataset = std::optional<raft::device_matrix<DataT, IdxT>>{};
      auto idx            = [this, &index_params]() {
        if (ps.host_dataset) {
          auto host_database = raft::make_host_matrix<DataT, IdxT>(ps.num_db_vecs, ps.dim);
          raft::copy(
            host_database.data_handle(), database.data(), ps.num_db_vecs * ps.dim, stream_);
          return brute_force::build(
            handle_, index_params, raft::make_const_mdspan(host_database.view()));
        } else {
          auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
            (const DataT*)database.data(), ps.num_db_vecs, ps.dim);
          return brute_force::build(handle_, index_params, database_view);
        }
      }();

      auto search_queries_view = raft::make_device_matrix_view<const DataT, IdxT>(
        search_queries.data(), ps.num_queries, ps.dim);
      auto indices_out_view = raft::make_device_matrix_view<IdxT, IdxT>(
        indices_bruteforce_dev.data(), ps.num_queries, ps.k);
      auto dists_out_view = raft::make_device_matrix_view<T, IdxT>(
        distances_bruteforce_dev.data(), ps.num_queries, ps.k);
      brute_force::serialize(handle_, std::string{"brute_force_index"}, idx);

      auto index_loaded =
        brute_force::deserialize<DataT>(handle_, std::string{"brute_force_index"});
      ASSERT_EQ(idx.size(), index_loaded.size());

      brute_force::search(handle_,
                          search_params,
                          index_loaded,
                          search_queries_view,
                          indices_out_view,
                          dists_out_view);

      resource::sync_stream(handle_);

      ASSERT_TRUE(raft::spatial::knn::devArrMatchKnnPair(indices_naive_dev.data(),
                                                         indices_bruteforce_dev.data(),
                                                         distances_naive_dev.data(),
                                                         distances_bruteforce_dev.data(),
                                                         ps.num_queries,
                                                         ps.k,
                                                         0.001f,
                                                         stream_,
                                                         true));
      brute_force::serialize(handle_, std::string{"brute_force_index"}, idx, false);
      index_loaded = brute_force::deserialize<DataT>(handle_, std::string{"brute_force_index"});
      index_loaded.update_dataset(handle_, idx.dataset());
      ASSERT_EQ(idx.size(), index_loaded.size());

      brute_force::search(handle_,
                          search_params,
                          index_loaded,
                          search_queries_view,
                          indices_out_view,
                          dists_out_view);

      resource::sync_stream(handle_);

      ASSERT_TRUE(raft::spatial::knn::devArrMatchKnnPair(indices_naive_dev.data(),
                                                         indices_bruteforce_dev.data(),
                                                         distances_naive_dev.data(),
                                                         distances_bruteforce_dev.data(),
                                                         ps.num_queries,
                                                         ps.k,
                                                         0.001f,
                                                         stream_,
                                                         true));
    }
  }

  void SetUp() override
  {
    database.resize(ps.num_db_vecs * ps.dim, stream_);
    search_queries.resize(ps.num_queries * ps.dim, stream_);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::uniform(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::uniform(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20));
      raft::random::uniformInt(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(1), DataT(20));
    }
    resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnBruteForceInputs<IdxT> ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

const std::vector<AnnBruteForceInputs<int64_t>> inputs = {
  // test various dims (aligned and not aligned to vector sizes)
  {1000, 10000, 1, 16, raft::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 2, 16, raft::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 3, 16, raft::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 4, 16, raft::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 5, 16, raft::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 8, 16, raft::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 5, 16, raft::distance::DistanceType::L2SqrtExpanded, true},
  {1000, 10000, 8, 16, raft::distance::DistanceType::L2SqrtExpanded, true},

  // test dims that do not fit into kernel shared memory limits
  {1000, 10000, 2048, 16, raft::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 2049, 16, raft::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 2050, 16, raft::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 2051, 16, raft::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 2052, 16, raft::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 2053, 16, raft::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 2056, 16, raft::distance::DistanceType::L2Expanded, true},

  // host input data
  {1000, 10000, 16, 10, raft::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 16, 10, raft::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 16, 10, raft::distance::DistanceType::L2Expanded, false},
  {100, 10000, 16, 10, raft::distance::DistanceType::L2Expanded, false},
  {20, 100000, 16, 10, raft::distance::DistanceType::L2Expanded, false},
  {1000, 100000, 16, 10, raft::distance::DistanceType::L2Expanded, false},
  {10000, 131072, 8, 10, raft::distance::DistanceType::L2Expanded, false},

  {1000, 10000, 16, 10, raft::distance::DistanceType::InnerProduct, false}};
}  // namespace raft::neighbors::brute_force
