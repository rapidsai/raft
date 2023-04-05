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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/cagra_serialize.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <thrust/sequence.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace raft::neighbors::experimental::cagra {

struct AnnCagraInputs {
  int n_queries;
  int n_rows;
  int dim;
  int k;
  search_algo algo;
  int max_queries;
  int team_size;
  int itopk_size;
  int num_parents;
  raft::distance::DistanceType metric;
  bool host_dataset;
  // std::optional<double>
  double min_recall;  // = std::nullopt;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnCagraInputs& p)
{
  std::vector<std::string> algo = {"single-cta", "multi_cta", "multi_kernel", "auto"};
  os << "{n_queries=" << p.n_queries << ", dataset shape=" << p.n_rows << "x" << p.dim
     << ", k=" << p.k << ", " << algo.at((int)p.algo) << ", max_queries=" << p.max_queries
     << ", itopk_size=" << p.itopk_size << ", num_parents=" << p.num_parents
     << ", metric=" << static_cast<int>(p.metric) << (p.host_dataset ? ", host" : ", device") << '}'
     << std::endl;
  return os;
}

template <typename DistanceT, typename DataT, typename IdxT>
class AnnCagraTest : public ::testing::TestWithParam<AnnCagraInputs> {
 public:
  AnnCagraTest()
    : stream_(handle_.get_stream()),
      ps(::testing::TestWithParam<AnnCagraInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  void testCagra()
  {
    size_t queries_size = ps.n_queries * ps.k;
    std::vector<IdxT> indices_Cagra(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_Cagra(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      naive_knn<DistanceT, DataT, IdxT>(distances_naive_dev.data(),
                                        indices_naive_dev.data(),
                                        search_queries.data(),
                                        database.data(),
                                        ps.n_queries,
                                        ps.n_rows,
                                        ps.dim,
                                        ps.k,
                                        ps.metric,
                                        stream_);
      update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      handle_.sync_stream(stream_);
    }

    {
      rmm::device_uvector<DistanceT> distances_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_dev(queries_size, stream_);

      {
        cagra::index_params index_params;
        index_params.metric = ps.metric;  // Note: currently ony the cagra::index_params metric is
                                          // not used for knn_graph building.
        cagra::search_params search_params;
        search_params.algo        = ps.algo;
        search_params.max_queries = ps.max_queries;
        search_params.team_size   = ps.team_size;

        auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
          (const DataT*)database.data(), ps.n_rows, ps.dim);

        {
          cagra::index<DataT, IdxT> index(handle_);
          if (ps.host_dataset) {
            auto database_host = raft::make_host_matrix<DataT, IdxT>(ps.n_rows, ps.dim);
            raft::copy(database_host.data_handle(), database.data(), database.size(), stream_);
            auto database_host_view = raft::make_host_matrix_view<const DataT, IdxT>(
              (const DataT*)database_host.data_handle(), ps.n_rows, ps.dim);
            index = cagra::build<DataT, IdxT>(handle_, index_params, database_host_view);
          } else {
            index = cagra::build<DataT, IdxT>(handle_, index_params, database_view);
          };
          cagra::serialize(handle_, "cagra_index", index);
        }
        auto index = cagra::deserialize<DataT, IdxT>(handle_, "cagra_index");

        auto search_queries_view = raft::make_device_matrix_view<const DataT, IdxT>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_out_view =
          raft::make_device_matrix_view<IdxT, IdxT>(indices_dev.data(), ps.n_queries, ps.k);
        auto dists_out_view =
          raft::make_device_matrix_view<DistanceT, IdxT>(distances_dev.data(), ps.n_queries, ps.k);

        cagra::search(
          handle_, search_params, index, search_queries_view, indices_out_view, dists_out_view);

        update_host(distances_Cagra.data(), distances_dev.data(), queries_size, stream_);
        update_host(indices_Cagra.data(), indices_dev.data(), queries_size, stream_);
        handle_.sync_stream(stream_);
      }
      // for (int i = 0; i < ps.n_queries; i++) {
      //   //  std::cout << "query " << i << std::end;
      //   print_vector("T", indices_naive.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("C", indices_Cagra.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("T", distances_naive.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("C", distances_Cagra.data() + i * ps.k, ps.k, std::cout);
      // }
      double min_recall = ps.min_recall;
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      ASSERT_TRUE(eval_distances(handle_,
                                 database.data(),
                                 search_queries.data(),
                                 indices_dev.data(),
                                 distances_dev.data(),
                                 ps.n_rows,
                                 ps.dim,
                                 ps.n_queries,
                                 ps.k,
                                 ps.metric,
                                 1.0e-4));
    }
  }

  void SetUp() override
  {
    std::cout << "Resizing database: " << ps.n_rows * ps.dim << std::endl;
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    std::cout << "Done.\nResizing queries" << std::endl;
    search_queries.resize(ps.n_queries * ps.dim, stream_);
    std::cout << "Done.\nRuning rng" << std::endl;
    raft::random::Rng r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      r.uniform(database.data(), ps.n_rows * ps.dim, DataT(0.1), DataT(2.0), stream_);
      r.uniform(search_queries.data(), ps.n_queries * ps.dim, DataT(0.1), DataT(2.0), stream_);
    } else {
      r.uniformInt(database.data(), ps.n_rows * ps.dim, DataT(1), DataT(20), stream_);
      r.uniformInt(search_queries.data(), ps.n_queries * ps.dim, DataT(1), DataT(20), stream_);
    }
    handle_.sync_stream(stream_);
  }

  void TearDown() override
  {
    handle_.sync_stream(stream_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::device_resources handle_;
  rmm::cuda_stream_view stream_;
  AnnCagraInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

inline std::vector<AnnCagraInputs> generate_inputs()
{
  // Todo(tfeher): MULTI_CTA tests a bug, consider disabling that mode.
  std::vector<AnnCagraInputs> inputs = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {8},
    {1, 16, 33},  // k
    {search_algo::SINGLE_CTA, search_algo::MULTI_KERNEL},
    {1, 10, 100},  // query size
    {0},
    {64},
    {1},
    {raft::distance::DistanceType::L2Expanded},
    {false},
    {0.995});

  auto inputs2 =
    raft::util::itertools::product<AnnCagraInputs>({100},
                                                   {1000},
                                                   {2, 4, 8, 64, 128, 196, 256, 512, 1024},  // dim
                                                   {16},
                                                   {search_algo::AUTO},
                                                   {10},
                                                   {0},
                                                   {64},
                                                   {1},
                                                   {raft::distance::DistanceType::L2Expanded},
                                                   {false},
                                                   {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());
  inputs2 =
    raft::util::itertools::product<AnnCagraInputs>({100},
                                                   {1000},
                                                   {64},
                                                   {16},
                                                   {search_algo::AUTO},
                                                   {10},
                                                   {0, 4, 8, 16, 32},  // team_size
                                                   {64},
                                                   {1},
                                                   {raft::distance::DistanceType::L2Expanded},
                                                   {false},
                                                   {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 =
    raft::util::itertools::product<AnnCagraInputs>({100},
                                                   {1000},
                                                   {64},
                                                   {16},
                                                   {search_algo::AUTO},
                                                   {10},
                                                   {0},  // team_size
                                                   {32, 64, 128, 256, 512, 768},
                                                   {1},
                                                   {raft::distance::DistanceType::L2Expanded},
                                                   {false},
                                                   {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 =
    raft::util::itertools::product<AnnCagraInputs>({100},
                                                   {10000, 20000},
                                                   {30},
                                                   {10},
                                                   {search_algo::AUTO},
                                                   {10},
                                                   {0},  // team_size
                                                   {64},
                                                   {1},
                                                   {raft::distance::DistanceType::L2Expanded},
                                                   {false, true},
                                                   {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 =
    raft::util::itertools::product<AnnCagraInputs>({100},
                                                   {10000, 20000},
                                                   {30},
                                                   {10},
                                                   {search_algo::AUTO},
                                                   {10},
                                                   {0},  // team_size
                                                   {64},
                                                   {1},
                                                   {raft::distance::DistanceType::L2Expanded},
                                                   {false, true},
                                                   {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  return inputs;
}

const std::vector<AnnCagraInputs> inputs = generate_inputs();

}  // namespace raft::neighbors::experimental::cagra