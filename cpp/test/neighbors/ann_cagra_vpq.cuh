/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/add.cuh>
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/cagra_serialize.cuh>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/sample_filter.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>

#include <raft_internal/neighbors/naive_knn.cuh>

#include <rmm/device_buffer.hpp>

#include <cuda_fp16.h>
#include <thrust/sequence.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
template <class T>
void GenerateDataset(T* const dataset_ptr,
                     T* const query_ptr,
                     const std::size_t dataset_size,
                     const std::size_t query_size,
                     const std::size_t dim,
                     const std::size_t num_centers,
                     cudaStream_t cuda_stream)
{
  auto center_list  = raft::make_host_matrix<T, int64_t>(num_centers, dim);
  auto host_dataset = raft::make_host_matrix<T, int64_t>(std::max(dataset_size, query_size), dim);

  std::normal_distribution<T> dist(0, 1);
  std::mt19937 mt(0);
  for (std::size_t i = 0; i < center_list.size(); i++) {
    center_list.data_handle()[i] = dist(mt);
  }

  std::uniform_int_distribution<std::size_t> i_dist(0, num_centers - 1);
  for (std::size_t i = 0; i < dataset_size; i++) {
    const auto center_index = i_dist(mt);
    for (std::size_t j = 0; j < dim; j++) {
      host_dataset.data_handle()[i * dim + j] =
        center_list.data_handle()[center_index + j] + dist(mt) * 1e-1;
    }
  }
  raft::copy(dataset_ptr, host_dataset.data_handle(), dataset_size * dim, cuda_stream);

  for (std::size_t i = 0; i < query_size; i++) {
    const auto center_index = i_dist(mt);
    for (std::size_t j = 0; j < dim; j++) {
      host_dataset.data_handle()[i * dim + j] =
        center_list.data_handle()[center_index + j] + dist(mt) * 1e-1;
    }
  }
  raft::copy(query_ptr, host_dataset.data_handle(), query_size * dim, cuda_stream);
}
}  // namespace

namespace raft::neighbors::cagra {
struct AnnCagraVpqInputs {
  int n_queries;
  int n_rows;
  int dim;
  int k;
  int pq_len;
  int pq_bits;
  graph_build_algo build_algo;
  search_algo algo;
  int max_queries;
  int team_size;
  int itopk_size;
  int search_width;
  raft::distance::DistanceType metric;
  bool host_dataset;
  bool include_serialized_dataset;
  // std::optional<double>
  double min_recall;  // = std::nullopt;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnCagraVpqInputs& p)
{
  std::vector<std::string> algo       = {"single-cta", "multi_cta", "multi_kernel", "auto"};
  std::vector<std::string> build_algo = {"IVF_PQ", "NN_DESCENT"};
  os << "{n_queries=" << p.n_queries << ", dataset shape=" << p.n_rows << "x" << p.dim
     << ", k=" << p.k << ", pq_bits=" << p.pq_bits << ", pq_len=" << p.pq_len << ", "
     << algo.at((int)p.algo) << ", max_queries=" << p.max_queries << ", itopk_size=" << p.itopk_size
     << ", search_width=" << p.search_width << ", metric=" << static_cast<int>(p.metric)
     << (p.host_dataset ? ", host" : ", device")
     << ", build_algo=" << build_algo.at((int)p.build_algo) << '}' << std::endl;
  return os;
}

template <typename DistanceT, typename DataT, typename IdxT>
class AnnCagraVpqTest : public ::testing::TestWithParam<AnnCagraVpqInputs> {
 public:
  AnnCagraVpqTest()
    : stream_(resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnCagraVpqInputs>::GetParam()),
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
      naive_knn<DistanceT, DataT, IdxT>(handle_,
                                        distances_naive_dev.data(),
                                        indices_naive_dev.data(),
                                        search_queries.data(),
                                        database.data(),
                                        ps.n_queries,
                                        ps.n_rows,
                                        ps.dim,
                                        ps.k,
                                        ps.metric);
      update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      resource::sync_stream(handle_);
    }

    const auto vpq_k = ps.k * 4;
    {
      rmm::device_uvector<DistanceT> distances_dev(vpq_k * ps.n_queries, stream_);
      rmm::device_uvector<IdxT> indices_dev(vpq_k * ps.n_queries, stream_);

      {
        if ((ps.dim % ps.pq_len) != 0) {
          // TODO: remove this requirement in the algorithm.
          GTEST_SKIP() << "(TODO) At the moment dim, (" << ps.dim
                       << ") must be a multiple of pq_len (" << ps.pq_len << ")";
        }
        cagra::index_params index_params;
        index_params.compression = vpq_params{.pq_bits = static_cast<uint32_t>(ps.pq_bits),
                                              .pq_dim  = static_cast<uint32_t>(ps.dim / ps.pq_len)};
        index_params.metric = ps.metric;  // Note: currently ony the cagra::index_params metric is
                                          // not used for knn_graph building.
        index_params.build_algo = ps.build_algo;
        cagra::search_params search_params;
        search_params.algo        = ps.algo;
        search_params.max_queries = ps.max_queries;
        search_params.team_size   = ps.team_size;
        search_params.itopk_size  = ps.itopk_size;

        auto database_view =
          raft::make_device_matrix_view<const DataT, int64_t>(database.data(), ps.n_rows, ps.dim);

        {
          cagra::index<DataT, IdxT> index(handle_);
          if (ps.host_dataset) {
            auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
            raft::copy(database_host.data_handle(), database.data(), database.size(), stream_);
            auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
              database_host.data_handle(), ps.n_rows, ps.dim);
            index = cagra::build<DataT, IdxT>(handle_, index_params, database_host_view);
          } else {
            index = cagra::build<DataT, IdxT>(handle_, index_params, database_view);
          };
          cagra::serialize(handle_, "cagra_index", index, ps.include_serialized_dataset);
        }

        auto index = cagra::deserialize<DataT, IdxT>(handle_, "cagra_index");
        if (!ps.include_serialized_dataset) { index.update_dataset(handle_, database_view); }

        // CAGRA-Q sanity check: we've built the right index type
        auto* vpq_dataset =
          dynamic_cast<const raft::neighbors::vpq_dataset<half, int64_t>*>(&index.data());
        EXPECT_NE(vpq_dataset, nullptr)
          << "Expected VPQ dataset, because we're testing CAGRA-Q here.";

        auto search_queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_out_view =
          raft::make_device_matrix_view<IdxT, int64_t>(indices_dev.data(), ps.n_queries, vpq_k);
        auto dists_out_view = raft::make_device_matrix_view<DistanceT, int64_t>(
          distances_dev.data(), ps.n_queries, vpq_k);

        cagra::search(
          handle_, search_params, index, search_queries_view, indices_out_view, dists_out_view);

        {
          auto host_dataset = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
          raft::copy(
            host_dataset.data_handle(), (const DataT*)database.data(), ps.n_rows * ps.dim, stream_);

          auto host_queries = raft::make_host_matrix<DataT, int64_t>(ps.n_queries, ps.dim);
          raft::copy(host_queries.data_handle(),
                     (const DataT*)search_queries_view.data_handle(),
                     ps.n_queries * ps.dim,
                     stream_);

          auto host_index_candidate = raft::make_host_matrix<IdxT, int64_t>(ps.n_queries, vpq_k);
          raft::copy(host_index_candidate.data_handle(),
                     indices_out_view.data_handle(),
                     ps.n_queries * vpq_k,
                     stream_);

          auto host_indices_Cagra_view =
            raft::make_host_matrix_view<IdxT, int64_t>(indices_Cagra.data(), ps.n_queries, ps.k);

          auto host_dists_Cagra_view =
            raft::make_host_matrix_view<float, int64_t>(distances_Cagra.data(), ps.n_queries, ps.k);

          resource::sync_stream(handle_);

          raft::neighbors::refine(handle_,
                                  raft::make_const_mdspan(host_dataset.view()),
                                  raft::make_const_mdspan(host_queries.view()),
                                  raft::make_const_mdspan(host_index_candidate.view()),
                                  host_indices_Cagra_view,
                                  host_dists_Cagra_view,
                                  ps.metric);

          raft::copy(indices_dev.data(),
                     host_indices_Cagra_view.data_handle(),
                     ps.k * ps.n_queries,
                     stream_);
          raft::copy(distances_dev.data(),
                     host_dists_Cagra_view.data_handle(),
                     ps.k * ps.n_queries,
                     stream_);
          resource::sync_stream(handle_);
        }
      }

      double min_recall = ps.min_recall;
      EXPECT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.003,
                                  min_recall));
      EXPECT_TRUE(eval_distances(handle_,
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
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    search_queries.resize(ps.n_queries * ps.dim, stream_);
    GenerateDataset(database.data(),
                    search_queries.data(),
                    ps.n_rows,
                    ps.n_queries,
                    ps.dim,
                    static_cast<std::size_t>(std::sqrt(ps.n_rows)),
                    stream_);
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
  AnnCagraVpqInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

const std::vector<AnnCagraVpqInputs> vpq_inputs = raft::util::itertools::product<AnnCagraVpqInputs>(
  {100},                                              // n_queries
  {1000, 10000},                                      // n_rows
  {128, 132, 192, 256, 512, 768},                     // dim
  {8, 12},                                            // k
  {2, 4},                                             // pq_len
  {8},                                                // pq_bits
  {graph_build_algo::NN_DESCENT},                     // build_algo
  {search_algo::SINGLE_CTA, search_algo::MULTI_CTA},  // algo
  {0},                                                // max_queries
  {0},                                                // team_size
  {512},                                              // itopk_size
  {1},                                                // search_width
  {raft::distance::DistanceType::L2Expanded},         // metric
  {false},                                            // host_dataset
  {true},                                             // include_serialized_dataset
  {0.8}                                               // min_recall
);

}  // namespace raft::neighbors::cagra
