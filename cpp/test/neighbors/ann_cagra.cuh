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
#include <raft/core/resource/cuda_stream.hpp>

#include <raft_internal/neighbors/naive_knn.cuh>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
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
namespace {
// For sort_knn_graph test
template <typename IdxT>
void RandomSuffle(raft::host_matrix_view<IdxT, IdxT> index)
{
  for (IdxT i = 0; i < index.extent(0); i++) {
    uint64_t rand       = i;
    IdxT* const row_ptr = index.data_handle() + i * index.extent(1);
    for (unsigned j = 0; j < index.extent(1); j++) {
      // Swap two indices at random
      rand          = raft::neighbors::experimental::cagra::detail::device::xorshift64(rand);
      const auto i0 = rand % index.extent(1);
      rand          = raft::neighbors::experimental::cagra::detail::device::xorshift64(rand);
      const auto i1 = rand % index.extent(1);

      const auto tmp = row_ptr[i0];
      row_ptr[i0]    = row_ptr[i1];
      row_ptr[i1]    = tmp;
    }
  }
}

template <typename DistanceT, typename DatatT, typename IdxT>
testing::AssertionResult CheckOrder(raft::host_matrix_view<IdxT, IdxT> index_test,
                                    raft::host_matrix_view<DatatT, IdxT> dataset)
{
  for (IdxT i = 0; i < index_test.extent(0); i++) {
    const DatatT* const base_vec = dataset.data_handle() + i * dataset.extent(1);
    const IdxT* const index_row  = index_test.data_handle() + i * index_test.extent(1);
    DistanceT prev_distance      = 0;
    for (unsigned j = 0; j < index_test.extent(1) - 1; j++) {
      const DatatT* const target_vec = dataset.data_handle() + index_row[j] * dataset.extent(1);
      DistanceT distance             = 0;
      for (unsigned l = 0; l < dataset.extent(1); l++) {
        const auto diff =
          static_cast<DistanceT>(target_vec[l]) - static_cast<DistanceT>(base_vec[l]);
        distance += diff * diff;
      }
      if (prev_distance > distance) {
        return testing::AssertionFailure()
               << "Wrong index order (row = " << i << ", neighbor_id = " << j
               << "). (distance[neighbor_id-1] = " << prev_distance
               << "should be larger than distance[neighbor_id] = " << distance << ")";
      }
      prev_distance = distance;
    }
  }
  return testing::AssertionSuccess();
}

// Generate dataset to ensure no rounding error occurs in the norm computation of any two vectors.
// When testing the CAGRA index sorting function, rounding errors can affect the norm and alter the
// order of the index. To ensure the accuracy of the test, we utilize the dataset. The generation
// method is based on the error-free transformation (EFT) method.
__global__ void GenerateRoundingErrorFreeDataset_kernel(float* const ptr,
                                                        const uint32_t size,
                                                        const uint32_t resolution)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) { return; }

  const float u32 = *reinterpret_cast<const uint32_t*>(ptr + tid);
  ptr[tid]        = u32 / resolution;
}

void GenerateRoundingErrorFreeDataset(float* const ptr,
                                      const uint32_t n_row,
                                      const uint32_t dim,
                                      raft::random::Rng& rng,
                                      cudaStream_t cuda_stream)
{
  const uint32_t size       = n_row * dim;
  const uint32_t block_size = 256;
  const uint32_t grid_size  = (size + block_size - 1) / block_size;

  const uint32_t resolution = 1u << static_cast<unsigned>(std::floor((24 - std::log2(dim)) / 2));
  rng.uniformInt(reinterpret_cast<uint32_t*>(ptr), size, 0u, resolution - 1, cuda_stream);

  GenerateRoundingErrorFreeDataset_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
    ptr, size, resolution);
}
}  // namespace

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
    : stream_(resource::get_cuda_stream(handle_)),
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
        resource::sync_stream(handle_);
      }
      // for (int i = 0; i < min(ps.n_queries, 10); i++) {
      //   //  std::cout << "query " << i << std::end;
      //   print_vector("T", indices_naive.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("C", indices_Cagra.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("T", distances_naive.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("C", distances_Cagra.data() + i * ps.k, ps.k, std::cout);
      // }
      double min_recall = ps.min_recall;
      EXPECT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.001,
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
    raft::random::Rng r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      r.normal(database.data(), ps.n_rows * ps.dim, DataT(0.1), DataT(2.0), stream_);
      r.normal(search_queries.data(), ps.n_queries * ps.dim, DataT(0.1), DataT(2.0), stream_);
    } else {
      r.uniformInt(database.data(), ps.n_rows * ps.dim, DataT(1), DataT(20), stream_);
      r.uniformInt(search_queries.data(), ps.n_queries * ps.dim, DataT(1), DataT(20), stream_);
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
  AnnCagraInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

template <typename DistanceT, typename DataT, typename IdxT>
class AnnCagraSortTest : public ::testing::TestWithParam<AnnCagraInputs> {
 public:
  AnnCagraSortTest()
    : ps(::testing::TestWithParam<AnnCagraInputs>::GetParam()), database(0, handle_.get_stream())
  {
  }

 protected:
  void testCagraSort()
  {
    {
      // Step 1: Build a sorted KNN graph by CAGRA knn build
      auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
        (const DataT*)database.data(), ps.n_rows, ps.dim);
      auto database_host = raft::make_host_matrix<DataT, IdxT>(ps.n_rows, ps.dim);
      raft::copy(
        database_host.data_handle(), database.data(), database.size(), handle_.get_stream());
      auto database_host_view = raft::make_host_matrix_view<const DataT, IdxT>(
        (const DataT*)database_host.data_handle(), ps.n_rows, ps.dim);

      cagra::index_params index_params;
      auto knn_graph =
        raft::make_host_matrix<IdxT, IdxT>(ps.n_rows, index_params.intermediate_graph_degree);

      if (ps.host_dataset) {
        cagra::build_knn_graph<DataT, IdxT>(handle_, database_host_view, knn_graph.view());
      } else {
        cagra::build_knn_graph<DataT, IdxT>(handle_, database_view, knn_graph.view());
      };

      handle_.sync_stream();
      ASSERT_TRUE(CheckOrder<DistanceT>(knn_graph.view(), database_host.view()));

      RandomSuffle(knn_graph.view());

      cagra::sort_knn_graph(handle_, database_view, knn_graph.view());
      handle_.sync_stream();

      ASSERT_TRUE(CheckOrder<DistanceT>(knn_graph.view(), database_host.view()));
    }
  }

  void SetUp() override
  {
    std::cout << "Resizing database: " << ps.n_rows * ps.dim << std::endl;
    database.resize(((size_t)ps.n_rows) * ps.dim, handle_.get_stream());
    std::cout << "Done.\nRuning rng" << std::endl;
    raft::random::Rng r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      GenerateRoundingErrorFreeDataset(database.data(), ps.n_rows, ps.dim, r, handle_.get_stream());
    } else {
      r.uniformInt(database.data(), ps.n_rows * ps.dim, DataT(1), DataT(20), handle_.get_stream());
    }
    handle_.sync_stream();
  }

  void TearDown() override
  {
    handle_.sync_stream();
    database.resize(0, handle_.get_stream());
  }

 private:
  raft::device_resources handle_;
  AnnCagraInputs ps;
  rmm::device_uvector<DataT> database;
};

inline std::vector<AnnCagraInputs> generate_inputs()
{
  // Todo(tfeher): MULTI_CTA tests a bug, consider disabling that mode.
  // TODO(tfeher): test MULTI_CTA kernel with num_Parents>1 to allow multiple CTA per queries
  std::vector<AnnCagraInputs> inputs = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {1, 8, 17},
    {1, 16},          // k
    {search_algo::SINGLE_CTA, search_algo::MULTI_CTA, search_algo::MULTI_KERNEL},
    {0, 1, 10, 100},  // query size
    {0},
    {256},
    {1},
    {raft::distance::DistanceType::L2Expanded},
    {false},
    {0.995});

  auto inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {1, 3, 5, 7, 8, 17, 64, 128, 137, 192, 256, 512, 619, 1024},  // dim
    {16},                                                         // k
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
                                                   {32},
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
                                                   {32},
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
