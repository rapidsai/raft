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
#pragma once

#undef RAFT_EXPLICIT_INSTANTIATE_ONLY  // Search with filter instantiation

#include "../test_utils.cuh"
#include "ann_utils.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/normalize.cuh>
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/cagra_serialize.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/sample_filter.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/itertools.hpp>

#include <raft_internal/neighbors/naive_knn.cuh>

#include <rmm/device_buffer.hpp>

#include <cuda_fp16.h>
#include <thrust/sequence.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace raft::neighbors::cagra {
namespace {

/* A filter that excludes all indices below `offset`. */
struct test_cagra_sample_filter {
  static constexpr unsigned offset = 300;
  inline _RAFT_HOST_DEVICE auto operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix) const
  {
    return sample_ix >= offset;
  }
};

// For sort_knn_graph test
template <typename IdxT>
void RandomSuffle(raft::host_matrix_view<IdxT, int64_t> index)
{
  for (IdxT i = 0; i < index.extent(0); i++) {
    uint64_t rand       = i;
    IdxT* const row_ptr = index.data_handle() + i * index.extent(1);
    for (unsigned j = 0; j < index.extent(1); j++) {
      // Swap two indices at random
      rand          = raft::neighbors::cagra::detail::device::xorshift64(rand);
      const auto i0 = rand % index.extent(1);
      rand          = raft::neighbors::cagra::detail::device::xorshift64(rand);
      const auto i1 = rand % index.extent(1);

      const auto tmp = row_ptr[i0];
      row_ptr[i0]    = row_ptr[i1];
      row_ptr[i1]    = tmp;
    }
  }
}

template <typename DistanceT, typename DatatT, typename IdxT>
testing::AssertionResult CheckOrder(raft::host_matrix_view<IdxT, int64_t> index_test,
                                    raft::host_matrix_view<DatatT, int64_t> dataset,
                                    raft::distance::DistanceType metric)
{
  for (IdxT i = 0; i < index_test.extent(0); i++) {
    const DatatT* const base_vec = dataset.data_handle() + i * dataset.extent(1);
    const IdxT* const index_row  = index_test.data_handle() + i * index_test.extent(1);
    DistanceT prev_distance      = metric == raft::distance::DistanceType::L2Expanded
                                     ? 0
                                     : std::numeric_limits<DistanceT>::max();
    for (unsigned j = 0; j < index_test.extent(1) - 1; j++) {
      const DatatT* const target_vec = dataset.data_handle() + index_row[j] * dataset.extent(1);
      DistanceT distance             = 0;
      switch (metric) {
        case raft::distance::DistanceType::L2Expanded:
          for (unsigned l = 0; l < dataset.extent(1); l++) {
            const auto diff =
              static_cast<DistanceT>(target_vec[l]) - static_cast<DistanceT>(base_vec[l]);
            distance += diff * diff;
          }
          if (prev_distance > distance) {
            return testing::AssertionFailure()
                   << "Wrong index order (row = " << i << ", neighbor_id = " << j
                   << "). (distance[neighbor_id-1] = " << prev_distance
                   << "should be lesser than distance[neighbor_id] = " << distance << ")";
          }
          break;
        case raft::distance::DistanceType::InnerProduct:
          for (unsigned l = 0; l < dataset.extent(1); l++) {
            const auto prod =
              static_cast<DistanceT>(target_vec[l]) * static_cast<DistanceT>(base_vec[l]);
            distance += prod;
          }
          if (prev_distance < distance) {
            return testing::AssertionFailure()
                   << "Wrong index order (row = " << i << ", neighbor_id = " << j
                   << "). (distance[neighbor_id-1] = " << prev_distance
                   << "should be greater than distance[neighbor_id] = " << distance << ")";
          }
          break;
        default:
          return testing::AssertionFailure()
                 << "Distance metric " << metric
                 << " not supported. Only L2Expanded and InnerProduct are supported";
      }
      prev_distance = distance;
    }
  }
  return testing::AssertionSuccess();
}

template <typename T>
struct fpi_mapper {};

template <>
struct fpi_mapper<double> {
  using type                         = int64_t;
  static constexpr int kBitshiftBase = 53;
};

template <>
struct fpi_mapper<float> {
  using type                         = int32_t;
  static constexpr int kBitshiftBase = 24;
};

template <>
struct fpi_mapper<half> {
  using type                         = int16_t;
  static constexpr int kBitshiftBase = 11;
};

// Generate dataset to ensure no rounding error occurs in the norm computation of any two vectors.
// When testing the CAGRA index sorting function, rounding errors can affect the norm and alter the
// order of the index. To ensure the accuracy of the test, we utilize the dataset. The generation
// method is based on the error-free transformation (EFT) method.
template <typename T>
RAFT_KERNEL GenerateRoundingErrorFreeDataset_kernel(T* const ptr,
                                                    const uint32_t size,
                                                    const typename fpi_mapper<T>::type resolution)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) { return; }

  const float u32 = *reinterpret_cast<const typename fpi_mapper<T>::type*>(ptr + tid);
  ptr[tid]        = u32 / resolution;
}

template <typename T>
void GenerateRoundingErrorFreeDataset(
  const raft::resources& handle,
  T* const ptr,
  const uint32_t n_row,
  const uint32_t dim,
  raft::random::RngState& rng,
  const bool diff_flag  // true if compute the norm between two vectors
)
{
  using mapper_type         = fpi_mapper<T>;
  using int_type            = typename mapper_type::type;
  auto cuda_stream          = resource::get_cuda_stream(handle);
  const uint32_t size       = n_row * dim;
  const uint32_t block_size = 256;
  const uint32_t grid_size  = (size + block_size - 1) / block_size;

  const auto bitshift = (mapper_type::kBitshiftBase - std::log2(dim) - (diff_flag ? 1 : 0)) / 2;
  // Skip the test when `dim` is too big for type `T` to allow rounding error-free test.
  if (bitshift <= 1) { GTEST_SKIP(); }
  const int_type resolution = int_type{1} << static_cast<unsigned>(std::floor(bitshift));
  raft::random::uniformInt<int_type>(
    handle, rng, reinterpret_cast<int_type*>(ptr), size, -resolution, resolution - 1);

  GenerateRoundingErrorFreeDataset_kernel<T>
    <<<grid_size, block_size, 0, cuda_stream>>>(ptr, size, resolution);
}

template <class T>
__global__ void Normalize8bitInt_kernel(T* const datatset_ptr,
                                        const std::uint32_t size,
                                        const std::uint32_t dim,
                                        const std::uint32_t normalized_norm)
{
  const auto tid      = threadIdx.x + blockDim.x * blockIdx.x;
  std::uint32_t norm2 = 0;
  for (std::uint32_t i = 0; i < dim; i++) {
    const std::uint32_t v = datatset_ptr[tid * dim + i];
    norm2 += v * v;
  }
  const float scale = normalized_norm / sqrtf(static_cast<float>(norm2));
  for (std::uint32_t i = 0; i < dim; i++) {
    const auto v = datatset_ptr[tid * dim + i] * scale;
    datatset_ptr[tid * dim + i] =
      std::max(std::min(v, static_cast<float>(std::numeric_limits<T>::max())),
               static_cast<float>(std::numeric_limits<T>::min()));
  }
}

template <class T>
void Normalize8bitInt(const raft::resources& handle,
                      T* const datatset_ptr,
                      const std::uint32_t size,
                      const std::uint32_t dim)
{
  static_assert(std::is_same_v<T, std::uint8_t> || std::is_same_v<T, std::int8_t>);

  const std::uint32_t block_size = 256;
  const std::uint32_t grid_size  = raft::ceildiv<std::uint32_t>(size, block_size);

  const std::uint32_t normalized_norm = (std::is_same_v<T, std::uint8_t> ? 40 : 20) * dim;

  Normalize8bitInt_kernel<<<grid_size, block_size, 0, raft::resource::get_cuda_stream(handle)>>>(
    datatset_ptr, size, dim, normalized_norm);
}

template <class DataT>
void InitDataset(const raft::resources& handle,
                 DataT* const datatset_ptr,
                 std::uint32_t size,
                 std::uint32_t dim,
                 raft::distance::DistanceType metric,
                 raft::random::RngState& r)
{
  if constexpr (std::is_same_v<DataT, float> || std::is_same_v<DataT, half>) {
    GenerateRoundingErrorFreeDataset(handle, datatset_ptr, size, dim, r, true);

    if (metric == raft::distance::InnerProduct) {
      auto dataset_view = raft::make_device_matrix_view(datatset_ptr, size, dim);
      raft::linalg::row_normalize(
        handle, raft::make_const_mdspan(dataset_view), dataset_view, raft::linalg::L2Norm);
    }
  } else {
    raft::random::uniformInt(handle, r, datatset_ptr, size * dim, DataT(1), DataT(20));

    if (metric == raft::distance::InnerProduct) {
      Normalize8bitInt(handle, datatset_ptr, size, dim);
    }
  }
}
}  // namespace

struct AnnCagraInputs {
  int n_queries;
  int n_rows;
  int dim;
  int k;
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

inline ::std::ostream& operator<<(::std::ostream& os, const AnnCagraInputs& p)
{
  std::vector<std::string> algo       = {"single-cta", "multi_cta", "multi_kernel", "auto"};
  std::vector<std::string> build_algo = {"IVF_PQ", "NN_DESCENT"};
  os << "{n_queries=" << p.n_queries << ", dataset shape=" << p.n_rows << "x" << p.dim
     << ", k=" << p.k << ", " << algo.at((int)p.algo) << ", max_queries=" << p.max_queries
     << ", itopk_size=" << p.itopk_size << ", search_width=" << p.search_width
     << ", metric=" << static_cast<int>(p.metric) << (p.host_dataset ? ", host" : ", device")
     << ", build_algo=" << build_algo.at((int)p.build_algo) << '}' << std::endl;
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
    // TODO (tarang-jain): remove when NN Descent index building support InnerProduct. Reference
    // issue: https://github.com/rapidsai/raft/issues/2276
    if (ps.metric == distance::InnerProduct && ps.build_algo == graph_build_algo::NN_DESCENT)
      GTEST_SKIP();

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
        index_params.build_algo = ps.build_algo;
        cagra::search_params search_params;
        search_params.algo        = ps.algo;
        search_params.max_queries = ps.max_queries;
        search_params.team_size   = ps.team_size;
        search_params.itopk_size  = ps.itopk_size;

        auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data(), ps.n_rows, ps.dim);

        {
          cagra::index<DataT, IdxT> index(handle_);
          if (ps.host_dataset) {
            auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
            raft::copy(database_host.data_handle(), database.data(), database.size(), stream_);
            auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
              (const DataT*)database_host.data_handle(), ps.n_rows, ps.dim);
            index = cagra::build<DataT, IdxT>(handle_, index_params, database_host_view);
          } else {
            index = cagra::build<DataT, IdxT>(handle_, index_params, database_view);
          };
          cagra::serialize(handle_, "cagra_index", index, ps.include_serialized_dataset);
        }

        auto index = cagra::deserialize<DataT, IdxT>(handle_, "cagra_index");
        if (!ps.include_serialized_dataset) { index.update_dataset(handle_, database_view); }

        auto search_queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_out_view =
          raft::make_device_matrix_view<IdxT, int64_t>(indices_dev.data(), ps.n_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<DistanceT, int64_t>(
          distances_dev.data(), ps.n_queries, ps.k);

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
    raft::random::RngState r(1234ULL);
    InitDataset(handle_, database.data(), ps.n_rows, ps.dim, ps.metric, r);
    InitDataset(handle_, search_queries.data(), ps.n_queries, ps.dim, ps.metric, r);
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
    if (ps.metric == distance::InnerProduct && ps.build_algo == graph_build_algo::NN_DESCENT)
      GTEST_SKIP();

    {
      // Step 1: Build a sorted KNN graph by CAGRA knn build
      auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
        (const DataT*)database.data(), ps.n_rows, ps.dim);
      auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
      raft::copy(
        database_host.data_handle(), database.data(), database.size(), handle_.get_stream());
      auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
        (const DataT*)database_host.data_handle(), ps.n_rows, ps.dim);

      cagra::index_params index_params;
      auto knn_graph =
        raft::make_host_matrix<IdxT, int64_t>(ps.n_rows, index_params.intermediate_graph_degree);

      if (ps.build_algo == graph_build_algo::IVF_PQ) {
        auto build_params = ivf_pq::index_params::from_dataset(database_view, ps.metric);
        if (ps.host_dataset) {
          cagra::build_knn_graph<DataT, IdxT>(
            handle_, database_host_view, knn_graph.view(), 2, build_params);
        } else {
          cagra::build_knn_graph<DataT, IdxT>(
            handle_, database_view, knn_graph.view(), 2, build_params);
        }
      } else {
        auto nn_descent_idx_params                      = experimental::nn_descent::index_params{};
        nn_descent_idx_params.graph_degree              = index_params.intermediate_graph_degree;
        nn_descent_idx_params.intermediate_graph_degree = index_params.intermediate_graph_degree;

        if (ps.host_dataset) {
          cagra::build_knn_graph<DataT, IdxT>(
            handle_, database_host_view, knn_graph.view(), nn_descent_idx_params);
        } else {
          cagra::build_knn_graph<DataT, IdxT>(
            handle_, database_host_view, knn_graph.view(), nn_descent_idx_params);
        }
      }

      handle_.sync_stream();
      ASSERT_TRUE(CheckOrder<DistanceT>(knn_graph.view(), database_host.view(), ps.metric));

      if (ps.metric != raft::distance::DistanceType::InnerProduct) {
        RandomSuffle(knn_graph.view());

        cagra::sort_knn_graph(handle_, database_view, knn_graph.view());
        handle_.sync_stream();

        ASSERT_TRUE(CheckOrder<DistanceT>(knn_graph.view(), database_host.view(), ps.metric));
      }
    }
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, handle_.get_stream());
    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same_v<DataT, float> || std::is_same_v<DataT, half>) {
      GenerateRoundingErrorFreeDataset(handle_, database.data(), ps.n_rows, ps.dim, r, false);
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.n_rows * ps.dim, DataT(1), DataT(20));
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

template <typename DistanceT, typename DataT, typename IdxT>
class AnnCagraFilterTest : public ::testing::TestWithParam<AnnCagraInputs> {
 public:
  AnnCagraFilterTest()
    : stream_(resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnCagraInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  void testCagraFilter()
  {
    if (ps.metric == distance::InnerProduct && ps.build_algo == graph_build_algo::NN_DESCENT)
      GTEST_SKIP();

    size_t queries_size = ps.n_queries * ps.k;
    std::vector<IdxT> indices_Cagra(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_Cagra(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      auto* database_filtered_ptr = database.data() + test_cagra_sample_filter::offset * ps.dim;
      naive_knn<DistanceT, DataT, IdxT>(handle_,
                                        distances_naive_dev.data(),
                                        indices_naive_dev.data(),
                                        search_queries.data(),
                                        database_filtered_ptr,
                                        ps.n_queries,
                                        ps.n_rows - test_cagra_sample_filter::offset,
                                        ps.dim,
                                        ps.k,
                                        ps.metric);
      raft::linalg::addScalar(indices_naive_dev.data(),
                              indices_naive_dev.data(),
                              IdxT(test_cagra_sample_filter::offset),
                              queries_size,
                              stream_);
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
        index_params.nn_descent_niter = 50;
        cagra::search_params search_params;
        search_params.algo         = ps.algo;
        search_params.max_queries  = ps.max_queries;
        search_params.team_size    = ps.team_size;
        search_params.hashmap_mode = cagra::hash_mode::HASH;

        // TODO: setting search_params.itopk_size here breaks the filter tests, but is required for
        // k>1024 skip these tests until fixed
        if (ps.k >= 1024) { GTEST_SKIP(); }
        // search_params.itopk_size   = ps.itopk_size;

        auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data(), ps.n_rows, ps.dim);

        cagra::index<DataT, IdxT> index(handle_);
        if (ps.host_dataset) {
          auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
          raft::copy(database_host.data_handle(), database.data(), database.size(), stream_);
          auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
            (const DataT*)database_host.data_handle(), ps.n_rows, ps.dim);
          index = cagra::build<DataT, IdxT>(handle_, index_params, database_host_view);
        } else {
          index = cagra::build<DataT, IdxT>(handle_, index_params, database_view);
        }

        if (!ps.include_serialized_dataset) { index.update_dataset(handle_, database_view); }

        auto search_queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_out_view =
          raft::make_device_matrix_view<IdxT, int64_t>(indices_dev.data(), ps.n_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<DistanceT, int64_t>(
          distances_dev.data(), ps.n_queries, ps.k);

        cagra::search_with_filtering(handle_,
                                     search_params,
                                     index,
                                     search_queries_view,
                                     indices_out_view,
                                     dists_out_view,
                                     test_cagra_sample_filter());
        update_host(distances_Cagra.data(), distances_dev.data(), queries_size, stream_);
        update_host(indices_Cagra.data(), indices_dev.data(), queries_size, stream_);
        resource::sync_stream(handle_);
      }

      // Test filter
      bool unacceptable_node = false;
      for (int q = 0; q < ps.n_queries; q++) {
        for (int i = 0; i < ps.k; i++) {
          const auto n      = indices_Cagra[q * ps.k + i];
          unacceptable_node = unacceptable_node | !test_cagra_sample_filter()(q, n);
        }
      }
      EXPECT_FALSE(unacceptable_node);

      double min_recall = ps.min_recall;
      // TODO(mfoerster): re-enable uniquenes test
      EXPECT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.003,
                                  min_recall,
                                  false));
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

  void testCagraRemoved()
  {
    if (ps.metric == distance::InnerProduct && ps.build_algo == graph_build_algo::NN_DESCENT)
      GTEST_SKIP();

    size_t queries_size = ps.n_queries * ps.k;
    std::vector<IdxT> indices_Cagra(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_Cagra(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      auto* database_filtered_ptr = database.data() + test_cagra_sample_filter::offset * ps.dim;
      naive_knn<DistanceT, DataT, IdxT>(handle_,
                                        distances_naive_dev.data(),
                                        indices_naive_dev.data(),
                                        search_queries.data(),
                                        database_filtered_ptr,
                                        ps.n_queries,
                                        ps.n_rows - test_cagra_sample_filter::offset,
                                        ps.dim,
                                        ps.k,
                                        ps.metric);
      raft::linalg::addScalar(indices_naive_dev.data(),
                              indices_naive_dev.data(),
                              IdxT(test_cagra_sample_filter::offset),
                              queries_size,
                              stream_);
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
        index_params.nn_descent_niter = 50;
        cagra::search_params search_params;
        search_params.algo         = ps.algo;
        search_params.max_queries  = ps.max_queries;
        search_params.team_size    = ps.team_size;
        search_params.hashmap_mode = cagra::hash_mode::HASH;

        // TODO: setting search_params.itopk_size here breaks the filter tests, but is required for
        // k>1024 skip these tests until fixed
        if (ps.k >= 1024) { GTEST_SKIP(); }
        // search_params.itopk_size   = ps.itopk_size;

        auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data(), ps.n_rows, ps.dim);

        cagra::index<DataT, IdxT> index(handle_);
        if (ps.host_dataset) {
          auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
          raft::copy(database_host.data_handle(), database.data(), database.size(), stream_);
          auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
            (const DataT*)database_host.data_handle(), ps.n_rows, ps.dim);
          index = cagra::build<DataT, IdxT>(handle_, index_params, database_host_view);
        } else {
          index = cagra::build<DataT, IdxT>(handle_, index_params, database_view);
        }

        if (!ps.include_serialized_dataset) { index.update_dataset(handle_, database_view); }

        auto search_queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_out_view =
          raft::make_device_matrix_view<IdxT, int64_t>(indices_dev.data(), ps.n_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<DistanceT, int64_t>(
          distances_dev.data(), ps.n_queries, ps.k);
        auto removed_indices =
          raft::make_device_vector<IdxT, int64_t>(handle_, test_cagra_sample_filter::offset);
        thrust::sequence(
          resource::get_thrust_policy(handle_),
          thrust::device_pointer_cast(removed_indices.data_handle()),
          thrust::device_pointer_cast(removed_indices.data_handle() + removed_indices.extent(0)));
        resource::sync_stream(handle_);
        raft::core::bitset<std::uint32_t, IdxT> removed_indices_bitset(
          handle_, removed_indices.view(), ps.n_rows);
        cagra::search_with_filtering(
          handle_,
          search_params,
          index,
          search_queries_view,
          indices_out_view,
          dists_out_view,
          raft::neighbors::filtering::bitset_filter(removed_indices_bitset.view()));
        update_host(distances_Cagra.data(), distances_dev.data(), queries_size, stream_);
        update_host(indices_Cagra.data(), indices_dev.data(), queries_size, stream_);
        resource::sync_stream(handle_);
      }

      double min_recall = ps.min_recall;
      // TODO(mfoerster): re-enable uniquenes test
      EXPECT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.003,
                                  min_recall,
                                  false));
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
    raft::random::RngState r(1234ULL);
    InitDataset(handle_, database.data(), ps.n_rows, ps.dim, ps.metric, r);
    InitDataset(handle_, search_queries.data(), ps.n_queries, ps.dim, ps.metric, r);
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

inline std::vector<AnnCagraInputs> generate_inputs()
{
  // TODO(tfeher): test MULTI_CTA kernel with search_width > 1 to allow multiple CTA per queries
  std::vector<AnnCagraInputs> inputs = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {1, 8, 17, 1599},
    {16},  // k
    {graph_build_algo::IVF_PQ, graph_build_algo::NN_DESCENT},
    {search_algo::SINGLE_CTA, search_algo::MULTI_CTA, search_algo::MULTI_KERNEL},
    {0, 1, 10, 100},  // query size
    {0},
    {256},
    {1},
    {raft::distance::DistanceType::L2Expanded, raft::distance::DistanceType::InnerProduct},
    {false},
    {true},
    {0.995});

  auto inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {1, 8, 17, 1599},
    {1},  // k
    {graph_build_algo::IVF_PQ, graph_build_algo::NN_DESCENT},
    {search_algo::SINGLE_CTA, search_algo::MULTI_CTA, search_algo::MULTI_KERNEL},
    {0, 1, 10, 100},  // query size
    {0},
    {256},
    {1},
    {raft::distance::DistanceType::L2Expanded, raft::distance::DistanceType::InnerProduct},
    {false},
    {true},
    {99. / 100}
    // smaller threshould than the other test cases because it is too strict for Top-1 search
  );
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {1, 3, 5, 7, 8, 17, 64, 128, 137, 192, 256, 512, 619, 1024},  // dim
    {16},                                                         // k
    {graph_build_algo::IVF_PQ, graph_build_algo::NN_DESCENT},
    {search_algo::AUTO},
    {10},
    {0},
    {64},
    {1},
    {raft::distance::DistanceType::L2Expanded, raft::distance::DistanceType::InnerProduct},
    {false},
    {true},
    {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {64},
    {16},
    {graph_build_algo::IVF_PQ, graph_build_algo::NN_DESCENT},
    {search_algo::AUTO},
    {10},
    {0, 4, 8, 16, 32},  // team_size
    {64},
    {1},
    {raft::distance::DistanceType::L2Expanded, raft::distance::DistanceType::InnerProduct},
    {false},
    {false},
    {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {64},
    {16},
    {graph_build_algo::IVF_PQ, graph_build_algo::NN_DESCENT},
    {search_algo::AUTO},
    {10},
    {0},  // team_size
    {32, 64, 128, 256, 512, 768},
    {1},
    {raft::distance::DistanceType::L2Expanded, raft::distance::DistanceType::InnerProduct},
    {false},
    {true},
    {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {10000, 20000},
    {32},
    {10},
    {graph_build_algo::IVF_PQ, graph_build_algo::NN_DESCENT},
    {search_algo::AUTO},
    {10},
    {0},  // team_size
    {64},
    {1},
    {raft::distance::DistanceType::L2Expanded, raft::distance::DistanceType::InnerProduct},
    {false, true},
    {false},
    {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {20000},
    {32},
    {2048},  // k
    {graph_build_algo::NN_DESCENT},
    {search_algo::AUTO},
    {10},
    {0},
    {4096},  // itopk_size
    {1},
    {raft::distance::DistanceType::L2Expanded, raft::distance::DistanceType::InnerProduct},
    {false},
    {false},
    {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  return inputs;
}

const std::vector<AnnCagraInputs> inputs = generate_inputs();

}  // namespace raft::neighbors::cagra
