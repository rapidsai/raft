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

#include <raft/neighbors/ann_mg_helpers.cuh>

#include <raft/neighbors/ivf_flat_mg.cuh>
#include <raft/neighbors/ivf_pq_mg.cuh>
#include <raft/neighbors/cagra_mg.cuh>

#include <raft/neighbors/ivf_flat_mg_serialize.cuh>
#include <raft/neighbors/ivf_pq_mg_serialize.cuh>
#include <raft/neighbors/cagra_mg_serialize.cuh>

#include <raft_internal/neighbors/naive_knn.cuh>

#include <raft/comms/std_comms.hpp>


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
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<T> distances_naive(queries_size);
    std::vector<IdxT> indices_ann(queries_size);
    std::vector<T> distances_ann(queries_size);
    std::vector<uint32_t> indices_naive_32bits(queries_size);
    std::vector<uint32_t> indices_ann_32bits(queries_size);

    {
      rmm::device_uvector<T> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      raft::neighbors::naive_knn<T, DataT, IdxT>(handle_,
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

    std::vector<int> device_ids{0, 1};

    // IVF-Flat
    for (parallel_mode d_mode : {parallel_mode::REPLICATION, parallel_mode::SHARDING}) {
      ivf_flat::dist_index_params index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.adaptive_centers         = ps.adaptive_centers;
      index_params.add_data_on_build        = false;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;
      index_params.mode                     = d_mode;

      ivf_flat::search_params search_params;
      search_params.n_probes = ps.nprobe;

      auto index_dataset = raft::make_host_matrix_view<const DataT, IdxT, row_major>(
        h_index_dataset.data(), ps.num_db_vecs, ps.dim);
      auto query_dataset = raft::make_host_matrix_view<const DataT, IdxT, row_major>(
        h_query_dataset.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<IdxT, IdxT, row_major>(
        indices_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, IdxT, row_major>(
        distances_ann.data(), ps.num_queries, ps.k);

      raft::neighbors::mg::nccl_clique clique(device_ids);
      {
        auto index = raft::neighbors::mg::build<DataT, IdxT>(handle_, clique, index_params, index_dataset);
        raft::neighbors::mg::extend<DataT, IdxT>(handle_, clique, index, index_dataset, std::nullopt);
        raft::neighbors::mg::serialize<DataT, IdxT>(handle_, clique, index, "./cpp/build/ann_mg_ivf_flat_index");
      }
      auto new_index = raft::neighbors::mg::deserialize_flat<DataT, IdxT>(handle_, clique, "./cpp/build/ann_mg_ivf_flat_index");
      raft::neighbors::mg::search<DataT, IdxT>(handle_, clique, new_index, search_params, query_dataset, neighbors, distances);
      resource::sync_stream(handle_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ann,
                                  distances_naive,
                                  distances_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(indices_ann.begin(), indices_ann.end(), 0);
      std::fill(distances_ann.begin(), distances_ann.end(), 0);
    }

    // IVF-PQ
    for (parallel_mode d_mode : {parallel_mode::REPLICATION, parallel_mode::SHARDING}) {
      ivf_pq::dist_index_params index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.add_data_on_build        = false;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;
      index_params.mode                     = d_mode;

      ivf_pq::search_params search_params;
      search_params.n_probes = ps.nprobe;

      auto index_dataset = raft::make_host_matrix_view<const DataT, IdxT, row_major>(
        h_index_dataset.data(), ps.num_db_vecs, ps.dim);
      auto query_dataset = raft::make_host_matrix_view<const DataT, IdxT, row_major>(
        h_query_dataset.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<IdxT, IdxT, row_major>(
        indices_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, IdxT, row_major>(
        distances_ann.data(), ps.num_queries, ps.k);

      raft::neighbors::mg::nccl_clique clique(device_ids);
      {
        auto index = raft::neighbors::mg::build<DataT, IdxT>(handle_, clique, index_params, index_dataset);
        raft::neighbors::mg::extend<DataT, IdxT>(handle_, clique, index, index_dataset, std::nullopt);
        raft::neighbors::mg::serialize<DataT, IdxT>(handle_, clique, index, "./cpp/build/ann_mg_ivf_pq_index");
      }
      auto new_index = raft::neighbors::mg::deserialize_pq<DataT, IdxT>(handle_, clique, "./cpp/build/ann_mg_ivf_pq_index");
      raft::neighbors::mg::search<DataT, IdxT>(handle_, clique, new_index, search_params, query_dataset, neighbors, distances);
      resource::sync_stream(handle_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ann,
                                  distances_naive,
                                  distances_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(indices_ann.begin(), indices_ann.end(), 0);
      std::fill(distances_ann.begin(), distances_ann.end(), 0);
    }

    // CAGRA
    for (parallel_mode d_mode : {parallel_mode::REPLICATION, parallel_mode::SHARDING}) {
      cagra::dist_index_params index_params;
      index_params.add_data_on_build              = true;
      index_params.intermediate_graph_degree      = 128;
      index_params.graph_degree                   = 64;
      index_params.build_algo                     = cagra::graph_build_algo::IVF_PQ;
      index_params.nn_descent_niter               = 20;
      index_params.mode                           = d_mode;

      cagra::search_params search_params;

      auto index_dataset = raft::make_host_matrix_view<const DataT, uint32_t, row_major>(
        h_index_dataset.data(), ps.num_db_vecs, ps.dim);
      auto query_dataset = raft::make_host_matrix_view<const DataT, uint32_t, row_major>(
        h_query_dataset.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<uint32_t, uint32_t, row_major>(
        indices_ann_32bits.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, uint32_t, row_major>(
        distances_ann.data(), ps.num_queries, ps.k);

      raft::neighbors::mg::nccl_clique clique(device_ids);
      /*
      {
        auto index = raft::neighbors::mg::build<DataT, uint32_t>(handle_, clique, index_params, index_dataset);
        raft::neighbors::mg::serialize<DataT, uint32_t>(handle_, clique, index, "./cpp/build/ann_mg_cagra_index");
      }
      auto new_index = raft::neighbors::mg::deserialize_cagra<DataT, uint32_t>(handle_, clique, "./cpp/build/ann_mg_cagra_index");
      */
      auto new_index = raft::neighbors::mg::build<DataT, uint32_t>(handle_, clique, index_params, index_dataset);
      raft::neighbors::mg::search<DataT, uint32_t>(handle_, clique, new_index, search_params, query_dataset, neighbors, distances);
      resource::sync_stream(handle_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(indices_naive_32bits,
                                  indices_ann_32bits,
                                  distances_naive,
                                  distances_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(indices_ann_32bits.begin(), indices_ann_32bits.end(), 0);
      std::fill(distances_ann.begin(), distances_ann.end(), 0);
    }

    {
      ivf_flat::index_params index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.adaptive_centers         = ps.adaptive_centers;
      index_params.add_data_on_build        = true;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;

      ivf_flat::search_params search_params;
      search_params.n_probes = ps.nprobe;

      RAFT_CUDA_TRY(cudaSetDevice(0));

      {
        auto index_dataset = raft::make_device_matrix_view<const DataT, IdxT>(d_index_dataset.data(), ps.num_db_vecs, ps.dim);
        auto index = raft::runtime::neighbors::ivf_flat::build(handle_, index_params, index_dataset);
        ivf_flat::serialize<DataT, IdxT>(handle_, "./cpp/build/local_ivf_flat_index", index);
      }

      auto query_dataset = raft::make_host_matrix_view<const DataT, IdxT, row_major>(h_query_dataset.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<IdxT, IdxT, row_major>(indices_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, IdxT, row_major>(distances_ann.data(), ps.num_queries, ps.k);

      raft::neighbors::mg::nccl_clique clique(device_ids);
      auto distributed_index = raft::neighbors::mg::distribute_flat<DataT, IdxT>(handle_, clique, "./cpp/build/local_ivf_flat_index");
      raft::neighbors::mg::search<DataT, IdxT>(handle_, clique, distributed_index, search_params, query_dataset, neighbors, distances);

      resource::sync_stream(handle_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ann,
                                  distances_naive,
                                  distances_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(indices_ann.begin(), indices_ann.end(), 0);
      std::fill(distances_ann.begin(), distances_ann.end(), 0);
    }

    {
      ivf_pq::index_params index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.add_data_on_build        = true;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;

      ivf_pq::search_params search_params;
      search_params.n_probes = ps.nprobe;

      RAFT_CUDA_TRY(cudaSetDevice(0));

      {
        auto index_dataset = raft::make_device_matrix_view<const DataT, IdxT>(d_index_dataset.data(), ps.num_db_vecs, ps.dim);
        auto index = raft::runtime::neighbors::ivf_pq::build(handle_, index_params, index_dataset);
        ivf_pq::serialize<IdxT>(handle_, "./cpp/build/local_ivf_pq_index", index);
      }

      auto query_dataset = raft::make_host_matrix_view<const DataT, IdxT, row_major>(h_query_dataset.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<IdxT, IdxT, row_major>(indices_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, IdxT, row_major>(distances_ann.data(), ps.num_queries, ps.k);

      raft::neighbors::mg::nccl_clique clique(device_ids);
      auto distributed_index = raft::neighbors::mg::distribute_pq<DataT, IdxT>(handle_, clique, "./cpp/build/local_ivf_pq_index");
      raft::neighbors::mg::search<DataT, IdxT>(handle_, clique, distributed_index, search_params, query_dataset, neighbors, distances);

      resource::sync_stream(handle_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ann,
                                  distances_naive,
                                  distances_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(indices_ann.begin(), indices_ann.end(), 0);
      std::fill(distances_ann.begin(), distances_ann.end(), 0);
    }

    {
      cagra::index_params index_params;
      index_params.intermediate_graph_degree      = 128;
      index_params.graph_degree                   = 64;
      index_params.build_algo                     = cagra::graph_build_algo::IVF_PQ;
      index_params.nn_descent_niter               = 20;

      cagra::search_params search_params;

      RAFT_CUDA_TRY(cudaSetDevice(0));

      {
        auto index_dataset = raft::make_device_matrix_view<const DataT, int64_t>(d_index_dataset.data(), ps.num_db_vecs, ps.dim);
        auto index = raft::runtime::neighbors::cagra::build(handle_, index_params, index_dataset);
        raft::neighbors::cagra::serialize(handle_, "./cpp/build/local_cagra_index", index);
      }

      auto query_dataset = raft::make_host_matrix_view<const DataT, int64_t, row_major>(h_query_dataset.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<uint32_t, int64_t, row_major>(indices_ann_32bits.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, int64_t, row_major>(distances_ann.data(), ps.num_queries, ps.k);

      raft::neighbors::mg::nccl_clique clique(device_ids);
      auto distributed_index = raft::neighbors::mg::distribute_cagra<DataT, uint32_t>(handle_, clique, "./cpp/build/local_cagra_index");
      raft::neighbors::mg::search<DataT, uint32_t>(handle_, clique, distributed_index, search_params, query_dataset, neighbors, distances);

      resource::sync_stream(handle_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(indices_naive_32bits,
                                  indices_ann_32bits,
                                  distances_naive,
                                  distances_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(indices_ann_32bits.begin(), indices_ann_32bits.end(), 0);
      std::fill(distances_ann.begin(), distances_ann.end(), 0);
    }

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
        handle_, r, d_index_dataset.data(), d_index_dataset.size(), DataT(0.1), DataT(2.0));
      raft::random::uniform(
        handle_, r, d_query_dataset.data(), d_query_dataset.size(), DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, d_index_dataset.data(), d_index_dataset.size(), DataT(1), DataT(20));
      raft::random::uniformInt(
        handle_, r, d_query_dataset.data(), d_query_dataset.size(), DataT(1), DataT(20));
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

const std::vector<AnnMGInputs<int64_t>> inputs = {
  {1000, 10000, 8, 16, 40, 1024, raft::distance::DistanceType::L2Expanded, true},
};
}  // namespace raft::neighbors::mg
