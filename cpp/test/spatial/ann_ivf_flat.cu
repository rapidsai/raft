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
#include "./ann_base_kernel.cuh"

#include <raft/core/logger.hpp>
#include <raft/distance/distance_type.hpp>
#include <raft/random/rng.cuh>
#include <raft/spatial/knn/ann.cuh>
#include <raft/spatial/knn/ivf_flat.cuh>
#include <raft/spatial/knn/knn.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <thrust/sequence.h>

#include <cstddef>
#include <iostream>
#include <vector>

namespace raft {
namespace spatial {
namespace knn {
struct AnnIvfFlatInputs {
  int num_queries;
  int num_db_vecs;
  int dim;
  int k;
  int nprobe;
  int nlist;
  raft::distance::DistanceType metric;
};

template <typename IdxT, typename DistT, typename compareDist>
struct idx_dist_pair {
  IdxT idx;
  DistT dist;
  compareDist eq_compare;
  bool operator==(const idx_dist_pair<IdxT, DistT, compareDist>& a) const
  {
    if (idx == a.idx) return true;
    if (eq_compare(dist, a.dist)) return true;
    return false;
  }
  idx_dist_pair(IdxT x, DistT y, compareDist op) : idx(x), dist(y), eq_compare(op) {}
};

template <typename T, typename DistT>
auto eval_knn(const std::vector<T>& expected_idx,
              const std::vector<T>& actual_idx,
              const std::vector<DistT>& expected_dist,
              const std::vector<DistT>& actual_dist,
              size_t rows,
              size_t cols,
              const DistT eps,
              double min_recall) -> testing::AssertionResult
{
  size_t match_count = 0;
  size_t total_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t k = 0; k < cols; ++k) {
      size_t idx_k  = i * cols + k;  // row major assumption!
      auto act_idx  = actual_idx[idx_k];
      auto act_dist = actual_dist[idx_k];
      for (size_t j = 0; j < cols; ++j) {
        size_t idx    = i * cols + j;  // row major assumption!
        auto exp_idx  = expected_idx[idx];
        auto exp_dist = expected_dist[idx];
        idx_dist_pair exp_kvp(exp_idx, exp_dist, raft::CompareApprox<DistT>(eps));
        idx_dist_pair act_kvp(act_idx, act_dist, raft::CompareApprox<DistT>(eps));
        if (exp_kvp == act_kvp) {
          match_count++;
          break;
        }
      }
    }
  }
  RAFT_LOG_INFO("Recall = %zu/%zu", match_count, total_count);
  double actual_recall = static_cast<double>(match_count) / static_cast<double>(total_count);
  if (actual_recall < min_recall - eps) {
    if (actual_recall < min_recall * min_recall - eps) {
      RAFT_LOG_ERROR("Recall is much lower than the minimum (%f < %f)", actual_recall, min_recall);
    } else {
      RAFT_LOG_WARN("Recall is suspiciously too low (%f < %f)", actual_recall, min_recall);
    }
    if (match_count == 0 || actual_recall < min_recall * std::min(min_recall, 0.5) - eps) {
      return testing::AssertionFailure()
             << "actual recall (" << actual_recall
             << ") is much smaller than the minimum expected recall (" << min_recall << ").";
    }
  }
  return testing::AssertionSuccess();
}

template <typename T, typename DataT>
class AnnIVFFlatTest : public ::testing::TestWithParam<AnnIvfFlatInputs> {
 public:
  AnnIVFFlatTest()
    : stream_(handle_.get_stream()),
      ps(::testing::TestWithParam<AnnIvfFlatInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  void testIVFFlat()
  {
    size_t queries_size = ps.num_queries * ps.k;
    std::vector<int64_t> indices_ivfflat(queries_size);
    std::vector<int64_t> indices_naive(queries_size);
    std::vector<T> distances_ivfflat(queries_size);
    std::vector<T> distances_naive(queries_size);

    {
      rmm::device_uvector<T> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<int64_t> indices_naive_dev(queries_size, stream_);
      using acc_t = typename detail::utils::config<DataT>::value_t;
      naiveBfKnn<DataT, acc_t>(distances_naive_dev.data(),
                               indices_naive_dev.data(),
                               search_queries.data(),
                               database.data(),
                               ps.num_queries,
                               ps.num_db_vecs,
                               ps.dim,
                               ps.k,
                               ps.metric,
                               2.0f,
                               stream_);
      update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      handle_.sync_stream(stream_);
    }

    {
      // unless something is really wrong with clustering, this could serve as a lower bound on
      // recall
      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);

      rmm::device_uvector<T> distances_ivfflat_dev(queries_size, stream_);
      rmm::device_uvector<int64_t> indices_ivfflat_dev(queries_size, stream_);

      {
        // legacy interface
        raft::spatial::knn::IVFFlatParam ivfParams;
        ivfParams.nprobe = ps.nprobe;
        ivfParams.nlist  = ps.nlist;
        raft::spatial::knn::knnIndex index;
        index.index   = nullptr;
        index.gpu_res = nullptr;

        approx_knn_build_index(handle_,
                               &index,
                               dynamic_cast<raft::spatial::knn::knnIndexParam*>(&ivfParams),
                               ps.metric,
                               0,
                               database.data(),
                               ps.num_db_vecs,
                               ps.dim);
        handle_.sync_stream(stream_);
        approx_knn_search(handle_,
                          distances_ivfflat_dev.data(),
                          indices_ivfflat_dev.data(),
                          &index,
                          ps.k,
                          search_queries.data(),
                          ps.num_queries);

        update_host(distances_ivfflat.data(), distances_ivfflat_dev.data(), queries_size, stream_);
        update_host(indices_ivfflat.data(), indices_ivfflat_dev.data(), queries_size, stream_);
        handle_.sync_stream(stream_);
      }

      ASSERT_TRUE(eval_knn(indices_naive,
                           indices_ivfflat,
                           distances_naive,
                           distances_ivfflat,
                           ps.num_queries,
                           ps.k,
                           float(0.001),
                           min_recall));
      {
        // new interface
        raft::spatial::knn::ivf_flat::index_params index_params;
        raft::spatial::knn::ivf_flat::search_params search_params;
        index_params.n_lists   = ps.nlist;
        index_params.metric    = ps.metric;
        search_params.n_probes = ps.nprobe;

        index_params.add_data_on_build        = false;
        index_params.kmeans_trainset_fraction = 0.5;
        auto index =
          ivf_flat::build(handle_, index_params, database.data(), int64_t(ps.num_db_vecs), ps.dim);

        rmm::device_uvector<int64_t> vector_indices(ps.num_db_vecs, stream_);
        thrust::sequence(handle_.get_thrust_policy(),
                         thrust::device_pointer_cast(vector_indices.data()),
                         thrust::device_pointer_cast(vector_indices.data() + ps.num_db_vecs));
        handle_.sync_stream(stream_);

        int64_t half_of_data = ps.num_db_vecs / 2;

        auto index_2 =
          ivf_flat::extend<DataT, int64_t>(handle_, index, database.data(), nullptr, half_of_data);

        ivf_flat::extend<DataT, int64_t>(handle_,
                                         &index_2,
                                         database.data() + half_of_data * ps.dim,
                                         vector_indices.data() + half_of_data,
                                         int64_t(ps.num_db_vecs) - half_of_data);

        ivf_flat::search(handle_,
                         search_params,
                         index_2,
                         search_queries.data(),
                         ps.num_queries,
                         ps.k,
                         indices_ivfflat_dev.data(),
                         distances_ivfflat_dev.data());

        update_host(distances_ivfflat.data(), distances_ivfflat_dev.data(), queries_size, stream_);
        update_host(indices_ivfflat.data(), indices_ivfflat_dev.data(), queries_size, stream_);
        handle_.sync_stream(stream_);
      }
      ASSERT_TRUE(eval_knn(indices_naive,
                           indices_ivfflat,
                           distances_naive,
                           distances_ivfflat,
                           ps.num_queries,
                           ps.k,
                           float(0.001),
                           min_recall));
    }
  }

  void SetUp() override
  {
    database.resize(ps.num_db_vecs * ps.dim, stream_);
    search_queries.resize(ps.num_queries * ps.dim, stream_);

    raft::random::Rng r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      r.uniform(database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0), stream_);
      r.uniform(search_queries.data(), ps.num_queries * ps.dim, DataT(0.1), DataT(2.0), stream_);
    } else {
      r.uniformInt(database.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20), stream_);
      r.uniformInt(search_queries.data(), ps.num_queries * ps.dim, DataT(1), DataT(20), stream_);
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
  raft::handle_t handle_;
  rmm::cuda_stream_view stream_;
  AnnIvfFlatInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

const std::vector<AnnIvfFlatInputs> inputs = {
  // test various dims (aligned and not aligned to vector sizes)
  {1000, 10000, 1, 16, 40, 1024, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 2, 16, 40, 1024, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 3, 16, 40, 1024, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 4, 16, 40, 1024, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 5, 16, 40, 1024, raft::distance::DistanceType::InnerProduct},
  {1000, 10000, 8, 16, 40, 1024, raft::distance::DistanceType::InnerProduct},

  // test dims that do not fit into kernel shared memory limits
  {1000, 10000, 2048, 16, 40, 1024, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 2049, 16, 40, 1024, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 2050, 16, 40, 1024, raft::distance::DistanceType::InnerProduct},
  {1000, 10000, 2051, 16, 40, 1024, raft::distance::DistanceType::InnerProduct},
  {1000, 10000, 2052, 16, 40, 1024, raft::distance::DistanceType::InnerProduct},
  {1000, 10000, 2053, 16, 40, 1024, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 2056, 16, 40, 1024, raft::distance::DistanceType::L2Expanded},

  // various random combinations
  {1000, 10000, 16, 10, 40, 1024, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 10, 50, 1024, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 10, 70, 1024, raft::distance::DistanceType::L2Expanded},
  {100, 10000, 16, 10, 20, 512, raft::distance::DistanceType::L2Expanded},
  {20, 100000, 16, 10, 20, 1024, raft::distance::DistanceType::L2Expanded},
  {1000, 100000, 16, 10, 20, 1024, raft::distance::DistanceType::L2Expanded},
  {10000, 131072, 8, 10, 20, 1024, raft::distance::DistanceType::L2Expanded},

  {1000, 10000, 16, 10, 40, 1024, raft::distance::DistanceType::InnerProduct},
  {1000, 10000, 16, 10, 50, 1024, raft::distance::DistanceType::InnerProduct},
  {1000, 10000, 16, 10, 70, 1024, raft::distance::DistanceType::InnerProduct},
  {100, 10000, 16, 10, 20, 512, raft::distance::DistanceType::InnerProduct},
  {20, 100000, 16, 10, 20, 1024, raft::distance::DistanceType::InnerProduct},
  {1000, 100000, 16, 10, 20, 1024, raft::distance::DistanceType::InnerProduct},
  {10000, 131072, 8, 10, 50, 1024, raft::distance::DistanceType::InnerProduct},

  {1000, 10000, 4096, 20, 50, 1024, raft::distance::DistanceType::InnerProduct},

  // test splitting the big query batches  (> max gridDim.y) into smaller batches
  {100000, 1024, 32, 10, 64, 64, raft::distance::DistanceType::InnerProduct},
  {98306, 1024, 32, 10, 64, 64, raft::distance::DistanceType::InnerProduct},

  // test radix_sort for getting the cluster selection
  {1000,
   10000,
   16,
   10,
   raft::spatial::knn::detail::topk::kMaxCapacity * 2,
   raft::spatial::knn::detail::topk::kMaxCapacity * 4,
   raft::distance::DistanceType::L2Expanded},
  {1000,
   10000,
   16,
   10,
   raft::spatial::knn::detail::topk::kMaxCapacity * 4,
   raft::spatial::knn::detail::topk::kMaxCapacity * 4,
   raft::distance::DistanceType::InnerProduct}};

typedef AnnIVFFlatTest<float, float> AnnIVFFlatTestF;
TEST_P(AnnIVFFlatTestF, AnnIVFFlat) { this->testIVFFlat(); }

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF, ::testing::ValuesIn(inputs));

typedef AnnIVFFlatTest<float, uint8_t> AnnIVFFlatTestF_uint8;
TEST_P(AnnIVFFlatTestF_uint8, AnnIVFFlat) { this->testIVFFlat(); }

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF_uint8, ::testing::ValuesIn(inputs));

typedef AnnIVFFlatTest<float, int8_t> AnnIVFFlatTestF_int8;
TEST_P(AnnIVFFlatTestF_int8, AnnIVFFlat) { this->testIVFFlat(); }

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF_int8, ::testing::ValuesIn(inputs));

}  // namespace knn
}  // namespace spatial
}  // namespace raft
