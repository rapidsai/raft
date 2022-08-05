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
#include <raft/sparse/detail/utils.h>
#include <raft/spatial/knn/ivf_pq.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <vector>

namespace raft {
namespace spatial {
namespace knn {
struct IvfPqInputs {
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

#define CUANN_CHECK(ret) RAFT_EXPECTS(ret == ivf_pq::CUANN_STATUS_SUCCESS, "cuann failure: %d", ret)

template <typename T, typename DataT>
class IvfPqTest : public ::testing::TestWithParam<IvfPqInputs> {
 public:
  IvfPqTest()
    : stream_(handle_.get_stream()),
      ps(::testing::TestWithParam<IvfPqInputs>::GetParam()),
      database(0, stream_, &managed_memory),
      search_queries(0, stream_)
  {
  }

 protected:
  void testIvfPq()
  {
    size_t queries_size = ps.num_queries * ps.k;
    std::vector<int64_t> indices_ivf_pq(queries_size);
    std::vector<int64_t> indices_naive(queries_size);
    std::vector<T> distances_ivf_pq(queries_size);
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

      rmm::device_uvector<T> distances_ivf_pq_dev(queries_size, stream_);
      rmm::device_uvector<uint64_t> indices_ivf_pq_dev(queries_size, stream_);

      {
        std::unique_ptr<ivf_pq::cuannContext, std::function<void(ivf_pq::cuannHandle_t)>>
          cuann_handle{[]() {
                         ivf_pq::cuannHandle_t h;
                         CUANN_CHECK(ivf_pq::cuannCreate(&h));
                         return h;
                       }(),
                       [](ivf_pq::cuannHandle_t h) { ivf_pq::cuannDestroy(h); }};
        std::unique_ptr<ivf_pq::cuannIvfPqDescriptor,
                        std::function<void(ivf_pq::cuannIvfPqDescriptor_t)>>
          cuann_desc{
            []() {
              ivf_pq::cuannIvfPqDescriptor_t d;
              CUANN_CHECK(ivf_pq::cuannIvfPqCreateDescriptor(&d));
              return d;
            }(),
            [](ivf_pq::cuannIvfPqDescriptor_t d) { ivf_pq::cuannIvfPqDestroyDescriptor(d); }};

        CUANN_CHECK(ivf_pq::cuannSetDevice(cuann_handle.get(), handle_.get_device()));
        CUANN_CHECK(ivf_pq::cuannSetStream(cuann_handle.get(), handle_.get_stream()));

        // Number of kmeans clusters.
        //
        // The number of vectors per cluster, or 'numDataset' / 'numClusters',
        // should be approximately 1,000 to 10,000.
        uint32_t n_clusters = ps.nlist;
        // Important parameters of the index to create.
        //
        // 'bitPq' is the bit length of the vector element after compression by PQ.
        // 'dimPq' is the dimensionality of the vector after compression by PQ.
        //
        // 'bitPq' is 4, 5, 6, 7, or 8. The smaller the 'bitPq', the smaller the
        // index size and the better the search performance, but the lower the recall.
        //
        // Similarly, a smaller 'dimPq' results in a smaller index size and better
        // search performance, but lower recall. If 'bitPq' is 8, 'dimPq' can be set
        // to any number, but multiple of 8 are desirable for good performance.
        // If 'bitPq' is not 8, 'dimPq' must be basically multiple of 8. For good
        // performance, multiple 32 is desirable.
        //
        uint32_t bitPq = 8;
        uint32_t dimPq = ps.dim;
        if (dimPq >= 128) {
          dimPq = raft::alignDown<uint32_t>(dimPq / 2, 32);
        } else if (dimPq >= 32) {
          dimPq = raft::alignDown<uint32_t>(dimPq, 32);
        } else if (dimPq >= 8) {
          dimPq = raft::alignDown<uint32_t>(dimPq, 8);
        }
        // If true, dataset and query vectors are rotated by random rotation matrix
        // created at indexing time.
        //
        bool randomRotation = ps.dim < 1024;  // disable for large-dimensional data (CPU intensive)
        // Number of iterations for kmeans training.
        uint32_t numIterations = 25;
        // metric
        ivf_pq::cuannSimilarity_t similarity =
          ps.metric == raft::distance::DistanceType::InnerProduct ? ivf_pq::CUANN_SIMILARITY_INNER
                                                                  : ivf_pq::CUANN_SIMILARITY_L2;
        // Specify whether PQ codebooks are created per subspace or per cluster.
        ivf_pq::cuannPqCenter_t typePqCenter = ivf_pq::CUANN_PQ_CENTER_PER_SUBSPACE;
        CUANN_CHECK(ivf_pq::cuannIvfPqSetIndexParameters(
          cuann_desc.get(),
          n_clusters,               /* Number of clusters */
          uint32_t(ps.num_db_vecs), /* Number of dataset entries */
          uint32_t(ps.dim),         /* Dimension of each entry */
          dimPq,                    /* Dimension of each entry after product quantization */
          bitPq,                    /* Bit length of PQ */
          similarity,
          typePqCenter));

        // Allocate memory for index
        size_t ivf_pq_index_size;
        CUANN_CHECK(ivf_pq::cuannIvfPqGetIndexSize(cuann_desc.get(), &ivf_pq_index_size));
        rmm::device_buffer ivf_pq_index_buf_managed(ivf_pq_index_size, stream_, &managed_memory);

        // Build index
        cudaDataType_t dtype;
        if constexpr (std::is_same_v<DataT, uint8_t>) {
          dtype = CUDA_R_8U;
        } else if constexpr (std::is_same_v<DataT, int8_t>) {
          dtype = CUDA_R_8I;
        } else if constexpr (std::is_same_v<DataT, float>) {
          dtype = CUDA_R_32F;
        }
        CUANN_CHECK(ivf_pq::cuannIvfPqBuildIndex(
          cuann_handle.get(),
          cuann_desc.get(),
          database.data(),  // dataset
          database.data(),  // ?kmeans? trainset
          dtype,
          uint32_t(ps.num_db_vecs),  // size of the trainset (I guess for kmeans)
          numIterations,
          randomRotation,
          true,                            // hierarchialClustering: always true in raft
          ivf_pq_index_buf_managed.data()  // memory allocated for the index
          ));
        handle_.sync_stream(stream_);

        // set search parameters
        CUANN_CHECK(ivf_pq::cuannIvfPqSetSearchParameters(cuann_desc.get(), ps.nprobe, ps.k));
        // Data type of LUT to be created dynamically at search time.
        //
        // The use of low-precision types reduces the amount of shared memory
        // required at search time, so fast shared memory kernels can be used even
        // for datasets with large dimansionality. Note that the recall is slightly
        // degraded when low-precision type is selected.
        //
        cudaDataType_t smemLutDtype = CUDA_R_32F;
        // smemLutDtype = CUDA_R_16F;
        // smemLutDtype = CUDA_R_8U;
        // Storage data type for distance/similarity computed at search time.
        //
        // If the performance limiter at search time is device memory access,
        // selecting FP16 will improve performance slightly.
        //
        cudaDataType_t internalDistanceDtype = CUDA_R_32F;
        // internalDistanceDtype = CUDA_R_16F;

        // Thread block size of the distance calculation kernel at search time.
        //
        // If 0, the thread block size is determined automatically.
        //
        uint32_t preferredThreadBlockSize = 0;  // 0, 256, 512, or 1024
        CUANN_CHECK(ivf_pq::cuannIvfPqSetSearchTuningParameters(
          cuann_desc.get(), internalDistanceDtype, smemLutDtype, preferredThreadBlockSize));
        // Maximum number of query vectors to search.
        uint32_t maxQueries = 1000000;
        // Maximum number of query vectors to search at the same time.
        uint32_t batchSize = maxQueries;
        // Maximum device memory size that may be used as workspace at search time.
        // maxSearchWorkspaceSize = 0;  // default
        size_t maxSearchWorkspaceSize = (size_t)2 * 1024 * 1024 * 1024;  // 2 GiB

        // Allocate memory for index
        size_t ivf_pq_search_workspace_size;
        CUANN_CHECK(ivf_pq::cuannIvfPqSearch_bufferSize(cuann_handle.get(),
                                                        cuann_desc.get(),
                                                        ivf_pq_index_buf_managed.data(),
                                                        batchSize,
                                                        maxSearchWorkspaceSize,
                                                        &ivf_pq_search_workspace_size));
        rmm::device_buffer ivf_pq_search_ws_buf(ivf_pq_search_workspace_size, stream_);

        // finally, search!
        CUANN_CHECK(cuannIvfPqSearch(cuann_handle.get(),
                                     cuann_desc.get(),
                                     ivf_pq_index_buf_managed.data(),
                                     search_queries.data(),
                                     dtype,
                                     ps.num_queries,
                                     indices_ivf_pq_dev.data(),
                                     distances_ivf_pq_dev.data(),
                                     ivf_pq_search_ws_buf.data()));
        handle_.sync_stream(stream_);

        update_host(distances_ivf_pq.data(), distances_ivf_pq_dev.data(), queries_size, stream_);
        update_host(indices_ivf_pq.data(),
                    reinterpret_cast<int64_t*>(indices_ivf_pq_dev.data()),
                    queries_size,
                    stream_);
        handle_.sync_stream(stream_);
      }
      handle_.sync_stream(stream_);
      ASSERT_TRUE(eval_knn(indices_naive,
                           indices_ivf_pq,
                           distances_naive,
                           distances_ivf_pq,
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
  rmm::mr::managed_memory_resource managed_memory;
  IvfPqInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

const std::vector<IvfPqInputs> inputs = {
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

typedef IvfPqTest<float, float> IvfPqTestF;
TEST_P(IvfPqTestF, IvfPq) { this->testIvfPq(); }

INSTANTIATE_TEST_CASE_P(IvfPqTest, IvfPqTestF, ::testing::ValuesIn(inputs));

typedef IvfPqTest<float, uint8_t> IvfPqTestF_uint8;
TEST_P(IvfPqTestF_uint8, IvfPq) { this->testIvfPq(); }

INSTANTIATE_TEST_CASE_P(IvfPqTest, IvfPqTestF_uint8, ::testing::ValuesIn(inputs));

typedef IvfPqTest<float, int8_t> IvfPqTestF_int8;
TEST_P(IvfPqTestF_int8, IvfPq) { this->testIvfPq(); }

INSTANTIATE_TEST_CASE_P(IvfPqTest, IvfPqTestF_int8, ::testing::ValuesIn(inputs));

}  // namespace knn
}  // namespace spatial
}  // namespace raft
