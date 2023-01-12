/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/StandardGpuResources.h>

#include <raft/core/device_mdspan.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/brute_force.cuh>
#include <raft/random/rng.cuh>
#include <raft/spatial/knn/detail/common_faiss.h>
#include <raft/spatial/knn/knn.cuh>

#if defined RAFT_NN_COMPILED
#include <raft/neighbors/specializations.cuh>
#endif

#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <vector>

namespace raft {
namespace spatial {
namespace knn {
struct FusedL2KNNInputs {
  int num_queries;
  int num_db_vecs;
  int dim;
  int k;
  raft::distance::DistanceType metric_;
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
testing::AssertionResult devArrMatchKnnPair(const T* expected_idx,
                                            const T* actual_idx,
                                            const DistT* expected_dist,
                                            const DistT* actual_dist,
                                            size_t rows,
                                            size_t cols,
                                            const DistT eps,
                                            cudaStream_t stream = 0)
{
  size_t size = rows * cols;
  std::unique_ptr<T[]> exp_idx_h(new T[size]);
  std::unique_ptr<T[]> act_idx_h(new T[size]);
  std::unique_ptr<DistT[]> exp_dist_h(new DistT[size]);
  std::unique_ptr<DistT[]> act_dist_h(new DistT[size]);
  raft::update_host<T>(exp_idx_h.get(), expected_idx, size, stream);
  raft::update_host<T>(act_idx_h.get(), actual_idx, size, stream);
  raft::update_host<DistT>(exp_dist_h.get(), expected_dist, size, stream);
  raft::update_host<DistT>(act_dist_h.get(), actual_dist, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < rows; ++i) {
    for (size_t j(0); j < cols; ++j) {
      auto idx      = i * cols + j;  // row major assumption!
      auto exp_idx  = exp_idx_h.get()[idx];
      auto act_idx  = act_idx_h.get()[idx];
      auto exp_dist = exp_dist_h.get()[idx];
      auto act_dist = act_dist_h.get()[idx];
      idx_dist_pair exp_kvp(exp_idx, exp_dist, raft::CompareApprox<DistT>(eps));
      idx_dist_pair act_kvp(act_idx, act_dist, raft::CompareApprox<DistT>(eps));
      if (!(exp_kvp == act_kvp)) {
        return testing::AssertionFailure()
               << "actual=" << act_kvp.idx << "," << act_kvp.dist << "!="
               << "expected" << exp_kvp.idx << "," << exp_kvp.dist << " @" << i << "," << j;
      }
    }
  }
  return testing::AssertionSuccess();
}

template <typename T>
class FusedL2KNNTest : public ::testing::TestWithParam<FusedL2KNNInputs> {
 public:
  FusedL2KNNTest()
    : stream_(handle_.get_stream()),
      params_(::testing::TestWithParam<FusedL2KNNInputs>::GetParam()),
      database(params_.num_db_vecs * params_.dim, stream_),
      search_queries(params_.num_queries * params_.dim, stream_),
      raft_indices_(params_.num_queries * params_.k, stream_),
      raft_distances_(params_.num_queries * params_.k, stream_),
      faiss_indices_(params_.num_queries * params_.k, stream_),
      faiss_distances_(params_.num_queries * params_.k, stream_)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(database.data(), 0, database.size() * sizeof(T), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(search_queries.data(), 0, search_queries.size() * sizeof(T), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(raft_indices_.data(), 0, raft_indices_.size() * sizeof(int64_t), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(raft_distances_.data(), 0, raft_distances_.size() * sizeof(T), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(faiss_indices_.data(), 0, faiss_indices_.size() * sizeof(int64_t), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(faiss_distances_.data(), 0, faiss_distances_.size() * sizeof(T), stream_));
  }

 protected:
  void testBruteForce()
  {
    launchFaissBfknn();

    auto index_view =
      raft::make_device_matrix_view<const T, int64_t>(database.data(), num_db_vecs, dim);
    auto query_view =
      raft::make_device_matrix_view<const T, int64_t>(search_queries.data(), num_queries, dim);
    auto out_indices_view =
      raft::make_device_matrix_view<int64_t, int64_t>(raft_indices_.data(), num_queries, k_);
    auto out_dists_view =
      raft::make_device_matrix_view<T, int64_t>(raft_distances_.data(), num_queries, k_);
    raft::neighbors::brute_force::fused_l2_knn(
      handle_, index_view, query_view, out_indices_view, out_dists_view, metric);

    // verify.
    devArrMatchKnnPair(faiss_indices_.data(),
                       raft_indices_.data(),
                       faiss_distances_.data(),
                       raft_distances_.data(),
                       num_queries,
                       k_,
                       float(0.001),
                       stream_);
  }

  void SetUp() override
  {
    num_queries = params_.num_queries;
    num_db_vecs = params_.num_db_vecs;
    dim         = params_.dim;
    k_          = params_.k;
    metric      = params_.metric_;

    unsigned long long int seed = 1234ULL;
    raft::random::RngState r(seed);
    uniform(handle_, r, database.data(), num_db_vecs * dim, T(-1.0), T(1.0));
    uniform(handle_, r, search_queries.data(), num_queries * dim, T(-1.0), T(1.0));
  }

  void launchFaissBfknn()
  {
    faiss::MetricType m = detail::build_faiss_metric(metric);

    faiss::gpu::StandardGpuResources gpu_res;

    gpu_res.noTempMemory();
    int device;
    RAFT_CUDA_TRY(cudaGetDevice(&device));
    gpu_res.setDefaultStream(device, stream_);

    faiss::gpu::GpuDistanceParams args;
    args.metric          = m;
    args.metricArg       = 0;
    args.k               = k_;
    args.dims            = dim;
    args.vectors         = database.data();
    args.vectorsRowMajor = true;
    args.numVectors      = num_db_vecs;
    args.queries         = search_queries.data();
    args.queriesRowMajor = true;
    args.numQueries      = num_queries;
    args.outDistances    = faiss_distances_.data();
    args.outIndices      = faiss_indices_.data();

    bfKnn(&gpu_res, args);
  }

 private:
  raft::device_resources handle_;
  cudaStream_t stream_ = 0;
  FusedL2KNNInputs params_;
  int num_queries;
  int num_db_vecs;
  int dim;
  rmm::device_uvector<T> database;
  rmm::device_uvector<T> search_queries;
  rmm::device_uvector<int64_t> raft_indices_;
  rmm::device_uvector<T> raft_distances_;
  rmm::device_uvector<int64_t> faiss_indices_;
  rmm::device_uvector<T> faiss_distances_;
  int k_;
  raft::distance::DistanceType metric;
};

const std::vector<FusedL2KNNInputs> inputs = {
  {100, 1000, 16, 10, raft::distance::DistanceType::L2Expanded},
  {256, 256, 30, 10, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 10, raft::distance::DistanceType::L2Expanded},
  {100, 1000, 16, 50, raft::distance::DistanceType::L2Expanded},
  {20, 10000, 16, 10, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 50, raft::distance::DistanceType::L2Expanded},
  {1000, 10000, 32, 50, raft::distance::DistanceType::L2Expanded},
  {10000, 40000, 32, 30, raft::distance::DistanceType::L2Expanded},
  {131072, 131072, 8, 60, raft::distance::DistanceType::L2Expanded},
  // L2 unexpanded
  {100, 1000, 16, 10, raft::distance::DistanceType::L2Unexpanded},
  {1000, 10000, 16, 10, raft::distance::DistanceType::L2Unexpanded},
  {100, 1000, 16, 50, raft::distance::DistanceType::L2Unexpanded},
  {20, 10000, 16, 50, raft::distance::DistanceType::L2Unexpanded},
  {1000, 10000, 16, 50, raft::distance::DistanceType::L2Unexpanded},
  {1000, 10000, 32, 50, raft::distance::DistanceType::L2Unexpanded},
  {10000, 40000, 32, 30, raft::distance::DistanceType::L2Unexpanded},
  {131072, 131072, 8, 60, raft::distance::DistanceType::L2Unexpanded}};

typedef FusedL2KNNTest<float> FusedL2KNNTestF;
TEST_P(FusedL2KNNTestF, FusedBruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(FusedL2KNNTest, FusedL2KNNTestF, ::testing::ValuesIn(inputs));

}  // namespace knn
}  // namespace spatial
}  // namespace raft
