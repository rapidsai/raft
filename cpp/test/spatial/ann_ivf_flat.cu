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
#include <raft/distance/distance_type.hpp>
#include <raft/random/rng.cuh>
#include <raft/spatial/knn/ann.cuh>
#include <raft/spatial/knn/detail/common_faiss.h>

#include <raft/spatial/knn/knn.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

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
  size_t match_count = 0;
  for (size_t i = 0; i < rows; ++i) {
    for (size_t k = 0; k < cols; ++k) {
      size_t idx_k  = i * cols + k;  // row major assumption!
      auto act_idx  = act_idx_h.get()[idx_k];
      auto act_dist = act_dist_h.get()[idx_k];
      for (size_t j = 0; j < cols; ++j) {
        size_t idx    = i * cols + j;  // row major assumption!
        auto exp_idx  = exp_idx_h.get()[idx];
        auto exp_dist = exp_dist_h.get()[idx];
        idx_dist_pair exp_kvp(exp_idx, exp_dist, raft::CompareApprox<DistT>(eps));
        idx_dist_pair act_kvp(act_idx, act_dist, raft::CompareApprox<DistT>(eps));
        if (!(exp_kvp == act_kvp)) {
          // return testing::AssertionFailure()
          //        << "actual=" << act_kvp.idx << "," << act_kvp.dist << "!="
          //        << "expected" << exp_kvp.idx << "," << exp_kvp.dist << " @" << i << "," << j;
          // std::cout<< "actual = " << act_kvp.idx << "," << act_kvp.dist << " != "  <<
          //           " expected = " << exp_kvp.idx << "," << exp_kvp.dist << " @" << i
          //           << "," << j << std::endl;
        } else {
          match_count++;
          break;
        }
      }
    }
  }
  std::cout << "Recall = " << match_count << "/" << rows * cols << std::endl;
  return testing::AssertionSuccess();
}

template <typename T, typename DataT>
class AnnIVFFlatTest : public ::testing::TestWithParam<AnnIvfFlatInputs> {
 public:
  AnnIVFFlatTest()
    : stream_(handle_.get_stream()),
      params_(::testing::TestWithParam<AnnIvfFlatInputs>::GetParam()),
      database(params_.num_db_vecs * params_.dim, stream_),
      search_queries(params_.num_queries * params_.dim, stream_),
      raft_indices_(params_.num_queries * params_.k, stream_),
      raft_distances_(params_.num_queries * params_.k, stream_),
      faiss_indices_(params_.num_queries * params_.k, stream_),
      faiss_distances_(params_.num_queries * params_.k, stream_)
  {
    handle_.sync_stream(stream_);
    RAFT_CUDA_TRY(cudaMemsetAsync(database.data(), 0, database.size() * sizeof(DataT), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(search_queries.data(), 0, search_queries.size() * sizeof(DataT), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(raft_indices_.data(), 0, raft_indices_.size() * sizeof(int64_t), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(raft_distances_.data(), 0, raft_distances_.size() * sizeof(T), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(faiss_indices_.data(), 0, faiss_indices_.size() * sizeof(int64_t), stream_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(faiss_distances_.data(), 0, faiss_distances_.size() * sizeof(T), stream_));
    handle_.sync_stream(stream_);
  }

 protected:
  void testIVFFlat(bool is8bit)
  {
    handle_.sync_stream(stream_);
    if constexpr (std::is_same<DataT, uint8_t>{}) {
      naiveBfKnn<uint8_t, uint32_t>(faiss_distances_.data(),
                                    faiss_indices_.data(),
                                    search_queries.data(),
                                    database.data(),
                                    num_queries,
                                    num_db_vecs,
                                    dim,
                                    k_,
                                    metric,
                                    true,
                                    2.0f,
                                    stream_);
    } else if constexpr (std::is_same<DataT, int8_t>{}) {
      naiveBfKnn<int8_t, int32_t>(faiss_distances_.data(),
                                  faiss_indices_.data(),
                                  search_queries.data(),
                                  database.data(),
                                  num_queries,
                                  num_db_vecs,
                                  dim,
                                  k_,
                                  metric,
                                  true,
                                  2.0f,
                                  stream_);
    } else if constexpr (std::is_same<DataT, float>{}) {
      naiveBfKnn<float, float>(faiss_distances_.data(),
                               faiss_indices_.data(),
                               search_queries.data(),
                               database.data(),
                               num_queries,
                               num_db_vecs,
                               dim,
                               k_,
                               metric,
                               true,
                               2.0f,
                               stream_);
    }
    handle_.sync_stream(stream_);

    raft::spatial::knn::IVFFlatParam ivfParams;
    ivfParams.nprobe = nprobe_;
    ivfParams.nlist  = nlist_;
    raft::spatial::knn::knnIndex index;
    index.index   = nullptr;
    index.gpu_res = nullptr;

    approx_knn_build_index(handle_,
                           &index,
                           dynamic_cast<raft::spatial::knn::knnIndexParam*>(&ivfParams),
                           metric,
                           0,
                           database.data(),
                           num_db_vecs,
                           dim);
    handle_.sync_stream(stream_);
    approx_knn_search(handle_,
                      raft_distances_.data(),
                      raft_indices_.data(),
                      &index,
                      dynamic_cast<raft::spatial::knn::knnIndexParam*>(&ivfParams),
                      k_,
                      search_queries.data(),
                      num_queries);
    handle_.sync_stream(stream_);
    // verify.
    devArrMatchKnnPair(faiss_indices_.data(),
                       raft_indices_.data(),
                       faiss_distances_.data(),
                       raft_distances_.data(),
                       num_queries,
                       k_,
                       float(0.001),
                       stream_);
    handle_.sync_stream(stream_);
  }

  void SetUp() override
  {
    handle_.sync_stream(stream_);
    num_queries = params_.num_queries;
    num_db_vecs = params_.num_db_vecs;
    dim         = params_.dim;
    k_          = params_.k;
    metric      = params_.metric_;
    nprobe_     = params_.nprobe;
    nlist_      = params_.nlist;

    unsigned long long int seed = 1234ULL;
    raft::random::Rng r(seed);
    if constexpr (std::is_same<DataT, float>{}) {
      r.uniform(database.data(), num_db_vecs * dim, DataT(0.1), DataT(2.0), stream_);
      r.uniform(search_queries.data(), num_queries * dim, DataT(0.1), DataT(2.0), stream_);
    } else {
      r.uniformInt(database.data(), num_db_vecs * dim, DataT(1), DataT(20), stream_);
      r.uniformInt(search_queries.data(), num_queries * dim, DataT(1), DataT(20), stream_);
    }
    handle_.sync_stream(stream_);
  }

 private:
  raft::handle_t handle_;
  rmm::cuda_stream_view stream_;
  AnnIvfFlatInputs params_;
  int num_queries;
  int num_db_vecs;
  int dim;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
  rmm::device_uvector<int64_t> raft_indices_;
  rmm::device_uvector<T> raft_distances_;
  rmm::device_uvector<int64_t> faiss_indices_;
  rmm::device_uvector<T> faiss_distances_;
  int k_;
  int nprobe_;
  int nlist_;
  raft::distance::DistanceType metric;
};

const std::vector<AnnIvfFlatInputs> inputs = {
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

  {1000, 10000, 4096, 20, 50, 1024, raft::distance::DistanceType::InnerProduct}};

typedef AnnIVFFlatTest<float, float> AnnIVFFlatTestF;
TEST_P(AnnIVFFlatTestF, AnnIVFFlat) { this->testIVFFlat(false); }

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF, ::testing::ValuesIn(inputs));

typedef AnnIVFFlatTest<float, uint8_t> AnnIVFFlatTestF_uint8;
TEST_P(AnnIVFFlatTestF_uint8, AnnIVFFlat) { this->testIVFFlat(true); }

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF_uint8, ::testing::ValuesIn(inputs));

typedef AnnIVFFlatTest<float, int8_t> AnnIVFFlatTestF_int8;
TEST_P(AnnIVFFlatTestF_int8, AnnIVFFlat) { this->testIVFFlat(true); }

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF_int8, ::testing::ValuesIn(inputs));

}  // namespace knn
}  // namespace spatial
}  // namespace raft
