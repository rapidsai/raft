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

#include "../test_utils.h"
#include <gtest/gtest.h>
#include <optional>
#include <vector>

#include <raft/cluster/kmeans.cuh>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cuda_utils.cuh>

#if defined RAFT_DISTANCE_COMPILED && defined RAFT_NN_COMPILED
#include <raft/cluster/specializations.cuh>
#endif

namespace raft {

template <typename T>
struct KmeansFindKInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  bool weighted;
};

template <typename T>
class KmeansFindKTest : public ::testing::TestWithParam<KmeansFindKInputs<T>> {
 protected:
  KmeansFindKTest() : stream(handle.get_stream()), best_k(raft::make_host_scalar<int>(0)) {}

  void basicTest()
  {
    testparams = ::testing::TestWithParam<KmeansFindKInputs<T>>::GetParam();

    int n_samples  = testparams.n_row;
    int n_features = testparams.n_col;
    int n_clusters = testparams.n_clusters;

    auto X      = raft::make_device_matrix<T, int>(handle, n_samples, n_features);
    auto labels = raft::make_device_vector<int, int>(handle, n_samples);

    raft::random::make_blobs<T, int>(X.data_handle(),
                                     labels.data_handle(),
                                     n_samples,
                                     n_features,
                                     n_clusters,
                                     stream,
                                     true,
                                     nullptr,
                                     nullptr,
                                     T(.001),
                                     false,
                                     (T)-10.0f,
                                     (T)10.0f,
                                     (uint64_t)1234);

    auto inertia = raft::make_host_scalar<T>(0);
    auto n_iter  = raft::make_host_scalar<int>(0);

    auto X_view =
      raft::make_device_matrix_view<const T, int>(X.data_handle(), X.extent(0), X.extent(1));

    raft::cluster::kmeans::find_k<int, T>(
      handle, X_view, best_k.view(), inertia.view(), n_iter.view(), n_clusters);

    handle.sync_stream(stream);
  }

  void SetUp() override { basicTest(); }

 protected:
  raft::device_resources handle;
  cudaStream_t stream;
  KmeansFindKInputs<T> testparams;
  raft::host_scalar<int> best_k;
};

const std::vector<KmeansFindKInputs<float>> inputsf2 = {{1000, 32, 8, 0.001f, true},
                                                        {1000, 32, 8, 0.001f, false},
                                                        {1000, 100, 20, 0.001f, true},
                                                        {1000, 100, 20, 0.001f, false},
                                                        {10000, 32, 10, 0.001f, true},
                                                        {10000, 32, 10, 0.001f, false},
                                                        {10000, 100, 50, 0.001f, true},
                                                        {10000, 100, 50, 0.001f, false},
                                                        {10000, 500, 100, 0.001f, true},
                                                        {10000, 500, 100, 0.001f, false}};

const std::vector<KmeansFindKInputs<double>> inputsd2 = {{1000, 32, 5, 0.0001, true},
                                                         {1000, 32, 5, 0.0001, false},
                                                         {1000, 100, 20, 0.0001, true},
                                                         {1000, 100, 20, 0.0001, false},
                                                         {10000, 32, 10, 0.0001, true},
                                                         {10000, 32, 10, 0.0001, false},
                                                         {10000, 100, 50, 0.0001, true},
                                                         {10000, 100, 50, 0.0001, false},
                                                         {10000, 500, 100, 0.0001, true},
                                                         {10000, 500, 100, 0.0001, false}};

typedef KmeansFindKTest<float> KmeansFindKTestF;
TEST_P(KmeansFindKTestF, Result)
{
  if (best_k.view()[0] != testparams.n_clusters) {
    std::cout << best_k.view()[0] << " " << testparams.n_clusters << std::endl;
  }
  ASSERT_TRUE(best_k.view()[0] == testparams.n_clusters);
}

typedef KmeansFindKTest<double> KmeansFindKTestD;
TEST_P(KmeansFindKTestD, Result)
{
  if (best_k.view()[0] != testparams.n_clusters) {
    std::cout << best_k.view()[0] << " " << testparams.n_clusters << std::endl;
  }

  ASSERT_TRUE(best_k.view()[0] == testparams.n_clusters);
}

INSTANTIATE_TEST_CASE_P(KmeansFindKTests, KmeansFindKTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KmeansFindKTests, KmeansFindKTestD, ::testing::ValuesIn(inputsd2));

}  // namespace raft
