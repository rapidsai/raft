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
#include <gtest/gtest.h>
#include <optional>
#include <vector>

#include <raft/cluster/kmeans.cuh>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>
#include <thrust/fill.h>

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
  KmeansFindKTest() : stream(handle.get_stream()) {}

  void basicTest()
  {
    testparams = ::testing::TestWithParam<KmeansFindKInputs<T>>::GetParam();

    int n_samples  = testparams.n_row;
    int n_features = testparams.n_col;

    auto X      = raft::make_device_matrix<T, int>(handle, n_samples, n_features);
    auto labels = raft::make_device_vector<int, int>(handle, n_samples);

    raft::random::make_blobs<T, int>(X.data_handle(),
                                     labels.data_handle(),
                                     n_samples,
                                     n_features,
                                     params.n_clusters,
                                     stream,
                                     true,
                                     nullptr,
                                     nullptr,
                                     T(1.0),
                                     false,
                                     (T)-10.0f,
                                     (T)10.0f,
                                     (uint64_t)1234);

    //            std::optional<raft::device_vector_view<const T>> d_sw = std::nullopt;
    //            auto d_centroids_view =
    //                    raft::make_device_matrix_view<T, int>(d_centroids.data(),
    //                    params.n_clusters, n_features);
    //            if (testparams.weighted) {
    //                d_sample_weight.resize(n_samples, stream);
    //                d_sw = std::make_optional(
    //                        raft::make_device_vector_view<const T, int>(d_sample_weight.data(),
    //                        n_samples));
    //                thrust::fill(thrust::cuda::par.on(stream),
    //                             d_sample_weight.data(),
    //                             d_sample_weight.data() + n_samples,
    //                             1);
    //            }
    //
    auto best_k  = raft::make_host_scalar<int>();
    auto inertia = raft::make_host_scalar<int>();
    auto n_iter  = raft::make_host_scalar<int>();

    auto X_view =
      raft::make_device_matrix_view<const T, int>(X.data_handle(), X.extent(0), X.extent(1));

    raft::cluster::kmeans::find_k(
      handle, X_view, best_k.view(), inertia.view(), n_iter.view(), testparams.n_clusters + 2);

    handle.sync_stream(stream);

    assert(best_k[0] == testparams.n_clusters);
  }

  void SetUp() override { basicTest(); }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;
  KmeansFindKInputs<T> testparams;
};

const std::vector<KmeansFindKInputs<float>> inputsf2 = {{1000, 32, 5, 0.0001f, true},
                                                        {1000, 32, 5, 0.0001f, false},
                                                        {1000, 100, 20, 0.0001f, true},
                                                        {1000, 100, 20, 0.0001f, false},
                                                        {10000, 32, 10, 0.0001f, true},
                                                        {10000, 32, 10, 0.0001f, false},
                                                        {10000, 100, 50, 0.0001f, true},
                                                        {10000, 100, 50, 0.0001f, false},
                                                        {10000, 500, 100, 0.0001f, true},
                                                        {10000, 500, 100, 0.0001f, false}};

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
TEST_P(KmeansFindKTestF, Result) { ASSERT_TRUE(score == 1.0); }

typedef KmeansFindKTest<double> KmeansFindKTestD;
TEST_P(KmeansFindKTestD, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(KmeansFindKTests, KmeansFindKTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KmeansFindKTests, KmeansFindKTestD, ::testing::ValuesIn(inputsd2));

}  // namespace raft
