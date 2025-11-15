/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/spatial/knn/detail/haversine_distance.cuh>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <vector>

namespace raft {
namespace spatial {
namespace knn {

template <typename value_idx, typename value_t>
class HaversineKNNTest : public ::testing::Test {
 public:
  HaversineKNNTest()
    : stream(resource::get_cuda_stream(handle)),
      d_train_inputs(0, stream),
      d_ref_I(0, stream),
      d_ref_D(0, stream),
      d_pred_I(0, stream),
      d_pred_D(0, stream)
  {
  }

 protected:
  void basicTest()
  {
    // Allocate input
    d_train_inputs.resize(n * d, stream);

    // Allocate reference arrays
    d_ref_I.resize(n * n, stream);
    d_ref_D.resize(n * n, stream);

    // Allocate predicted arrays
    d_pred_I.resize(n * n, stream);
    d_pred_D.resize(n * n, stream);

    // make testdata on host
    std::vector<value_t> h_train_inputs = {0.71113885,
                                           -1.29215058,
                                           0.59613176,
                                           -2.08048115,
                                           0.74932804,
                                           -1.33634042,
                                           0.51486728,
                                           -1.65962873,
                                           0.53154002,
                                           -1.47049808,
                                           0.72891737,
                                           -1.54095137};

    h_train_inputs.resize(d_train_inputs.size());
    raft::update_device(
      d_train_inputs.data(), h_train_inputs.data(), d_train_inputs.size(), stream);

    std::vector<value_t> h_res_D = {0., 0.05041587, 0.18767063, 0.23048252, 0.35749438, 0.62925595,
                                    0., 0.36575755, 0.44288665, 0.5170737,  0.59501296, 0.62925595,
                                    0., 0.05041587, 0.152463,   0.2426416,  0.34925285, 0.59501296,
                                    0., 0.16461092, 0.2345792,  0.34925285, 0.35749438, 0.36575755,
                                    0., 0.16461092, 0.20535265, 0.23048252, 0.2426416,  0.5170737,
                                    0., 0.152463,   0.18767063, 0.20535265, 0.2345792,  0.44288665};
    h_res_D.resize(n * n);
    raft::update_device(d_ref_D.data(), h_res_D.data(), n * n, stream);

    std::vector<value_idx> h_res_I = {0, 2, 5, 4, 3, 1, 1, 3, 5, 4, 2, 0, 2, 0, 5, 4, 3, 1,
                                      3, 4, 5, 2, 0, 1, 4, 3, 5, 0, 2, 1, 5, 2, 0, 4, 3, 1};
    h_res_I.resize(n * n);
    raft::update_device<value_idx>(d_ref_I.data(), h_res_I.data(), n * n, stream);

    raft::spatial::knn::detail::haversine_knn(d_pred_I.data(),
                                              d_pred_D.data(),
                                              d_train_inputs.data(),
                                              d_train_inputs.data(),
                                              n,
                                              n,
                                              k,
                                              stream);

    resource::sync_stream(handle, stream);
  }

  void SetUp() override { basicTest(); }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  rmm::device_uvector<value_t> d_train_inputs;

  int n = 6;
  int d = 2;

  int k = 6;

  rmm::device_uvector<value_idx> d_pred_I;
  rmm::device_uvector<value_t> d_pred_D;

  rmm::device_uvector<value_idx> d_ref_I;
  rmm::device_uvector<value_t> d_ref_D;
};

typedef HaversineKNNTest<int, float> HaversineKNNTestF;

TEST_F(HaversineKNNTestF, Fit)
{
  ASSERT_TRUE(raft::devArrMatch(
    d_ref_D.data(), d_pred_D.data(), n * n, raft::CompareApprox<float>(1e-3), stream));
  ASSERT_TRUE(
    raft::devArrMatch(d_ref_I.data(), d_pred_I.data(), n * n, raft::Compare<int>(), stream));
}

}  // namespace knn
}  // namespace spatial
}  // namespace raft
