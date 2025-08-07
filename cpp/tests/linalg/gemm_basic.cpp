/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/gemm.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace raft::linalg {

// Matrix dimensions: A (2x3) * B (3x2) = C (2x2)
constexpr int M = 2, N = 2, K = 3;

// Non-trivial alpha and beta constants
float alpha_val = 2.0f;
float beta_val  = 3.0f;

// Input matrices with small integer values (stored as float)
// Matrix A (2x3):
// [ 1,  2,  3]
// [ 4,  5,  6]
std::vector<float> a_host = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

// Matrix B (3x2):
// [ 7,  8]
// [ 9, 10]
// [11, 12]
std::vector<float> b_host = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

// Initial matrix C (2x2):
// [1, 2]
// [3, 4]
std::vector<float> c_host = {1.0f, 2.0f, 3.0f, 4.0f};

// Result matrix (2x2):
// [119, 134]
// [287, 320]
std::vector<float> r_host = {119.0f, 134.0f, 287.0f, 320.0f};

// Result matrix without coefficients (2x2):
// A * B result when alpha=1, beta=0 (default values)
// [58, 64]
// [139, 154]
std::vector<float> r_host_nocoefs = {58.0f, 64.0f, 139.0f, 154.0f};

TEST(Raft, GemmPointerModeHost)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  // Create device matrices
  auto a_device = raft::make_device_matrix<float>(res, M, K);
  auto b_device = raft::make_device_matrix<float>(res, K, N);
  auto c_device = raft::make_device_matrix<float>(res, M, N);

  // Copy data to device
  raft::copy(a_device.data_handle(), a_host.data(), a_host.size(), stream);
  raft::copy(b_device.data_handle(), b_host.data(), b_host.size(), stream);
  raft::copy(c_device.data_handle(), c_host.data(), c_host.size(), stream);

  // Create scalar views for alpha and beta
  auto alpha_scalar = raft::make_host_scalar(alpha_val);
  auto beta_scalar  = raft::make_host_scalar(beta_val);

  // Perform GEMM: C = alpha * A * B + beta * C
  raft::linalg::gemm(res,
                     a_device.view(),
                     b_device.view(),
                     c_device.view(),
                     std::make_optional(alpha_scalar.view()),
                     std::make_optional(beta_scalar.view()));

  // Copy result back to host
  std::vector<float> result(M * N);
  raft::copy(result.data(), c_device.data_handle(), result.size(), stream);
  raft::resource::sync_stream(res);

  // Compare results
  for (int i = 0; i < M * N; ++i) {
    EXPECT_FLOAT_EQ(result[i], r_host[i]) << "Mismatch at index " << i;
  }
}

TEST(Raft, GemmPointerModeDevice)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  // Create device matrices
  auto a_device = raft::make_device_matrix<float>(res, M, K);
  auto b_device = raft::make_device_matrix<float>(res, K, N);
  auto c_device = raft::make_device_matrix<float>(res, M, N);

  // Copy data to device
  raft::copy(a_device.data_handle(), a_host.data(), a_host.size(), stream);
  raft::copy(b_device.data_handle(), b_host.data(), b_host.size(), stream);
  raft::copy(c_device.data_handle(), c_host.data(), c_host.size(), stream);

  // Create scalar views for alpha and beta
  auto alpha_scalar = raft::make_device_scalar(res, alpha_val);
  auto beta_scalar  = raft::make_device_scalar(res, beta_val);

  // Perform GEMM: C = alpha * A * B + beta * C
  raft::linalg::gemm(res,
                     a_device.view(),
                     b_device.view(),
                     c_device.view(),
                     std::make_optional(alpha_scalar.view()),
                     std::make_optional(beta_scalar.view()));

  // Copy result back to host
  std::vector<float> result(M * N);
  raft::copy(result.data(), c_device.data_handle(), result.size(), stream);
  raft::resource::sync_stream(res);

  // Compare results
  for (int i = 0; i < M * N; ++i) {
    EXPECT_FLOAT_EQ(result[i], r_host[i]) << "Mismatch at index " << i;
  }
}

TEST(Raft, GemmNoEnforcedPointerMode)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  // Create device matrices
  auto a_device = raft::make_device_matrix<float>(res, M, K);
  auto b_device = raft::make_device_matrix<float>(res, K, N);
  auto c_device = raft::make_device_matrix<float>(res, M, N);

  // Copy data to device
  raft::copy(a_device.data_handle(), a_host.data(), a_host.size(), stream);
  raft::copy(b_device.data_handle(), b_host.data(), b_host.size(), stream);
  raft::copy(c_device.data_handle(), c_host.data(), c_host.size(), stream);

  // Perform GEMM: C = alpha * A * B + beta * C
  raft::linalg::gemm(res, a_device.view(), b_device.view(), c_device.view());

  // Copy result back to host
  std::vector<float> result(M * N);
  raft::copy(result.data(), c_device.data_handle(), result.size(), stream);
  raft::resource::sync_stream(res);

  // Compare results
  for (int i = 0; i < M * N; ++i) {
    EXPECT_FLOAT_EQ(result[i], r_host_nocoefs[i]) << "Mismatch at index " << i;
  }
}

}  // namespace raft::linalg
