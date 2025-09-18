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
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/select_k.cuh>

#include <gtest/gtest.h>

#include <vector>

namespace raft::matrix {

template <typename DataT = float, typename IdxT = uint32_t>
auto run_max_k(const std::vector<DataT>& h_in, int64_t k)
{
  using raft::matrix::SelectAlgo;

  raft::device_resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  int64_t n_rows = 1;
  int64_t n_cols = h_in.size();

  // Device allocations (explicitly row_major optional)
  auto in_values =
    raft::make_device_matrix<DataT, int64_t, raft::row_major>(handle, n_rows, n_cols);
  auto out_values  = raft::make_device_matrix<DataT, int64_t, raft::row_major>(handle, n_rows, k);
  auto out_indices = raft::make_device_matrix<IdxT, int64_t, raft::row_major>(handle, n_rows, k);

  // Copy input to device
  raft::copy(in_values.data_handle(), h_in.data(), n_cols, stream);

  // Run select_k for largest k
  raft::matrix::select_k<DataT, IdxT>(handle,
                                      in_values.view(),
                                      std::nullopt,
                                      out_values.view(),
                                      out_indices.view(),
                                      /*select_min=*/false,
                                      /*sorted=*/true);

  // Copy results back to host
  std::vector<DataT> h_out(k);
  std::vector<IdxT> h_idx(k);
  raft::copy(h_out.data(), out_values.data_handle(), k, stream);
  raft::copy(h_idx.data(), out_indices.data_handle(), k, stream);
  raft::resource::sync_stream(handle, stream);

  return std::make_tuple(h_out, h_idx);
}

TEST(SelectK, EdgeNaN)
{
  int64_t k               = 3;
  std::vector<float> h_in = {0.5f, NAN, NAN, NAN, NAN};
  auto [h_out, h_idx]     = run_max_k(h_in, k);

  // NaN values are the largest values as per IEEE 754
  for (int i = 0; i < k; i++) {
    ASSERT_TRUE(std::isnan(h_out[i])) << "h_out[" << i << "] is not NaN";
    ASSERT_NE(h_idx[i], 0) << "h_idx[" << i << "] is 0";
  }
}

TEST(SelectK, EdgeMinusInfinity)
{
  int64_t k               = 3;
  std::vector<float> h_in = {0.5f, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
  auto [h_out, h_idx]     = run_max_k(h_in, k);

  // First element is the largest in this example
  ASSERT_EQ(h_out[0], 0.5f);
  ASSERT_EQ(h_idx[0], 0);
  // Rest are all -infinity
  for (int i = 1; i < k; i++) {
    ASSERT_TRUE(std::isinf(h_out[i]) && h_out[i] < 0) << "h_out[" << i << "] is not -infinity";
    ASSERT_NE(h_idx[i], 0) << "h_idx[" << i << "] is 0";
  }
}

TEST(SelectK, EdgePlusInfinity)
{
  int64_t k               = 3;
  std::vector<float> h_in = {0.5f, INFINITY, INFINITY, INFINITY, INFINITY};
  auto [h_out, h_idx]     = run_max_k(h_in, k);

  // All values are infinity and the first input element doesn't make it to the output
  for (int i = 0; i < k; i++) {
    ASSERT_TRUE(std::isinf(h_out[i]) && h_out[i] > 0) << "h_out[" << i << "] is not infinity";
    ASSERT_NE(h_idx[i], 0) << "h_idx[" << i << "] is 0";
  }
}

TEST(SelectK, EdgeInfNaN)
{
  int64_t k               = 3;
  std::vector<float> h_in = {0.5f, -INFINITY, NAN, -INFINITY, INFINITY};
  auto [h_out, h_idx]     = run_max_k(h_in, k);

  // The order is well defined here
  ASSERT_TRUE(std::isnan(h_out[0]));
  ASSERT_TRUE(std::isinf(h_out[1]) && h_out[1] > 0);
  ASSERT_EQ(h_out[2], 0.5f);

  ASSERT_EQ(h_idx[0], 2);
  ASSERT_EQ(h_idx[1], 4);
  ASSERT_EQ(h_idx[2], 0);
}

}  // namespace raft::matrix
