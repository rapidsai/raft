/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include "select_k.cuh"

#include <cuda_bf16.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <iostream>

namespace raft::matrix {

auto inputs_random_longlist = testing::Values(select::params{1, 130, 15, false},
                                              select::params{1, 128, 15, false},
                                              select::params{20, 700, 1, true},
                                              select::params{20, 700, 2, true},
                                              select::params{20, 700, 3, true},
                                              select::params{20, 700, 4, true},
                                              select::params{20, 700, 5, true},
                                              select::params{20, 700, 6, true},
                                              select::params{20, 700, 7, true},
                                              select::params{20, 700, 8, true},
                                              select::params{20, 700, 9, true},
                                              select::params{20, 700, 10, true, false},
                                              select::params{20, 700, 11, true},
                                              select::params{20, 700, 12, true},
                                              select::params{20, 700, 16, true},
                                              select::params{100, 1700, 17, true},
                                              select::params{100, 1700, 31, true, false},
                                              select::params{100, 1700, 32, false},
                                              select::params{100, 1700, 33, false},
                                              select::params{100, 1700, 63, false},
                                              select::params{100, 1700, 64, false, false},
                                              select::params{100, 1700, 65, false},
                                              select::params{100, 1700, 255, true},
                                              select::params{100, 1700, 256, true},
                                              select::params{100, 1700, 511, false},
                                              select::params{100, 1700, 512, true},
                                              select::params{100, 1700, 1023, false, false},
                                              select::params{100, 1700, 1024, true},
                                              select::params{100, 1700, 1700, true});

auto inputs_random_largesize = testing::Values(select::params{100, 100000, 1, true},
                                               select::params{100, 100000, 2, true},
                                               select::params{100, 100000, 3, true, false},
                                               select::params{100, 100000, 7, true},
                                               select::params{100, 100000, 16, true},
                                               select::params{100, 100000, 31, true},
                                               select::params{100, 100000, 32, true, false},
                                               select::params{100, 100000, 60, true},
                                               select::params{100, 100000, 100, true, false},
                                               select::params{100, 100000, 200, true},
                                               select::params{100000, 100, 100, false},
                                               select::params{1, 1000000000, 1, true},
                                               select::params{1, 1000000000, 16, false, false},
                                               select::params{1, 1000000000, 64, false},
                                               select::params{1, 1000000000, 128, true, false},
                                               select::params{1, 1000000000, 256, false, false});

auto inputs_random_largek = testing::Values(select::params{100, 100000, 1000, true},
                                            select::params{100, 100000, 2000, false},
                                            select::params{100, 100000, 100000, true, false},
                                            select::params{100, 100000, 2048, false},
                                            select::params{100, 100000, 1237, true});

auto inputs_random_many_infs =
  testing::Values(select::params{10, 100000, 1, true, false, false, true, 0.9},
                  select::params{10, 100000, 16, true, false, false, true, 0.9},
                  select::params{10, 100000, 64, true, false, false, true, 0.9},
                  select::params{10, 100000, 128, true, false, false, true, 0.9},
                  select::params{10, 100000, 256, true, false, false, true, 0.9},
                  select::params{1000, 10000, 1, true, false, false, true, 0.9},
                  select::params{1000, 10000, 16, true, false, false, true, 0.9},
                  select::params{1000, 10000, 64, true, false, false, true, 0.9},
                  select::params{1000, 10000, 128, true, false, false, true, 0.9},
                  select::params{1000, 10000, 256, true, false, false, true, 0.9},
                  select::params{10, 100000, 1, true, false, false, true, 0.999},
                  select::params{10, 100000, 16, true, false, false, true, 0.999},
                  select::params{10, 100000, 64, true, false, false, true, 0.999},
                  select::params{10, 100000, 128, true, false, false, true, 0.999},
                  select::params{10, 100000, 256, true, false, false, true, 0.999},
                  select::params{1000, 10000, 1, true, false, false, true, 0.999},
                  select::params{1000, 10000, 16, true, false, false, true, 0.999},
                  select::params{1000, 10000, 64, true, false, false, true, 0.999},
                  select::params{1000, 10000, 128, true, false, false, true, 0.999},
                  select::params{1000, 10000, 256, true, false, false, true, 0.999});

using ReferencedRandomFloatInt =
  SelectK<float, uint32_t, with_ref<SelectAlgo::kAuto>::params_random>;
TEST_P(ReferencedRandomFloatInt, Run) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                          // NOLINT
  SelectK,
  ReferencedRandomFloatInt,
  testing::Combine(inputs_random_longlist,
                   testing::Values(SelectAlgo::kRadix8bits,
                                   SelectAlgo::kRadix11bits,
                                   SelectAlgo::kRadix11bitsExtraPass,
                                   SelectAlgo::kWarpImmediate,
                                   SelectAlgo::kWarpFiltered,
                                   SelectAlgo::kWarpDistributed,
                                   SelectAlgo::kWarpDistributedShm)));

using ReferencedRandomDoubleSizeT =
  SelectK<double, int64_t, with_ref<SelectAlgo::kAuto>::params_random>;
TEST_P(ReferencedRandomDoubleSizeT, Run) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                             // NOLINT
  SelectK,
  ReferencedRandomDoubleSizeT,
  testing::Combine(inputs_random_longlist,
                   testing::Values(SelectAlgo::kRadix8bits,
                                   SelectAlgo::kRadix11bits,
                                   SelectAlgo::kRadix11bitsExtraPass,
                                   SelectAlgo::kWarpImmediate,
                                   SelectAlgo::kWarpFiltered,
                                   SelectAlgo::kWarpDistributed,
                                   SelectAlgo::kWarpDistributedShm)));

using ReferencedRandomDoubleInt =
  SelectK<double, uint32_t, with_ref<SelectAlgo::kRadix11bits>::params_random>;
TEST_P(ReferencedRandomDoubleInt, LargeSize) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                                 // NOLINT
  SelectK,
  ReferencedRandomDoubleInt,
  testing::Combine(inputs_random_largesize,
                   testing::Values(SelectAlgo::kWarpAuto,
                                   SelectAlgo::kRadix8bits,
                                   SelectAlgo::kRadix11bits,
                                   SelectAlgo::kRadix11bitsExtraPass)));

using ReferencedRandomFloatIntkWarpsortAsGT =
  SelectK<float, uint32_t, with_ref<SelectAlgo::kWarpImmediate>::params_random>;
TEST_P(ReferencedRandomFloatIntkWarpsortAsGT, Run) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                                       // NOLINT
  SelectK,
  ReferencedRandomFloatIntkWarpsortAsGT,
  testing::Combine(inputs_random_many_infs,
                   testing::Values(SelectAlgo::kRadix8bits,
                                   SelectAlgo::kRadix11bits,
                                   SelectAlgo::kRadix11bitsExtraPass)));


TEST(SelectKBF16Negative, TopK3)
{
  raft::device_resources handle;
  auto stream = resource::get_cuda_stream(handle);

  using DataT = nv_bfloat16;
  using IdxT  = uint32_t;

  // Input row with negatives and positives
  std::vector<float> h_in = {-3.5f, -1.5f, -7.5f, 0.5f, -2.5f};
  int64_t n_rows = 1;
  int64_t n_cols = h_in.size();
  int64_t k      = 3;

  // Convert host input to bf16
  std::vector<DataT> h_in_bf16(n_cols);
  for (int i = 0; i < n_cols; i++) {
    h_in_bf16[i] = __float2bfloat16(h_in[i]);
  }

  // Device allocations (explicitly row_major optional)
  auto in_values   = raft::make_device_matrix<DataT, int64_t, row_major>(handle, n_rows, n_cols);
  auto out_values  = raft::make_device_matrix<DataT, int64_t, row_major>(handle, n_rows, k);
  auto out_indices = raft::make_device_matrix<IdxT, int64_t, row_major>(handle, n_rows, k);

  // Copy input to device
  raft::copy(in_values.data_handle(), h_in_bf16.data(), n_cols, stream);

  // Run select_k for largest k
  raft::matrix::select_k<DataT, IdxT>(
    handle, in_values.view(), std::nullopt, out_values.view(), out_indices.view(),
    /*select_min=*/false, /*sorted=*/true);

  // Copy results back to host
  std::vector<DataT> h_out(k);
  std::vector<IdxT> h_idx(k);
  raft::copy(h_out.data(), out_values.data_handle(), k, stream);
  raft::copy(h_idx.data(), out_indices.data_handle(), k, stream);
  resource::sync_stream(handle, stream);

  // Debug print
  std::cout << "Input: ";
  for (auto v : h_in) std::cout << v << " ";
  std::cout << "\nSelected top-" << k << ": ";
  for (int i = 0; i < k; i++) {
    std::cout << __bfloat162float(h_out[i]) << " (idx=" << h_idx[i] << ") ";
  }
  std::cout << std::endl;

  // Expected top-3 largest values: [0.5, -1.5, -2.5]
  EXPECT_NEAR(__bfloat162float(h_out[0]), 0.5f, 0.01f);
  EXPECT_NEAR(__bfloat162float(h_out[1]), -1.5f, 0.01f);
  EXPECT_NEAR(__bfloat162float(h_out[2]), -2.5f, 0.01f);
}

}  // namespace raft::matrix
