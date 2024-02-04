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

#include "../test_utils.cuh"
#include "../test_utils.h"
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/detail/compress_to_bits.cuh>
#include <raft/matrix/init.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/itertools.hpp>

namespace raft::distance::masked_nn::compress_to_bits {

/**
 * @brief Transpose and decompress 2D bitfield to boolean matrix
 *
 * Inverse operation of compress_to_bits
 *
 * @tparam T
 *
 * @parameter[in]  in       An `m x n` bitfield matrix. Row major.
 * @parameter      in_rows  The number of rows of `in`, i.e. `m`.
 * @parameter      in_cols  The number of cols of `in`, i.e. `n`.
 *
 * @parameter[out] out      An `(m * bits_per_elem) x n` boolean matrix.
 */
template <typename T = uint64_t, typename = std::enable_if_t<std::is_integral<T>::value>>
RAFT_KERNEL decompress_bits_kernel(const T* in, int in_rows, int in_cols, bool* out)
{
  constexpr int bits_per_element = 8 * sizeof(T);

  const size_t i = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t j = threadIdx.x + blockIdx.x * blockDim.x;

  if (in_rows <= i || in_cols <= j) { return; }

  const size_t out_rows = in_rows * bits_per_element;
  const size_t out_cols = in_cols;
  const size_t out_i    = i * bits_per_element;
  const size_t out_j    = j;

  if (out_rows <= out_i && out_cols <= out_j) { return; }

  T bitfield = in[i * in_cols + j];
  for (int bitpos = 0; bitpos < bits_per_element; ++bitpos) {
    bool bit                                 = ((T(1) << bitpos) & bitfield) != 0;
    out[(out_i + bitpos) * out_cols + out_j] = bit;
  }
}

/**
 * @brief Transpose and decompress 2D bitfield to boolean matrix
 *
 * Inverse operation of compress_to_bits
 *
 * @tparam T
 *
 * @parameter[in]  in       An `m x n` bitfield matrix. Row major.
 * @parameter      in_rows  The number of rows of `in`, i.e. `m`.
 * @parameter      in_cols  The number of cols of `in`, i.e. `n`.
 *
 * @parameter[out] out      An `n x (m * bits_per_elem)` boolean matrix.
 */
template <typename T = uint64_t, typename = std::enable_if_t<std::is_integral<T>::value>>
void decompress_bits(const raft::handle_t& handle, const T* in, int in_rows, int in_cols, bool* out)
{
  auto stream = resource::get_cuda_stream(handle);
  dim3 grid(raft::ceildiv(in_cols, 32), raft::ceildiv(in_rows, 32));
  dim3 block(32, 32);
  decompress_bits_kernel<<<grid, block, 0, stream>>>(in, in_rows, in_cols, out);
  RAFT_CUDA_TRY(cudaGetLastError());
}

// Params holds parameters for test case
struct Params {
  int m, n;
};

inline auto operator<<(std::ostream& os, const Params& p) -> std::ostream&
{
  return os << "m: " << p.m << ", n: " << p.n;
}

// Check that the following holds
//
//  decompress(compress(x)) == x
//
// for 2D boolean matrices x.
template <typename T>
void check_invertible(const Params& p)
{
  using raft::distance::detail::compress_to_bits;
  constexpr int bits_per_elem = sizeof(T) * 8;

  // Make m and n that are safe to ceildiv.
  int m = raft::round_up_safe(p.m, bits_per_elem);
  int n = p.n;

  // Generate random input
  raft::handle_t handle{};
  raft::random::RngState r(1ULL);
  auto in = raft::make_device_matrix<bool, int>(handle, m, n);
  raft::random::bernoulli(handle, r, in.data_handle(), m * n, 0.5f);

  int tmp_m = raft::ceildiv(m, bits_per_elem);
  int out_m = tmp_m * bits_per_elem;

  auto tmp = raft::make_device_matrix<T, int>(handle, tmp_m, n);
  auto out = raft::make_device_matrix<bool, int>(handle, out_m, n);

  resource::sync_stream(handle);
  RAFT_CUDA_TRY(cudaGetLastError());

  ASSERT_EQ(in.extent(0), out.extent(0)) << "M does not match";
  ASSERT_EQ(in.extent(1), out.extent(1)) << "N does not match";

  compress_to_bits(handle, in.view(), tmp.view());
  resource::sync_stream(handle);
  RAFT_CUDA_TRY(cudaGetLastError());

  decompress_bits(handle, tmp.data_handle(), tmp.extent(0), tmp.extent(1), out.data_handle());
  resource::sync_stream(handle);
  RAFT_CUDA_TRY(cudaGetLastError());

  // Check for differences.
  ASSERT_TRUE(raft::devArrMatch(in.data_handle(),
                                out.data_handle(),
                                in.extent(0) * in.extent(1),
                                raft::Compare<bool>(),
                                resource::get_cuda_stream(handle)));
  resource::sync_stream(handle);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void check_all_true(const Params& p)
{
  using raft::distance::detail::compress_to_bits;
  using T                     = uint64_t;
  constexpr int bits_per_elem = sizeof(T) * 8;

  // Make m and n that are safe to ceildiv.
  int m = raft::round_up_safe(p.m, bits_per_elem);
  int n = p.n;

  raft::handle_t handle{};
  raft::random::RngState r(1ULL);
  auto in = raft::make_device_matrix<bool, int>(handle, m, n);
  raft::matrix::fill(handle, in.view(), true);

  int tmp_m = raft::ceildiv(m, bits_per_elem);
  auto tmp  = raft::make_device_matrix<T, int>(handle, tmp_m, n);
  resource::sync_stream(handle);
  RAFT_CUDA_TRY(cudaGetLastError());

  compress_to_bits(handle, in.view(), tmp.view());
  resource::sync_stream(handle);
  RAFT_CUDA_TRY(cudaGetLastError());

  auto expected = raft::make_device_matrix<T, int>(handle, tmp_m, n);
  raft::matrix::fill(handle, expected.view(), ~T(0));

  // Check for differences.
  ASSERT_TRUE(raft::devArrMatch(expected.data_handle(),
                                tmp.data_handle(),
                                tmp.extent(0) * tmp.extent(1),
                                raft::Compare<T>(),
                                resource::get_cuda_stream(handle)));
  resource::sync_stream(handle);
  RAFT_CUDA_TRY(cudaGetLastError());
}

class CompressToBitsTest : public ::testing::TestWithParam<Params> {
  // Empty.
};

TEST_P(CompressToBitsTest, CheckTrue64) { check_all_true(GetParam()); }

TEST_P(CompressToBitsTest, CheckInvertible64)
{
  using T = uint64_t;
  check_invertible<T>(GetParam());
}

TEST_P(CompressToBitsTest, CheckInvertible32)
{
  using T = uint32_t;
  check_invertible<T>(GetParam());
}

std::vector<Params> params = raft::util::itertools::product<Params>(
  {1, 3, 32, 33, 63, 64, 65, 128, 10013}, {1, 3, 32, 33, 63, 64, 65, 13001});

INSTANTIATE_TEST_CASE_P(CompressToBits, CompressToBitsTest, ::testing::ValuesIn(params));

}  // namespace raft::distance::masked_nn::compress_to_bits