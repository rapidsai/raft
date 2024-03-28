/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <raft/sparse/linalg/degree.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <iostream>

namespace raft {
namespace sparse {

template <typename T>
struct SparseDegreeInputs {
  int m, n, nnz;
  unsigned long long int seed;
};

template <typename T>
class SparseDegreeTests : public ::testing::TestWithParam<SparseDegreeInputs<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
  SparseDegreeInputs<T> params;
};

const std::vector<SparseDegreeInputs<float>> inputsf = {{5, 10, 5, 1234ULL}};

typedef SparseDegreeTests<float> COODegree;
TEST_P(COODegree, Result)
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int in_rows_h[5] = {0, 0, 1, 2, 2};
  int verify_h[5]  = {2, 1, 2, 0, 0};

  rmm::device_uvector<int> in_rows(5, stream);
  rmm::device_uvector<int> verify(5, stream);
  rmm::device_uvector<int> results(5, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(verify.data(), 0, verify.size() * sizeof(int), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(results.data(), 0, results.size() * sizeof(int), stream));

  raft::update_device(in_rows.data(), *&in_rows_h, 5, stream);
  raft::update_device(verify.data(), *&verify_h, 5, stream);

  linalg::coo_degree(in_rows.data(), 5, results.data(), stream);
  cudaDeviceSynchronize();

  ASSERT_TRUE(raft::devArrMatch<int>(verify.data(), results.data(), 5, raft::Compare<int>()));

  RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}

typedef SparseDegreeTests<float> COODegreeNonzero;
TEST_P(COODegreeNonzero, Result)
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int in_rows_h[5]   = {0, 0, 1, 2, 2};
  float in_vals_h[5] = {0.0, 5.0, 0.0, 1.0, 1.0};
  int verify_h[5]    = {1, 0, 2, 0, 0};

  rmm::device_uvector<int> in_rows(5, stream);
  rmm::device_uvector<int> verify(5, stream);
  rmm::device_uvector<int> results(5, stream);
  rmm::device_uvector<float> in_vals(5, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(verify.data(), 0, verify.size() * sizeof(int), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(results.data(), 0, results.size() * sizeof(int), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(in_vals.data(), 0, in_vals.size() * sizeof(float), stream));

  raft::update_device(in_rows.data(), *&in_rows_h, 5, stream);
  raft::update_device(verify.data(), *&verify_h, 5, stream);
  raft::update_device(in_vals.data(), *&in_vals_h, 5, stream);

  linalg::coo_degree_nz<float>(in_rows.data(), in_vals.data(), 5, results.data(), stream);
  cudaDeviceSynchronize();

  ASSERT_TRUE(raft::devArrMatch<int>(verify.data(), results.data(), 5, raft::Compare<int>()));

  RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}

INSTANTIATE_TEST_CASE_P(SparseDegreeTests, COODegree, ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_CASE_P(SparseDegreeTests, COODegreeNonzero, ::testing::ValuesIn(inputsf));

}  // namespace sparse
}  // namespace raft
