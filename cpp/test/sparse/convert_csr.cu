/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_utils.cuh>

#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>

#include <iostream>

namespace raft {
namespace sparse {

/**************************** sorted COO to CSR ****************************/

template <typename T>
struct SparseConvertCSRInputs {
  int m, n, nnz;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SparseConvertCSRInputs<T>& dims)
{
  return os;
}

template <typename T>
class SparseConvertCSRTest : public ::testing::TestWithParam<SparseConvertCSRInputs<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
  SparseConvertCSRInputs<T> params;
};

const std::vector<SparseConvertCSRInputs<float>> inputsf = {{5, 10, 5, 1234ULL}};

typedef SparseConvertCSRTest<float> SortedCOOToCSR;
TEST_P(SortedCOOToCSR, Result)
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int nnz = 8;

  int* in_h  = new int[nnz]{0, 0, 1, 1, 2, 2, 3, 3};
  int* exp_h = new int[4]{0, 2, 4, 6};

  rmm::device_uvector<int> in(nnz, stream);
  rmm::device_uvector<int> exp(4, stream);
  rmm::device_uvector<int> out(4, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(in.data(), 0, in.size() * sizeof(int), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(exp.data(), 0, exp.size() * sizeof(int), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(out.data(), 0, out.size() * sizeof(int), stream));

  raft::update_device(in.data(), in_h, nnz, stream);
  raft::update_device(exp.data(), exp_h, 4, stream);

  convert::sorted_coo_to_csr<int>(in.data(), nnz, out.data(), 4, stream);

  ASSERT_TRUE(raft::devArrMatch<int>(out.data(), exp.data(), 4, raft::Compare<int>(), stream));

  cudaStreamDestroy(stream);

  delete[] in_h;
  delete[] exp_h;
}

INSTANTIATE_TEST_CASE_P(SparseConvertCSRTest, SortedCOOToCSR, ::testing::ValuesIn(inputsf));

/******************************** adj graph ********************************/

template <typename index_t>
RAFT_KERNEL init_adj_kernel(bool* adj, index_t num_rows, index_t num_cols, index_t divisor)
{
  index_t r = blockDim.y * blockIdx.y + threadIdx.y;
  index_t c = blockDim.x * blockIdx.x + threadIdx.x;

  for (; r < num_rows; r += gridDim.y * blockDim.y) {
    for (; c < num_cols; c += gridDim.x * blockDim.x) {
      adj[r * num_cols + c] = c % divisor == 0;
    }
  }
}

template <typename index_t>
void init_adj(bool* adj, index_t num_rows, index_t num_cols, index_t divisor, cudaStream_t stream)
{
  // adj matrix: element a_ij is set to one if j is divisible by divisor.
  dim3 block(32, 32);
  const index_t max_y_grid_dim = 65535;
  dim3 grid(num_cols / 32 + 1, (int)min(num_rows / 32 + 1, max_y_grid_dim));
  init_adj_kernel<index_t><<<grid, block, 0, stream>>>(adj, num_rows, num_cols, divisor);
  RAFT_CHECK_CUDA(stream);
}

template <typename index_t>
struct CSRAdjGraphInputs {
  index_t n_rows;
  index_t n_cols;
  index_t divisor;
};

template <typename index_t>
class CSRAdjGraphTest : public ::testing::TestWithParam<CSRAdjGraphInputs<index_t>> {
 public:
  CSRAdjGraphTest()
    : stream(resource::get_cuda_stream(handle)),
      params(::testing::TestWithParam<CSRAdjGraphInputs<index_t>>::GetParam()),
      adj(params.n_rows * params.n_cols, stream),
      row_ind(params.n_rows, stream),
      row_counters(params.n_rows, stream),
      col_ind(params.n_rows * params.n_cols, stream),
      row_ind_host(params.n_rows)
  {
  }

 protected:
  void SetUp() override
  {
    // Initialize adj matrix: element a_ij equals one if j is divisible by
    // params.divisor.
    init_adj(adj.data(), params.n_rows, params.n_cols, params.divisor, stream);
    // Initialize row_ind
    for (size_t i = 0; i < row_ind_host.size(); ++i) {
      size_t nnz_per_row = raft::ceildiv(params.n_cols, params.divisor);
      row_ind_host[i]    = nnz_per_row * i;
    }
    raft::update_device(row_ind.data(), row_ind_host.data(), row_ind.size(), stream);

    // Initialize result to 1, so we can catch any errors.
    RAFT_CUDA_TRY(cudaMemsetAsync(col_ind.data(), 1, col_ind.size() * sizeof(index_t), stream));
  }

  void Run()
  {
    convert::adj_to_csr<index_t>(handle,
                                 adj.data(),
                                 row_ind.data(),
                                 params.n_rows,
                                 params.n_cols,
                                 row_counters.data(),
                                 col_ind.data());

    std::vector<index_t> col_ind_host(col_ind.size());
    raft::update_host(col_ind_host.data(), col_ind.data(), col_ind.size(), stream);
    std::vector<index_t> row_counters_host(params.n_rows);
    raft::update_host(row_counters_host.data(), row_counters.data(), row_counters.size(), stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    // 1. Check that each row contains enough values
    index_t nnz_per_row = raft::ceildiv(params.n_cols, params.divisor);
    for (index_t i = 0; i < params.n_rows; ++i) {
      ASSERT_EQ(row_counters_host[i], nnz_per_row) << "where i = " << i;
    }
    // 2. Check that all column indices are divisble by divisor
    for (index_t i = 0; i < params.n_rows; ++i) {
      index_t row_base = row_ind_host[i];
      for (index_t j = 0; j < nnz_per_row; ++j) {
        ASSERT_EQ(0, col_ind_host[row_base + j] % params.divisor);
      }
    }
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  CSRAdjGraphInputs<index_t> params;
  rmm::device_uvector<bool> adj;
  rmm::device_uvector<index_t> row_ind;
  rmm::device_uvector<index_t> row_counters;
  rmm::device_uvector<index_t> col_ind;
  std::vector<index_t> row_ind_host;
};

using CSRAdjGraphTestI = CSRAdjGraphTest<int>;
TEST_P(CSRAdjGraphTestI, Result) { Run(); }

using CSRAdjGraphTestL = CSRAdjGraphTest<int64_t>;
TEST_P(CSRAdjGraphTestL, Result) { Run(); }

const std::vector<CSRAdjGraphInputs<int>> csradjgraph_inputs_i     = {{10, 10, 2}};
const std::vector<CSRAdjGraphInputs<int64_t>> csradjgraph_inputs_l = {
  {0, 0, 2},
  {10, 10, 2},
  {64 * 1024 + 10, 2, 3},  // 64K + 10 is slightly over maximum of blockDim.y
  {16, 16, 3},             // No peeling-remainder
  {17, 16, 3},             // Check peeling-remainder
  {18, 16, 3},             // Check peeling-remainder
  {32 + 9, 33, 2},         // Check peeling-remainder
};

INSTANTIATE_TEST_CASE_P(SparseConvertCSRTest,
                        CSRAdjGraphTestI,
                        ::testing::ValuesIn(csradjgraph_inputs_i));
INSTANTIATE_TEST_CASE_P(SparseConvertCSRTest,
                        CSRAdjGraphTestL,
                        ::testing::ValuesIn(csradjgraph_inputs_l));

}  // namespace sparse
}  // namespace raft
