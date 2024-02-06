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

#include <gtest/gtest.h>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include "../test_utils.cuh"

#include <iostream>

namespace raft {
namespace sparse {

template <typename value_idx, typename value_t>
RAFT_KERNEL assert_symmetry(
  value_idx* rows, value_idx* cols, value_t* vals, value_idx nnz, value_idx* sum)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= nnz) return;

  atomicAdd(sum, rows[tid]);
  atomicAdd(sum, -1 * cols[tid]);
}

template <typename value_idx, typename value_t>
struct SparseSymmetrizeInputs {
  value_idx n_cols;

  std::vector<value_idx> indptr_h;
  std::vector<value_idx> indices_h;
  std::vector<value_t> data_h;
};

template <typename value_idx, typename value_t>
::std::ostream& operator<<(::std::ostream& os,
                           const SparseSymmetrizeInputs<value_idx, value_t>& dims)
{
  return os;
}

template <typename value_idx, typename value_t>
class SparseSymmetrizeTest
  : public ::testing::TestWithParam<SparseSymmetrizeInputs<value_idx, value_t>> {
 public:
  SparseSymmetrizeTest()
    : params(::testing::TestWithParam<SparseSymmetrizeInputs<value_idx, value_t>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      indptr(0, stream),
      indices(0, stream),
      data(0, stream)
  {
  }

 protected:
  void make_data()
  {
    std::vector<value_idx> indptr_h  = params.indptr_h;
    std::vector<value_idx> indices_h = params.indices_h;
    std::vector<value_t> data_h      = params.data_h;

    indptr.resize(indptr_h.size(), stream);
    indices.resize(indices_h.size(), stream);
    data.resize(data_h.size(), stream);

    update_device(indptr.data(), indptr_h.data(), indptr_h.size(), stream);
    update_device(indices.data(), indices_h.data(), indices_h.size(), stream);
    update_device(data.data(), data_h.data(), data_h.size(), stream);
  }

  void SetUp() override
  {
    make_data();

    value_idx m   = params.indptr_h.size() - 1;
    value_idx n   = params.n_cols;
    value_idx nnz = params.indices_h.size();

    rmm::device_uvector<value_idx> coo_rows(nnz, stream);

    raft::sparse::convert::csr_to_coo(indptr.data(), m, coo_rows.data(), nnz, stream);

    raft::sparse::COO<value_t, value_idx> out(stream);

    raft::sparse::linalg::symmetrize(
      handle, coo_rows.data(), indices.data(), data.data(), m, n, coo_rows.size(), out);

    rmm::device_scalar<value_idx> sum(stream);
    sum.set_value_to_zero_async(stream);

    assert_symmetry<<<raft::ceildiv(out.nnz, 256), 256, 0, stream>>>(
      out.rows(), out.cols(), out.vals(), out.nnz, sum.data());

    sum_h = sum.value(stream);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  // input data
  rmm::device_uvector<value_idx> indptr, indices;
  rmm::device_uvector<value_t> data;

  value_idx sum_h;

  SparseSymmetrizeInputs<value_idx, value_t> params;
};

template <typename T>
struct COOSymmetrizeInputs {
  int m, n, nnz;
  unsigned long long int seed;
};

template <typename T>
class COOSymmetrizeTest : public ::testing::TestWithParam<COOSymmetrizeInputs<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

const std::vector<COOSymmetrizeInputs<float>> inputsf = {{5, 10, 5, 1234ULL}};

typedef COOSymmetrizeTest<float> COOSymmetrize;
TEST_P(COOSymmetrize, Result)
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int nnz = 8;

  int* in_rows_h   = new int[nnz]{0, 0, 1, 1, 2, 2, 3, 3};
  int* in_cols_h   = new int[nnz]{1, 3, 2, 3, 0, 1, 0, 2};
  float* in_vals_h = new float[nnz]{0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5};

  int* exp_rows_h = new int[nnz * 2]{1, 0, 0, 0, 1, 3, 1, 0, 0, 2, 2, 0, 3, 2, 3, 0};
  int* exp_cols_h = new int[nnz * 2]{0, 1, 3, 0, 2, 1, 3, 0, 2, 0, 1, 0, 0, 3, 2, 0};
  float* exp_vals_h =
    new float[nnz * 2]{0.5, 0.5, 1.5, 0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0, 1.5, 0.5, 0.5, 0.0};

  COO<float> in(stream, nnz, 4, 4);
  raft::update_device(in.rows(), *&in_rows_h, nnz, stream);
  raft::update_device(in.cols(), *&in_cols_h, nnz, stream);
  raft::update_device(in.vals(), *&in_vals_h, nnz, stream);

  COO<float> out(stream);

  linalg::coo_symmetrize<float>(
    &in,
    &out,
    [] __device__(int row, int col, float val, float trans) { return val + trans; },
    stream);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  std::cout << out << std::endl;

  ASSERT_TRUE(out.nnz == nnz * 2);
  ASSERT_TRUE(raft::devArrMatch<int>(out.rows(), exp_rows_h, out.nnz, raft::Compare<int>()));
  ASSERT_TRUE(raft::devArrMatch<int>(out.cols(), exp_cols_h, out.nnz, raft::Compare<int>()));
  ASSERT_TRUE(raft::devArrMatch<float>(out.vals(), exp_vals_h, out.nnz, raft::Compare<float>()));

  cudaStreamDestroy(stream);

  delete[] in_rows_h;
  delete[] in_cols_h;
  delete[] in_vals_h;

  delete[] exp_rows_h;
  delete[] exp_cols_h;
  delete[] exp_vals_h;
}

INSTANTIATE_TEST_CASE_P(COOSymmetrizeTest, COOSymmetrize, ::testing::ValuesIn(inputsf));

const std::vector<SparseSymmetrizeInputs<int, float>> symm_inputs_fint = {
  // Test n_clusters == n_points
  {
    2,
    {0, 2, 4, 6, 8},
    {0, 1, 0, 1, 0, 1, 0, 1},
    {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f},
  },
  {2,
   {0, 2, 4, 6, 8},
   {0, 1, 0, 1, 0, 1, 0, 1},  // indices
   {1.0f, 3.0f, 1.0f, 5.0f, 50.0f, 28.0f, 16.0f, 2.0f}},

};

typedef SparseSymmetrizeTest<int, float> SparseSymmetrizeTestF_int;
TEST_P(SparseSymmetrizeTestF_int, Result) { ASSERT_TRUE(sum_h == 0); }

INSTANTIATE_TEST_CASE_P(SparseSymmetrizeTest,
                        SparseSymmetrizeTestF_int,
                        ::testing::ValuesIn(symm_inputs_fint));

}  // namespace sparse
}  // namespace raft
