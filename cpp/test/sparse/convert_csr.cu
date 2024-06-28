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

#include <raft/core/bitmap.cuh>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/util/cuda_utils.cuh>

#include <gtest/gtest.h>

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

/******************************** bitmap to csr ********************************/

template <typename index_t>
struct BitmapToCSRInputs {
  index_t n_rows;
  index_t n_cols;
  float sparsity;
  bool owning;
};

template <typename bitmap_t, typename index_t, typename value_t>
class BitmapToCSRTest : public ::testing::TestWithParam<BitmapToCSRInputs<index_t>> {
 public:
  BitmapToCSRTest()
    : stream(resource::get_cuda_stream(handle)),
      params(::testing::TestWithParam<BitmapToCSRInputs<index_t>>::GetParam()),
      bitmap_d(0, stream),
      indices_d(0, stream),
      indptr_d(0, stream),
      values_d(0, stream),
      indptr_expected_d(0, stream),
      indices_expected_d(0, stream),
      values_expected_d(0, stream)
  {
  }

 protected:
  index_t create_sparse_matrix(index_t m, index_t n, float sparsity, std::vector<bitmap_t>& bitmap)
  {
    index_t total    = static_cast<index_t>(m * n);
    index_t num_ones = static_cast<index_t>((total * 1.0f) * sparsity);
    index_t res      = num_ones;

    for (auto& item : bitmap) {
      item = static_cast<bitmap_t>(0);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<index_t> dis(0, total - 1);

    while (num_ones > 0) {
      index_t index = dis(gen);

      bitmap_t& element    = bitmap[index / (8 * sizeof(bitmap_t))];
      index_t bit_position = index % (8 * sizeof(bitmap_t));

      if (((element >> bit_position) & 1) == 0) {
        element |= (static_cast<index_t>(1) << bit_position);
        num_ones--;
      }
    }
    return res;
  }

  void cpu_convert_to_csr(std::vector<bitmap_t>& bitmap,
                          index_t rows,
                          index_t cols,
                          std::vector<index_t>& indices,
                          std::vector<index_t>& indptr)
  {
    index_t offset_indptr   = 0;
    index_t offset_values   = 0;
    indptr[offset_indptr++] = 0;

    index_t index        = 0;
    bitmap_t element     = 0;
    index_t bit_position = 0;

    for (index_t i = 0; i < rows; ++i) {
      for (index_t j = 0; j < cols; ++j) {
        index        = i * cols + j;
        element      = bitmap[index / (8 * sizeof(bitmap_t))];
        bit_position = index % (8 * sizeof(bitmap_t));

        if (((element >> bit_position) & 1)) {
          indices[offset_values] = static_cast<index_t>(j);
          offset_values++;
        }
      }
      indptr[offset_indptr++] = static_cast<index_t>(offset_values);
    }
  }

  bool csr_compare(const std::vector<index_t>& row_ptrs1,
                   const std::vector<index_t>& col_indices1,
                   const std::vector<index_t>& row_ptrs2,
                   const std::vector<index_t>& col_indices2)
  {
    if (row_ptrs1.size() != row_ptrs2.size()) { return false; }

    if (col_indices1.size() != col_indices2.size()) { return false; }

    if (!std::equal(row_ptrs1.begin(), row_ptrs1.end(), row_ptrs2.begin())) { return false; }

    for (size_t i = 0; i < row_ptrs1.size() - 1; ++i) {
      size_t start_idx = row_ptrs1[i];
      size_t end_idx   = row_ptrs1[i + 1];

      std::vector<int> cols1(col_indices1.begin() + start_idx, col_indices1.begin() + end_idx);
      std::vector<int> cols2(col_indices2.begin() + start_idx, col_indices2.begin() + end_idx);

      std::sort(cols1.begin(), cols1.end());
      std::sort(cols2.begin(), cols2.end());

      if (cols1 != cols2) { return false; }
    }

    return true;
  }

  void SetUp() override
  {
    index_t element = raft::ceildiv(params.n_rows * params.n_cols, index_t(sizeof(bitmap_t) * 8));
    std::vector<bitmap_t> bitmap_h(element);
    nnz = create_sparse_matrix(params.n_rows, params.n_cols, params.sparsity, bitmap_h);

    std::vector<index_t> indices_h(nnz);
    std::vector<index_t> indptr_h(params.n_rows + 1);

    cpu_convert_to_csr(bitmap_h, params.n_rows, params.n_cols, indices_h, indptr_h);

    bitmap_d.resize(bitmap_h.size(), stream);
    indptr_d.resize(params.n_rows + 1, stream);
    indices_d.resize(nnz, stream);

    indptr_expected_d.resize(params.n_rows + 1, stream);
    indices_expected_d.resize(nnz, stream);
    values_expected_d.resize(nnz, stream);

    thrust::fill_n(resource::get_thrust_policy(handle), values_expected_d.data(), nnz, value_t{1});

    values_d.resize(nnz, stream);

    update_device(indices_expected_d.data(), indices_h.data(), indices_h.size(), stream);
    update_device(indptr_expected_d.data(), indptr_h.data(), indptr_h.size(), stream);
    update_device(bitmap_d.data(), bitmap_h.data(), bitmap_h.size(), stream);

    resource::sync_stream(handle);
  }

  void Run()
  {
    auto bitmap =
      raft::core::bitmap_view<bitmap_t, index_t>(bitmap_d.data(), params.n_rows, params.n_cols);

    if (params.owning) {
      auto csr =
        raft::make_device_csr_matrix<value_t, index_t>(handle, params.n_rows, params.n_cols, nnz);
      auto csr_view = csr.structure_view();

      convert::bitmap_to_csr(handle, bitmap, csr);
      raft::copy(indptr_d.data(), csr_view.get_indptr().data(), indptr_d.size(), stream);
      raft::copy(indices_d.data(), csr_view.get_indices().data(), indices_d.size(), stream);
      raft::copy(values_d.data(), csr.get_elements().data(), nnz, stream);
    } else {
      auto csr_view = raft::make_device_compressed_structure_view<index_t, index_t, index_t>(
        indptr_d.data(), indices_d.data(), params.n_rows, params.n_cols, nnz);
      auto csr = raft::make_device_csr_matrix<value_t, index_t>(handle, csr_view);

      convert::bitmap_to_csr(handle, bitmap, csr);
      raft::copy(values_d.data(), csr.get_elements().data(), nnz, stream);
    }
    resource::sync_stream(handle);

    std::vector<index_t> indices_h(indices_expected_d.size(), 0);
    std::vector<index_t> indices_expected_h(indices_expected_d.size(), 0);
    update_host(indices_h.data(), indices_d.data(), indices_h.size(), stream);
    update_host(indices_expected_h.data(), indices_expected_d.data(), indices_h.size(), stream);

    std::vector<index_t> indptr_h(indptr_expected_d.size(), 0);
    std::vector<index_t> indptr_expected_h(indptr_expected_d.size(), 0);
    update_host(indptr_h.data(), indptr_d.data(), indptr_h.size(), stream);
    update_host(indptr_expected_h.data(), indptr_expected_d.data(), indptr_h.size(), stream);

    resource::sync_stream(handle);

    ASSERT_TRUE(csr_compare(indptr_h, indices_h, indptr_expected_h, indices_expected_h));
    ASSERT_TRUE(raft::devArrMatch<value_t>(
      values_expected_d.data(), values_d.data(), nnz, raft::Compare<value_t>(), stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  BitmapToCSRInputs<index_t> params;

  rmm::device_uvector<bitmap_t> bitmap_d;

  index_t nnz;

  rmm::device_uvector<index_t> indptr_d;
  rmm::device_uvector<index_t> indices_d;
  rmm::device_uvector<float> values_d;

  rmm::device_uvector<index_t> indptr_expected_d;
  rmm::device_uvector<index_t> indices_expected_d;
  rmm::device_uvector<float> values_expected_d;
};

using BitmapToCSRTestI = BitmapToCSRTest<uint32_t, int, float>;
TEST_P(BitmapToCSRTestI, Result) { Run(); }

using BitmapToCSRTestL = BitmapToCSRTest<uint32_t, int64_t, float>;
TEST_P(BitmapToCSRTestL, Result) { Run(); }

template <typename index_t>
const std::vector<BitmapToCSRInputs<index_t>> bitmaptocsr_inputs = {
  {0, 0, 0.2, false},
  {10, 32, 0.4, false},
  {10, 3, 0.2, false},
  {32, 1024, 0.4, false},
  {1024, 1048576, 0.01, false},
  {1024, 1024, 0.4, false},
  {64 * 1024 + 10, 2, 0.3, false},  // 64K + 10 is slightly over maximum of blockDim.y
  {16, 16, 0.3, false},             // No peeling-remainder
  {17, 16, 0.3, false},             // Check peeling-remainder
  {18, 16, 0.3, false},             // Check peeling-remainder
  {32 + 9, 33, 0.2, false},         // Check peeling-remainder
  {2, 33, 0.2, false},              // Check peeling-remainder
  {0, 0, 0.2, true},
  {10, 32, 0.4, true},
  {10, 3, 0.2, true},
  {32, 1024, 0.4, true},
  {1024, 1048576, 0.01, true},
  {1024, 1024, 0.4, true},
  {64 * 1024 + 10, 2, 0.3, true},  // 64K + 10 is slightly over maximum of blockDim.y
  {16, 16, 0.3, true},             // No peeling-remainder
  {17, 16, 0.3, true},             // Check peeling-remainder
  {18, 16, 0.3, true},             // Check peeling-remainder
  {32 + 9, 33, 0.2, true},         // Check peeling-remainder
  {2, 33, 0.2, true},              // Check peeling-remainder
};

INSTANTIATE_TEST_CASE_P(SparseConvertCSRTest,
                        BitmapToCSRTestI,
                        ::testing::ValuesIn(bitmaptocsr_inputs<int>));
INSTANTIATE_TEST_CASE_P(SparseConvertCSRTest,
                        BitmapToCSRTestL,
                        ::testing::ValuesIn(bitmaptocsr_inputs<int64_t>));

}  // namespace sparse
}  // namespace raft
