/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cuda_utils.cuh>

#include <gtest/gtest.h>

#include <iostream>

namespace raft {
namespace sparse {

template <typename index_t>
struct SegmentedCopyInputs {
  index_t n_rows;
  index_t n_cols;
  index_t top_k;
  float sparsity;
};

template <typename value_t, typename index_t>
class SegmentedCopyTest : public ::testing::TestWithParam<SegmentedCopyInputs<index_t>> {
 public:
  SegmentedCopyTest()
    : stream(resource::get_cuda_stream(handle)),
      params(::testing::TestWithParam<SegmentedCopyInputs<index_t>>::GetParam()),
      indices_d(0, stream),
      indptr_d(0, stream),
      values_d(0, stream),
      dst_values_d(0, stream),
      dst_values_expected_d(0, stream),
      dst_indices_d(0, stream),
      dst_indices_expected_d(0, stream)
  {
  }

 protected:
  index_t create_sparse_matrix(index_t m, index_t n, value_t sparsity, std::vector<bool>& matrix)
  {
    index_t total_elements = static_cast<index_t>(m * n);
    index_t num_ones       = static_cast<index_t>((total_elements * 1.0f) * sparsity);
    index_t res            = num_ones;

    for (index_t i = 0; i < total_elements; ++i) {
      matrix[i] = false;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_idx(0, total_elements - 1);

    while (num_ones > 0) {
      size_t index = dis_idx(gen);
      if (matrix[index] == false) {
        matrix[index] = true;
        num_ones--;
      }
    }
    return res;
  }
  void convert_to_csr(std::vector<bool>& matrix,
                      index_t rows,
                      index_t cols,
                      std::vector<index_t>& indices,
                      std::vector<index_t>& indptr)
  {
    index_t offset_indptr   = 0;
    index_t offset_values   = 0;
    indptr[offset_indptr++] = 0;

    for (index_t i = 0; i < rows; ++i) {
      for (index_t j = 0; j < cols; ++j) {
        if (matrix[i * cols + j]) {
          indices[offset_values] = static_cast<index_t>(j);
          offset_values++;
        }
      }
      indptr[offset_indptr++] = static_cast<index_t>(offset_values);
    }
  }

  template <typename dst_t>
  void cpu_segmented_copy(index_t rows,
                          index_t max_len_per_row,
                          const std::vector<dst_t>& src,
                          const std::vector<index_t>& offsets,
                          std::vector<dst_t>& dst)
  {
    for (index_t row = 0; row < rows; ++row) {
      index_t start  = offsets[row];
      index_t end    = offsets[row + 1];  //(row < rows - 1) ? offsets[row + 1] : src.size();
      index_t length = std::min(end - start, max_len_per_row);
      if (length == 0) continue;
      std::copy(
        src.begin() + start, src.begin() + start + length, dst.begin() + row * max_len_per_row);
    }
  }

  void SetUp() override
  {
    std::vector<bool> dense_values_h(params.n_rows * params.n_cols);
    nnz = create_sparse_matrix(params.n_rows, params.n_cols, params.sparsity, dense_values_h);

    std::vector<value_t> values_h(nnz);
    std::vector<index_t> indices_h(nnz);
    std::vector<index_t> indptr_h(params.n_rows + 1);
    std::vector<value_t> dst_values_h(params.n_rows * params.top_k, static_cast<value_t>(2.0f));

    std::vector<index_t> dst_indices_h(params.n_rows * params.top_k,
                                       static_cast<index_t>(params.n_rows * params.n_cols + 1));

    // sync up the initial values in advance to 2.0 which is out of random range [-1.0, 1.0].
    dst_values_d.resize(params.n_rows * params.top_k, stream);
    dst_indices_d.resize(params.n_rows * params.top_k, stream);

    update_device(dst_values_d.data(), dst_values_h.data(), dst_values_h.size(), stream);
    update_device(dst_indices_d.data(), dst_indices_h.data(), dst_indices_h.size(), stream);
    resource::sync_stream(handle);

    auto blobs_values = raft::make_device_matrix<value_t, index_t>(handle, 1, dst_values_h.size());
    auto labels       = raft::make_device_vector<index_t, index_t>(handle, 1);

    raft::random::make_blobs<value_t, index_t>(blobs_values.data_handle(),
                                               labels.data_handle(),
                                               1,
                                               dst_values_h.size(),
                                               1,
                                               stream,
                                               false,
                                               nullptr,
                                               nullptr,
                                               value_t(1.0),
                                               false,
                                               value_t(-1.0f),
                                               value_t(1.0f),
                                               uint64_t(2024));
    raft::copy(dst_values_h.data(), blobs_values.data_handle(), dst_values_h.size(), stream);
    raft::copy(dst_values_d.data(), blobs_values.data_handle(), dst_values_h.size(), stream);
    resource::sync_stream(handle);

    convert_to_csr(dense_values_h, params.n_rows, params.n_cols, indices_h, indptr_h);

    cpu_segmented_copy<value_t>(params.n_rows, params.top_k, values_h, indptr_h, dst_values_h);
    cpu_segmented_copy<index_t>(params.n_rows, params.top_k, indices_h, indptr_h, dst_indices_h);

    values_d.resize(nnz, stream);
    indices_d.resize(nnz, stream);
    indptr_d.resize(params.n_rows + 1, stream);
    dst_values_expected_d.resize(params.n_rows * params.top_k, stream);
    dst_indices_expected_d.resize(params.n_rows * params.top_k, stream);

    update_device(values_d.data(), values_h.data(), values_h.size(), stream);
    update_device(indices_d.data(), indices_h.data(), indices_h.size(), stream);
    update_device(indptr_d.data(), indptr_h.data(), indptr_h.size(), stream);
    update_device(dst_values_expected_d.data(), dst_values_h.data(), dst_values_h.size(), stream);
    update_device(
      dst_indices_expected_d.data(), dst_indices_h.data(), dst_indices_h.size(), stream);

    resource::sync_stream(handle);
  }

  void Run()
  {
    auto src_values  = raft::make_device_vector_view<value_t, index_t>(values_d.data(), nnz);
    auto src_indices = raft::make_device_vector_view<index_t, index_t>(indices_d.data(), nnz);
    auto offsets =
      raft::make_device_vector_view<index_t, index_t>(indptr_d.data(), params.n_rows + 1);
    auto dst_values = raft::make_device_matrix_view<value_t, index_t, raft::row_major>(
      dst_values_d.data(), params.n_rows, params.top_k);
    auto dst_indices = raft::make_device_matrix_view<index_t, index_t, raft::row_major>(
      dst_indices_d.data(), params.n_rows, params.top_k);

    raft::matrix::segmented_copy(handle, params.top_k, src_values, offsets, dst_values);
    raft::matrix::segmented_copy(handle, params.top_k, src_indices, offsets, dst_indices);

    resource::sync_stream(handle);

    ASSERT_TRUE(raft::devArrMatch<value_t>(dst_values_expected_d.data(),
                                           dst_values_d.data(),
                                           params.n_rows * params.top_k,
                                           raft::CompareApprox<value_t>(1e-6f),
                                           stream));

    ASSERT_TRUE(raft::devArrMatch<index_t>(dst_indices_expected_d.data(),
                                           dst_indices_d.data(),
                                           params.n_rows * params.top_k,
                                           raft::Compare<index_t>(),
                                           stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SegmentedCopyInputs<index_t> params;

  index_t nnz;

  rmm::device_uvector<value_t> values_d;
  rmm::device_uvector<index_t> indptr_d;
  rmm::device_uvector<index_t> indices_d;

  rmm::device_uvector<value_t> dst_values_d;
  rmm::device_uvector<value_t> dst_values_expected_d;

  rmm::device_uvector<index_t> dst_indices_d;
  rmm::device_uvector<index_t> dst_indices_expected_d;
};

using SegmentedCopyTest_float_int = SegmentedCopyTest<float, int>;
TEST_P(SegmentedCopyTest_float_int, Result) { Run(); }

using SegmentedCopyTest_double_int64 = SegmentedCopyTest<double, int64_t>;
TEST_P(SegmentedCopyTest_double_int64, Result) { Run(); }

template <typename index_t>
const std::vector<SegmentedCopyInputs<index_t>> segmentedcopy_inputs = {
  {10, 32, 10, 0.0},
  {10, 32, 10, 0.3},
  {32, 1024, 63, 0.3},
  {1024, 1024, 128, 0.2},
  {1024, 1024 * 2000, 251, 0.2},
  {2048, 1024 * 100, 1000, 0.3},
  {2048, 1024 * 100, 2100, 0.5}};

INSTANTIATE_TEST_CASE_P(SegmentedCopyTest,
                        SegmentedCopyTest_float_int,
                        ::testing::ValuesIn(segmentedcopy_inputs<int>));
INSTANTIATE_TEST_CASE_P(SegmentedCopyTest,
                        SegmentedCopyTest_double_int64,
                        ::testing::ValuesIn(segmentedcopy_inputs<int64_t>));

}  // namespace sparse
}  // namespace raft
