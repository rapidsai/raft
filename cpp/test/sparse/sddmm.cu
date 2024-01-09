/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <iostream>
#include <limits>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/linalg/sddmm.cuh>
#include <raft/util/cudart_utils.hpp>

#include "../test_utils.cuh"

namespace raft {
namespace sparse {

template <typename ValueType, typename IndexType>
struct SDDMMInputs {
  IndexType m;
  IndexType k;
  IndexType n;

  ValueType alpha;
  ValueType beta;

  std::vector<ValueType> a_data;
  std::vector<ValueType> b_data;

  std::vector<IndexType> c_indptr;
  std::vector<IndexType> c_indices;
  std::vector<ValueType> c_data;

  std::vector<ValueType> c_expected_data;
};

template <typename ValueType, typename IndexType>
::std::ostream& operator<<(::std::ostream& os, const SDDMMInputs<ValueType, IndexType>& params)
{
  return os;
}

template <typename ValueType,
          typename IndexType,
          typename LayoutPolicyA = raft::layout_c_contiguous,
          typename LayoutPolicyB = raft::layout_c_contiguous>
class SDDMMTest : public ::testing::TestWithParam<SDDMMInputs<ValueType, IndexType>> {
 public:
  SDDMMTest()
    : params(::testing::TestWithParam<SDDMMInputs<ValueType, IndexType>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      a_data_d(0, resource::get_cuda_stream(handle)),
      b_data_d(0, resource::get_cuda_stream(handle)),
      c_indptr_d(0, resource::get_cuda_stream(handle)),
      c_indices_d(0, resource::get_cuda_stream(handle)),
      c_data_d(0, resource::get_cuda_stream(handle)),
      c_expected_data_d(0, resource::get_cuda_stream(handle))
  {
  }

 protected:
  void make_data()
  {
    std::vector<ValueType> a_data_h = params.a_data;
    std::vector<ValueType> b_data_h = params.b_data;

    std::vector<IndexType> c_indptr_h        = params.c_indptr;
    std::vector<IndexType> c_indices_h       = params.c_indices;
    std::vector<ValueType> c_data_h          = params.c_data;
    std::vector<ValueType> c_expected_data_h = params.c_expected_data;

    a_data_d.resize(a_data_h.size(), stream);
    b_data_d.resize(b_data_h.size(), stream);
    c_indptr_d.resize(c_indptr_h.size(), stream);
    c_indices_d.resize(c_indices_h.size(), stream);
    c_data_d.resize(c_data_h.size(), stream);
    c_expected_data_d.resize(c_expected_data_h.size(), stream);

    update_device(a_data_d.data(), a_data_h.data(), a_data_h.size(), stream);
    update_device(b_data_d.data(), b_data_h.data(), b_data_h.size(), stream);

    update_device(c_indptr_d.data(), c_indptr_h.data(), c_indptr_h.size(), stream);
    update_device(c_indices_d.data(), c_indices_h.data(), c_indices_h.size(), stream);
    update_device(c_data_d.data(), c_data_h.data(), c_data_h.size(), stream);
    update_device(
      c_expected_data_d.data(), c_expected_data_h.data(), c_expected_data_h.size(), stream);

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void SetUp() override { make_data(); }

  void Run()
  {
    // Check params
    ASSERT_EQ(params.a_data.size(), params.m * params.k);
    ASSERT_EQ(params.b_data.size(), params.n * params.k);
    ASSERT_EQ(params.c_data.size(), params.c_indices.size());
    ASSERT_GE(params.c_indices.size(), 0);

    auto a = raft::make_device_matrix_view<const ValueType, IndexType, LayoutPolicyA>(
      a_data_d.data(), params.m, params.k);
    auto b = raft::make_device_matrix_view<const ValueType, IndexType, LayoutPolicyB>(
      b_data_d.data(),
      ((std::is_same_v<LayoutPolicyA, LayoutPolicyB>) ? params.n : params.k),
      ((std::is_same_v<LayoutPolicyA, LayoutPolicyB>) ? params.k : params.n));

    auto c_structure = raft::make_device_compressed_structure_view<IndexType, IndexType, IndexType>(
      c_indptr_d.data(),
      c_indices_d.data(),
      params.m,
      params.n,
      static_cast<IndexType>(c_indices_d.size()));

    auto c = raft::make_device_csr_matrix_view<ValueType>(c_data_d.data(), c_structure);

    RAFT_CUDA_TRY(cudaStreamSynchronize(resource::get_cuda_stream(handle)));

    auto op_a = raft::linalg::Operation::NON_TRANSPOSE;
    auto op_b = !(std::is_same_v<LayoutPolicyA, LayoutPolicyB>)
                  ? raft::linalg::Operation::NON_TRANSPOSE
                  : raft::linalg::Operation::TRANSPOSE;

    raft::sparse::linalg::sddmm(handle,
                                a,
                                b,
                                c,
                                op_a,
                                op_b,
                                raft::make_host_scalar_view<ValueType>(&params.alpha),
                                raft::make_host_scalar_view<ValueType>(&params.beta));

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    ASSERT_TRUE(raft::devArrMatch<ValueType>(c_expected_data_d.data(),
                                             c.get_elements().data(),
                                             params.c_indices.size(),
                                             raft::CompareApprox<ValueType>(1e-6f),
                                             stream));
  }

  raft::resources handle;
  SDDMMInputs<ValueType, IndexType> params;
  cudaStream_t stream;

  rmm::device_uvector<ValueType> a_data_d;
  rmm::device_uvector<ValueType> b_data_d;

  rmm::device_uvector<IndexType> c_indptr_d;
  rmm::device_uvector<IndexType> c_indices_d;
  rmm::device_uvector<ValueType> c_data_d;

  rmm::device_uvector<ValueType> c_expected_data_d;
};

using SDDMMTestF_Row_Col = SDDMMTest<float, int, raft::row_major, raft::col_major>;
TEST_P(SDDMMTestF_Row_Col, Result) { Run(); }

using SDDMMTestF_Col_Row = SDDMMTest<float, int, raft::col_major, raft::row_major>;
TEST_P(SDDMMTestF_Col_Row, Result) { Run(); }

using SDDMMTestF_Row_Row = SDDMMTest<float, int, raft::row_major, raft::row_major>;
TEST_P(SDDMMTestF_Row_Row, Result) { Run(); }

using SDDMMTestF_Col_Col = SDDMMTest<float, int, raft::col_major, raft::col_major>;
TEST_P(SDDMMTestF_Col_Col, Result) { Run(); }

using SDDMMTestD_Row_Col = SDDMMTest<double, int, raft::row_major, raft::col_major>;
TEST_P(SDDMMTestD_Row_Col, Result) { Run(); }

using SDDMMTestD_Col_Row = SDDMMTest<double, int, raft::col_major, raft::row_major>;
TEST_P(SDDMMTestD_Col_Row, Result) { Run(); }

using SDDMMTestD_Row_Row = SDDMMTest<double, int, raft::row_major, raft::row_major>;
TEST_P(SDDMMTestD_Row_Row, Result) { Run(); }

using SDDMMTestD_Col_Col = SDDMMTest<double, int, raft::col_major, raft::col_major>;
TEST_P(SDDMMTestD_Col_Col, Result) { Run(); }

const std::vector<SDDMMInputs<float, int>> sddmm_inputs_row_col_f = {
  {
    4,
    4,
    3,
    1.0,
    0.0,
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
    {1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.0, 80.0, 90.0, 184.0, 246.0, 288.0, 330.0, 334.0, 450.0},
  },
  {
    4,
    4,
    3,
    1.0,
    0.5,
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
    {1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.5, 80.5, 90.5, 184.5, 246.5, 288.5, 330.5, 334.5, 450.5},
  }};
const std::vector<SDDMMInputs<float, int>> sddmm_inputs_col_row_f = {
  {
    4,
    4,
    3,
    1.0,
    0.0,
    {1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0},
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.0, 80.0, 90.0, 184.0, 246.0, 288.0, 330.0, 334.0, 450.0},
  },
  {
    4,
    4,
    3,
    1.0,
    0.5,
    {1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0},
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.5, 80.5, 90.5, 184.5, 246.5, 288.5, 330.5, 334.5, 450.5},
  }};
const std::vector<SDDMMInputs<float, int>> sddmm_inputs_row_row_f = {
  {
    4,
    4,
    3,
    1.0,
    0.0,
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
    {1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.0, 80.0, 90.0, 184.0, 246.0, 288.0, 330.0, 334.0, 450.0},
  },
  {
    4,
    4,
    3,
    1.0,
    0.5,
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
    {1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.5, 80.5, 90.5, 184.5, 246.5, 288.5, 330.5, 334.5, 450.5},
  }};
const std::vector<SDDMMInputs<float, int>> sddmm_inputs_col_col_f = {
  {
    4,
    4,
    3,
    1.0,
    0.0,
    {1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0},
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.0, 80.0, 90.0, 184.0, 246.0, 288.0, 330.0, 334.0, 450.0},
  },
  {
    4,
    4,
    3,
    1.0,
    0.5,
    {1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0},
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.5, 80.5, 90.5, 184.5, 246.5, 288.5, 330.5, 334.5, 450.5},
  }};

const std::vector<SDDMMInputs<double, int>> sddmm_inputs_row_col_d = {
  {
    4,
    4,
    3,
    1.0,
    0.0,
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
    {1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.0, 80.0, 90.0, 184.0, 246.0, 288.0, 330.0, 334.0, 450.0},
  },
  {
    4,
    4,
    3,
    1.0,
    0.5,
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
    {1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.5, 80.5, 90.5, 184.5, 246.5, 288.5, 330.5, 334.5, 450.5},
  }};

const std::vector<SDDMMInputs<double, int>> sddmm_inputs_col_row_d = {
  {
    4,
    4,
    3,
    1.0,
    0.0,
    {1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0},
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.0, 80.0, 90.0, 184.0, 246.0, 288.0, 330.0, 334.0, 450.0},
  },
  {
    4,
    4,
    3,
    1.0,
    0.5,
    {1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0},
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.5, 80.5, 90.5, 184.5, 246.5, 288.5, 330.5, 334.5, 450.5},
  }};
const std::vector<SDDMMInputs<double, int>> sddmm_inputs_row_row_d = {
  {
    4,
    4,
    3,
    1.0,
    0.0,
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
    {1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.0, 80.0, 90.0, 184.0, 246.0, 288.0, 330.0, 334.0, 450.0},
  },
  {
    4,
    4,
    3,
    1.0,
    0.5,
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
    {1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.5, 80.5, 90.5, 184.5, 246.5, 288.5, 330.5, 334.5, 450.5},
  }};
const std::vector<SDDMMInputs<double, int>> sddmm_inputs_col_col_d = {
  {
    4,
    4,
    3,
    1.0,
    0.0,
    {1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0},
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.0, 80.0, 90.0, 184.0, 246.0, 288.0, 330.0, 334.0, 450.0},
  },
  {
    4,
    4,
    3,
    1.0,
    0.5,
    {1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0},
    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
    {0, 3, 4, 7, 9},
    {0, 1, 2, 1, 0, 1, 2, 0, 2},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    {70.5, 80.5, 90.5, 184.5, 246.5, 288.5, 330.5, 334.5, 450.5},
  }};

INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestF_Row_Col, ::testing::ValuesIn(sddmm_inputs_row_col_f));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestF_Col_Row, ::testing::ValuesIn(sddmm_inputs_col_row_f));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestF_Row_Row, ::testing::ValuesIn(sddmm_inputs_row_row_f));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestF_Col_Col, ::testing::ValuesIn(sddmm_inputs_col_col_f));

INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestD_Row_Col, ::testing::ValuesIn(sddmm_inputs_row_col_d));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestD_Col_Row, ::testing::ValuesIn(sddmm_inputs_col_row_d));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestD_Row_Row, ::testing::ValuesIn(sddmm_inputs_row_row_d));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestD_Col_Col, ::testing::ValuesIn(sddmm_inputs_col_col_d));

}  // namespace sparse
}  // namespace raft
