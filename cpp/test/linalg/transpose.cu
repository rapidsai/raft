/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename T>
struct TranposeInputs {
  T tolerance;
  int len;
  int n_row;
  int n_col;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const TranposeInputs<T>& dims)
{
  return os;
}

template <typename T>
class TransposeTest : public ::testing::TestWithParam<TranposeInputs<T>> {
 public:
  TransposeTest()
    : params(::testing::TestWithParam<TranposeInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.len, stream),
      data_trans_ref(params.len, stream),
      data_trans(params.len, stream)
  {
  }

 protected:
  void SetUp() override
  {
    int len = params.len;
    ASSERT(params.len == 9, "This test works only with len=9!");
    T data_h[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    raft::update_device(data.data(), data_h, len, stream);
    T data_ref_h[] = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0};
    raft::update_device(data_trans_ref.data(), data_ref_h, len, stream);

    transpose(handle, data.data(), data_trans.data(), params.n_row, params.n_col, stream);
    transpose(data.data(), params.n_row, stream);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  TranposeInputs<T> params;
  rmm::device_uvector<T> data, data_trans, data_trans_ref;
};

const std::vector<TranposeInputs<float>> inputsf2 = {{0.1f, 3 * 3, 3, 3, 1234ULL}};

const std::vector<TranposeInputs<double>> inputsd2 = {{0.1, 3 * 3, 3, 3, 1234ULL}};

typedef TransposeTest<float> TransposeTestValF;
TEST_P(TransposeTestValF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                data_trans.data(),
                                params.len,
                                raft::CompareApproxAbs<float>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                data.data(),
                                params.len,
                                raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef TransposeTest<double> TransposeTestValD;
TEST_P(TransposeTestValD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                data_trans.data(),
                                params.len,
                                raft::CompareApproxAbs<double>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                data.data(),
                                params.len,
                                raft::CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTestValD, ::testing::ValuesIn(inputsd2));

namespace {
/**
 * We hide these functions in tests for now until we have a heterogeneous mdarray
 * implementation.
 */

/**
 * @brief Transpose a matrix. The output has same layout policy as the input.
 *
 * @tparam T Data type of input matrix elements.
 * @tparam LayoutPolicy Layout type of the input matrix. When layout is strided, it can
 *                      be a submatrix of a larger matrix. Arbitrary stride is not supported.
 *
 * @param[in] handle raft handle for managing expensive cuda resources.
 * @param[in] in     Input matrix.
 *
 * @return The transposed matrix.
 */
template <typename T, typename IndexType, typename LayoutPolicy>
[[nodiscard]] auto transpose(raft::resources const& handle,
                             device_matrix_view<T, IndexType, LayoutPolicy> in)
  -> std::enable_if_t<std::is_floating_point_v<T> &&
                        (std::is_same_v<LayoutPolicy, layout_c_contiguous> ||
                         std::is_same_v<LayoutPolicy, layout_f_contiguous>),
                      device_matrix<T, IndexType, LayoutPolicy>>
{
  auto out = make_device_matrix<T, IndexType, LayoutPolicy>(handle, in.extent(1), in.extent(0));
  ::raft::linalg::transpose(handle, in, out.view());
  return out;
}

/**
 * @brief Transpose a matrix. The output has same layout policy as the input.
 *
 * @tparam T Data type of input matrix elements.
 * @tparam LayoutPolicy Layout type of the input matrix. When layout is strided, it can
 *                      be a submatrix of a larger matrix. Arbitrary stride is not supported.
 *
 * @param[in] handle raft handle for managing expensive cuda resources.
 * @param[in] in     Input matrix.
 *
 * @return The transposed matrix.
 */
template <typename T, typename IndexType>
[[nodiscard]] auto transpose(raft::resources const& handle,
                             device_matrix_view<T, IndexType, layout_stride> in)
  -> std::enable_if_t<std::is_floating_point_v<T>, device_matrix<T, IndexType, layout_stride>>
{
  matrix_extent<size_t> exts{in.extent(1), in.extent(0)};
  using policy_type =
    typename raft::device_matrix<T, IndexType, layout_stride>::container_policy_type;
  policy_type policy{};

  RAFT_EXPECTS(in.stride(0) == 1 || in.stride(1) == 1, "Unsupported matrix layout.");
  if (in.stride(1) == 1) {
    // row-major submatrix
    std::array<size_t, 2> strides{in.extent(0), 1};
    auto layout = layout_stride::mapping<matrix_extent<size_t>>{exts, strides};
    raft::device_matrix<T, IndexType, layout_stride> out{handle, layout, policy};
    ::raft::linalg::transpose(handle, in, out.view());
    return out;
  } else {
    // col-major submatrix
    std::array<size_t, 2> strides{1, in.extent(1)};
    auto layout = layout_stride::mapping<matrix_extent<size_t>>{exts, strides};
    raft::device_matrix<T, IndexType, layout_stride> out{handle, layout, policy};
    ::raft::linalg::transpose(handle, in, out.view());
    return out;
  }
}

template <typename T, typename LayoutPolicy>
void test_transpose_with_mdspan()
{
  raft::resources handle;
  auto v = make_device_matrix<T, size_t, LayoutPolicy>(handle, 32, 3);
  T k{0};
  for (size_t i = 0; i < v.extent(0); ++i) {
    for (size_t j = 0; j < v.extent(1); ++j) {
      v(i, j) = k++;
    }
  }
  auto out = transpose(handle, v.view());
  static_assert(std::is_same_v<LayoutPolicy, typename decltype(out)::layout_type>);
  ASSERT_EQ(out.extent(0), v.extent(1));
  ASSERT_EQ(out.extent(1), v.extent(0));

  k = 0;
  for (size_t i = 0; i < out.extent(1); ++i) {
    for (size_t j = 0; j < out.extent(0); ++j) {
      ASSERT_EQ(out(j, i), k++);
    }
  }
}
}  // namespace

TEST(TransposeTest, MDSpan)
{
  test_transpose_with_mdspan<float, layout_c_contiguous>();
  test_transpose_with_mdspan<double, layout_c_contiguous>();

  test_transpose_with_mdspan<float, layout_f_contiguous>();
  test_transpose_with_mdspan<double, layout_f_contiguous>();
}

namespace {
template <typename T, typename LayoutPolicy>
void test_transpose_submatrix()
{
  raft::resources handle;
  auto v = make_device_matrix<T, size_t, LayoutPolicy>(handle, 32, 33);
  T k{0};
  size_t row_beg{3}, row_end{13}, col_beg{2}, col_end{11};
  for (size_t i = row_beg; i < row_end; ++i) {
    for (size_t j = col_beg; j < col_end; ++j) {
      v(i, j) = k++;
    }
  }

  auto vv     = v.view();
  auto submat = std::experimental::submdspan(
    vv, std::make_tuple(row_beg, row_end), std::make_tuple(col_beg, col_end));
  static_assert(std::is_same_v<typename decltype(submat)::layout_type, layout_stride>);

  auto out = transpose(handle, submat);
  ASSERT_EQ(out.extent(0), submat.extent(1));
  ASSERT_EQ(out.extent(1), submat.extent(0));

  k = 0;
  for (size_t i = 0; i < out.extent(1); ++i) {
    for (size_t j = 0; j < out.extent(0); ++j) {
      ASSERT_EQ(out(j, i), k++);
    }
  }
}
}  // namespace

TEST(TransposeTest, SubMatrix)
{
  test_transpose_submatrix<float, layout_c_contiguous>();
  test_transpose_submatrix<double, layout_c_contiguous>();

  test_transpose_submatrix<float, layout_f_contiguous>();
  test_transpose_submatrix<double, layout_f_contiguous>();
}
}  // end namespace linalg
}  // end namespace raft
