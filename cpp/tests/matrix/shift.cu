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

#include "../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/shift.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

#include <gtest/gtest.h>

#include <cstdint>

namespace raft {
namespace matrix {

enum MODE { VAL, VALUES, SELF };
template <typename T>
struct ShiftInputs {
  MODE mode;
  std::vector<T> input_matrix;
  std::vector<T> output_matrix;
  std::vector<T> values;
  T val;
  size_t k;
  size_t n_rows;
  size_t n_cols;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const ShiftInputs<T>& dims)
{
  return os;
}

template <typename T>
class ShiftTest : public ::testing::TestWithParam<ShiftInputs<T>> {
 public:
  ShiftTest()
    : params(::testing::TestWithParam<ShiftInputs<T>>::GetParam()),
      in_out(raft::make_device_matrix<T, std::uint32_t, row_major>(
        handle, params.n_rows, params.n_cols)),
      expected(raft::make_device_matrix<T, std::uint32_t>(handle, params.n_rows, params.n_cols))
  {
    raft::update_device(in_out.data_handle(),
                        params.input_matrix.data(),
                        params.input_matrix.size(),
                        resource::get_cuda_stream(handle));
    raft::update_device(expected.data_handle(),
                        params.output_matrix.data(),
                        params.output_matrix.size(),
                        resource::get_cuda_stream(handle));

    switch (params.mode) {
      case VAL: {
        raft::matrix::col_right_shift(handle, in_out.view(), params.val, params.k);
        break;
      }
      case VALUES: {
        size_t values_cols = params.values.size() / params.n_rows;
        auto values =
          raft::make_device_matrix<T, std::uint32_t>(handle, params.n_rows, values_cols);
        raft::update_device(values.data_handle(),
                            params.values.data(),
                            params.n_rows * values_cols,
                            resource::get_cuda_stream(handle));
        raft::matrix::col_right_shift(
          handle, in_out.view(), raft::make_const_mdspan(values.view()));  // Hypothetical API
        break;
      }
      case SELF: {
        raft::matrix::col_right_shift_self(handle, in_out.view(), params.k);  // Hypothetical API
        break;
      }
    }

    resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  ShiftInputs<T> params;

  raft::device_matrix<T, std::uint32_t, row_major> in_out;
  raft::device_matrix<T, std::uint32_t> expected;
};

const std::vector<ShiftInputs<float>> inputs_val = {
  {
    VAL,
    {0.1f, 0.2f, 0.3f, 0.4f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.5f, 0.0f},              // input
    {100.0f, 100.0f, 0.1f, 0.2f, 100.0f, 100.0f, 0.4f, 0.3f, 100.0f, 100.0f, 0.2f, 0.3f},  // output
    {0.0f},  // values (not used here)
    100.0f,  // val
    2lu,     // k
    3lu,     // n_rows
    4lu      // n_cols
  },
  {
    VAL,
    {0.1f, 0.2f, 0.3f, 0.4f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.5f, 0.0f},          // input
    {200.0f, 0.1f, 0.2f, 200.0f, 0.4f, 0.4f, 200.0f, 0.2f, 0.1f, 200.0f, 0.3f, 0.5f},  // output
    {0.0f},  // values (not used here)
    200.0f,  // val
    1lu,     // k
    4lu,     // n_rows
    3lu      // n_cols
  },
};

const std::vector<ShiftInputs<float>> inputs_values = {
  {
    VALUES,
    {0.1f, 0.2f, 0.3f, 0.4f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.5f, 0.0f},              // input
    {100.0f, 200.0f, 0.1f, 0.2f, 300.0f, 400.0f, 0.4f, 0.3f, 500.0f, 600.0f, 0.2f, 0.3f},  // output
    {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f},                                      // values
    0.0f,  // val (not used here)
    0lu,   // k (not used here)
    3lu,   // n_rows
    4lu    // n_cols
  },
  {
    VALUES,
    {0.1f, 0.2f, 0.3f, 0.4f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.5f, 0.0f},          // input
    {100.0f, 0.1f, 0.2f, 200.0f, 0.4f, 0.4f, 300.0f, 0.2f, 0.1f, 400.0f, 0.3f, 0.5f},  // output
    {100.0f, 200.0f, 300.0f, 400.0f},                                                  // values
    0.0f,  // val (not used here)
    0lu,   // k (not used here)
    4lu,   // n_rows
    3lu    // n_cols
  },
};

const std::vector<ShiftInputs<float>> inputs_self = {
  {
    SELF,
    {0.1f, 0.2f, 0.3f, 0.4f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.5f, 0.0f},  // input
    {0.0f, 0.1f, 0.2f, 0.3f, 1.0f, 0.4f, 0.3f, 0.2f, 2.0f, 0.2f, 0.3f, 0.5f},  // output
    {0.0f},  // values (not used here)
    0.0f,    // val (not used here)
    1lu,     // k
    3lu,     // n_rows
    4lu      // n_cols
  },
  {
    SELF,
    {0.1f, 0.2f, 0.3f, 0.4f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.5f, 0.0f},        // input
    {0.0f, 0.0f, 0.1f, 1.0f, 1.0f, 0.4f, 2.0f, 2.0f, 0.2f, 0.1f, 3.0f, 3.0f, 0.3f},  // output
    {0.0f},  // values (not used here)
    0.0f,    // val (not used here)
    2lu,     // k
    4lu,     // n_rows
    3lu      // n_cols
  }};

typedef ShiftTest<float> ShiftTestF;
TEST_P(ShiftTestF, Result)
{
  ASSERT_TRUE(devArrMatch(expected.data_handle(),
                          in_out.data_handle(),
                          params.n_rows * params.n_cols,
                          Compare<float>(),
                          resource::get_cuda_stream(handle)));
}

INSTANTIATE_TEST_SUITE_P(ShiftTestVal, ShiftTestF, ::testing::ValuesIn(inputs_val));
INSTANTIATE_TEST_SUITE_P(ShiftTestValues, ShiftTestF, ::testing::ValuesIn(inputs_values));
INSTANTIATE_TEST_SUITE_P(ShiftTestSelf, ShiftTestF, ::testing::ValuesIn(inputs_self));

}  // namespace matrix
}  // namespace raft
