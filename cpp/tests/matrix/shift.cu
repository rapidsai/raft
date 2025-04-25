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

enum MODE { SINGLE_VAL, VALUES, SELF };
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
  ShiftDirection shift_direction;
  ShiftType shift_type;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const ShiftInputs<T>& inputs)
{
  os << "dataset shape=" << inputs.n_rows << "x" << inputs.dim << ", shift_direction="
     << (inputs.shift_direction == ShiftDirection::TOWARDS_END ? "towards_end"
                                                               : "TOWARDS_BEGINNING")
     << ", shift_type=" << (inputs.shift_type == ShiftType::COL ? "col" : "row") << ", mode="
     << (inputs.shift_direction == MODE::SINGLE_VAL
           ? "single_value"
           : (inputs.shift_direction == MODE::VALUES ? "values matrix" : "self"))
     << std::endl;
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
      case SINGLE_VAL: {
        raft::matrix::shift(
          handle, in_out.view(), params.val, params.k, params.shift_direction, params.shift_type);
        break;
      }
      case VALUES: {
        size_t values_cols, values_rows;
        if (params.shift_type == ShiftType::COL) {
          values_rows = params.n_rows;
          values_cols = params.values.size() / params.n_rows;
        } else {
          values_rows = params.values.size() / params.n_cols;
          values_cols = params.n_cols;
        }
        auto values = raft::make_device_matrix<T, std::uint32_t>(handle, values_rows, values_cols);
        raft::update_device(values.data_handle(),
                            params.values.data(),
                            values_rows * values_cols,
                            resource::get_cuda_stream(handle));
        raft::matrix::shift(handle,
                            in_out.view(),
                            raft::make_const_mdspan(values.view()),
                            params.shift_direction,
                            params.shift_type);
        break;
      }
      case SELF: {
        raft::matrix::shift_self(
          handle, in_out.view(), params.k, params.shift_direction, params.shift_type);
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

std::vector<float> input_matrix = {
  0.1f, 0.2f, 0.3f, 0.4f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.5f, 0.0f};

const std::vector<ShiftInputs<float>> inputs_val = {
  {SINGLE_VAL,
   input_matrix,                                                                          // input
   {100.0f, 100.0f, 0.1f, 0.2f, 100.0f, 100.0f, 0.4f, 0.3f, 100.0f, 100.0f, 0.2f, 0.3f},  // output
   {0.0f},  // values (not used here)
   100.0f,  // val
   2lu,     // k
   3lu,     // n_rows
   4lu,     // n_cols
   ShiftDirection::TOWARDS_END,
   ShiftType::COL},
  {SINGLE_VAL,
   input_matrix,                                                                    // input
   {0.2f, 0.3f, 0.4f, 100.0f, 0.3f, 0.2f, 0.1f, 100.0f, 0.3f, 0.5f, 0.0f, 100.0f},  // output
   {0.0f},  // values (not used here)
   100.0f,  // val
   1lu,     // k
   3lu,     // n_rows
   4lu,     // n_cols
   ShiftDirection::TOWARDS_BEGINNING,
   ShiftType::COL},
  {SINGLE_VAL,
   input_matrix,                                                                      // input
   {100.0f, 100.0f, 100.0f, 100.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.4f, 0.3f, 0.2f, 0.1f},  // output
   {0.0f},  // values (not used here)
   100.0f,  // val
   1lu,     // k
   3lu,     // n_rows
   4lu,     // n_cols
   ShiftDirection::TOWARDS_END,
   ShiftType::ROW},
  {SINGLE_VAL,
   input_matrix,  // input
   {0.2f,
    0.3f,
    0.5f,
    0.0f,
    100.0f,
    100.0f,
    100.0f,
    100.0f,
    100.0f,
    100.0f,
    100.0f,
    100.0f},  // output
   {0.0f},    // values (not used here)
   100.0f,    // val
   2lu,       // k
   3lu,       // n_rows
   4lu,       // n_cols
   ShiftDirection::TOWARDS_BEGINNING,
   ShiftType::ROW},
};

const std::vector<ShiftInputs<float>> inputs_values = {
  {VALUES,
   input_matrix,                                                              // input
   {9.1f, 9.2f, 0.1f, 0.2f, 9.3f, 9.4f, 0.4f, 0.3f, 9.5f, 9.6f, 0.2f, 0.3f},  // output
   {9.1f, 9.2f, 9.3f, 9.4f, 9.5f, 9.6f},                                      // values
   0.0f,  // val  (not used here)
   2lu,   // k
   3lu,   // n_rows
   4lu,   // n_cols
   ShiftDirection::TOWARDS_END,
   ShiftType::COL},
  {VALUES,
   input_matrix,                                                              // input
   {0.2f, 0.3f, 0.4f, 8.1f, 0.3f, 0.2f, 0.1f, 8.2f, 0.3f, 0.5f, 0.0f, 8.3f},  // output
   {8.1f, 8.2f, 8.3f},                                                        // values
   0.0f,  // val  (not used here)
   1lu,   // k
   3lu,   // n_rows
   4lu,   // n_cols
   ShiftDirection::TOWARDS_BEGINNING,
   ShiftType::COL},
  {VALUES,
   input_matrix,                                                              // input
   {7.1f, 7.2f, 7.3f, 7.4f, 0.1f, 0.2f, 0.3f, 0.4f, 0.4f, 0.3f, 0.2f, 0.1f},  // output
   {7.1f, 7.2f, 7.3f, 7.4f},                                                  // values
   0.0f,  // val  (not used here)
   1lu,   // k
   3lu,   // n_rows
   4lu,   // n_cols
   ShiftDirection::TOWARDS_END,
   ShiftType::ROW},
  {VALUES,
   input_matrix,                                                              // input
   {0.2f, 0.3f, 0.5f, 0.0f, 6.1f, 6.2f, 6.3f, 6.4f, 6.5f, 6.6f, 6.7f, 6.8f},  // output
   {6.1f, 6.2f, 6.3f, 6.4f, 6.5f, 6.6f, 6.7f, 6.8f},                          // values
   0.0f,  // val  (not used here)
   2lu,   // k
   3lu,   // n_rows
   4lu,   // n_cols
   ShiftDirection::TOWARDS_BEGINNING,
   ShiftType::ROW},
};

const std::vector<ShiftInputs<float>> inputs_self = {
  {SELF,
   input_matrix,                                                              // input
   {0.0f, 0.0f, 0.1f, 0.2f, 1.0f, 1.0f, 0.4f, 0.3f, 2.0f, 2.0f, 0.2f, 0.3f},  // output
   {},    // values (not used here)
   0.0f,  // val  (not used here)
   2lu,   // k
   3lu,   // n_rows
   4lu,   // n_cols
   ShiftDirection::TOWARDS_END,
   ShiftType::COL},
  {SELF,
   input_matrix,                                                              // input
   {0.2f, 0.3f, 0.4f, 0.0f, 0.3f, 0.2f, 0.1f, 1.0f, 0.3f, 0.5f, 0.0f, 2.0f},  // output
   {},    // values (not used here)
   0.0f,  // val  (not used here)
   1lu,   // k
   3lu,   // n_rows
   4lu,   // n_cols
   ShiftDirection::TOWARDS_BEGINNING,
   ShiftType::COL},
  {SELF,
   input_matrix,                                                              // input
   {0.0f, 1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.4f, 0.3f, 0.2f, 0.1f},  // output
   {},    // values (not used here)
   0.0f,  // val  (not used here)
   1lu,   // k
   3lu,   // n_rows
   4lu,   // n_cols
   ShiftDirection::TOWARDS_END,
   ShiftType::ROW},
  {SELF,
   input_matrix,                                                              // input
   {0.2f, 0.3f, 0.5f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f},  // output
   {},    // values (not used here)
   0.0f,  // val  (not used here)
   2lu,   // k
   3lu,   // n_rows
   4lu,   // n_cols
   ShiftDirection::TOWARDS_BEGINNING,
   ShiftType::ROW}};

typedef ShiftTest<float> ShiftTestF;
TEST_P(ShiftTestF, Result)
{
  ASSERT_TRUE(devArrMatch(expected.data_handle(),
                          in_out.data_handle(),
                          params.n_rows * params.n_cols,
                          Compare<float>(),
                          resource::get_cuda_stream(handle)));
}

INSTANTIATE_TEST_SUITE_P(ShiftTestSingleVal, ShiftTestF, ::testing::ValuesIn(inputs_val));
INSTANTIATE_TEST_SUITE_P(ShiftTestValues, ShiftTestF, ::testing::ValuesIn(inputs_values));
INSTANTIATE_TEST_SUITE_P(ShiftTestSelf, ShiftTestF, ::testing::ValuesIn(inputs_self));

}  // namespace matrix
}  // namespace raft
