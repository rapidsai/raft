/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

 #include "../test_utils.h"
 #include <gtest/gtest.h>
 #include <raft/matrix/slice.cuh>
 #include <raft/random/rng.cuh>
 #include <raft/util/cudart_utils.hpp>
 #include <rmm/device_scalar.hpp>
 
 namespace raft {
 namespace matrix {
 
 template <typename T>
 struct SliceInputs {
   T tolerance;
   int rows, cols;
   unsigned long long int seed;
 };
 
 template <typename T>
 ::std::ostream& operator<<(::std::ostream& os, const SliceInputs<T>& I)
 {
   os << "{ " << I.tolerance << ", " << I.rows << ", " << I.cols << ", " 
      << I.seed << '}' << std::endl;
   return os;
 }
 
// Col-major slice reference test
 template <typename Type>
 void naiveSlice(
   const Type* in, Type* out, int rows, int cols, int x1, int y1, int x2, int y2)
{
    int out_rows = x2 - x1;
    // int out_cols = y2 - y1;
    for (int j = y1; j < y2; ++j) {
        for (int i = x1; i < x2; ++i) {
            out[(i - x1) + (j - y1) * out_rows] = in[i + j * rows];
        }
   }
}
 
 template <typename T>
 class SliceTest : public ::testing::TestWithParam<SliceInputs<T>> {
  public:
   SliceTest()
     : params(::testing::TestWithParam<SliceInputs<T>>::GetParam()),
       stream(handle.get_stream()),
       data(params.rows * params.cols, stream)
   {
   }
 
   void SetUp() override
   {
     std::random_device rd;
     std::default_random_engine dre(rd());
     raft::random::RngState r(params.seed);
     int rows = params.rows, cols = params.cols, len = rows * cols;
     uniform(handle, r, data.data(), len, T(-10.0), T(10.0));

     std::uniform_int_distribution<int> rowGenerator(0, rows / 2);
     int row1 = rowGenerator(dre);
     int row2 = rowGenerator(dre) + rows / 2;

     std::uniform_int_distribution<int> colGenerator(0, cols / 2);
     int col1 = colGenerator(dre);
     int col2 = colGenerator(dre) + cols / 2;

     std::vector<T> h_data(rows*cols);
     raft::update_host(h_data.data(), data.data(), rows*cols, stream);
     exp_result = naiveSlice(h_data.data(), rows, cols, row1, col1, row2, col2);
     auto input = raft::make_device_matrix_view<const T, uint32_t, raft::col_major>(
       data.data(), params.rows, params.cols, row1, col1, row2, col2);
     act_result = slice(handle, input, output);
     handle.sync_stream(stream);
   }
 
  protected:
   raft::handle_t handle;
   cudaStream_t stream;
 
   SliceInputs<T> params;
   rmm::device_uvector<T> data;
   rmm::device_uvector<T> exp_result, act_result;
 };

 ///// Row- and column-wise tests
 const std::vector<SliceInputs<float>> inputsf = {{0.00001f, 32, 1024, 1234ULL},
                                                  {0.00001f, 64, 1024, 1234ULL},
                                                  {0.00001f, 128, 1024, 1234ULL},
                                                  {0.00001f, 256, 1024, 1234ULL},
                                                  {0.00001f, 512, 512, 1234ULL},
                                                  {0.00001f, 1024, 32, 1234ULL},
                                                  {0.00001f, 1024, 64, 1234ULL},
                                                  {0.00001f, 1024, 128, 1234ULL},
                                                  {0.00001f, 1024, 256, 1234ULL}};
 
 const std::vector<SliceInputs<double>> inputsd = {
   {0.00000001, 32, 1024, 1234ULL},
   {0.00000001, 64, 1024, 1234ULL},
   {0.00000001, 128, 1024, 1234ULL},
   {0.00000001, 256, 1024, 1234ULL},
   {0.00000001, 512, 512, 1234ULL},
   {0.00000001, 1024, 32, 1234ULL},
   {0.00000001, 1024, 64, 1234ULL},
   {0.00000001, 1024, 128, 1234ULL},
   {0.00000001, 1024, 256, 1234ULL},};
 
 typedef SliceTest<float> SliceTestF;
 TEST_P(SliceTestF, Result)
 {
    ASSERT_NEAR(exp_result, act_result, params.tolerance);
    ASSERT_TRUE(devArrMatch(
        d_out_exp.data(), d_out_act.data(), params.map_length * params.ncols, raft::Compare<float>()));
 }
 
 typedef SliceTest<double> SliceTestD;
 TEST_P(SliceTestD, Result)
 {
    ASSERT_NEAR(exp_result, act_result, params.tolerance);
 }
 
 INSTANTIATE_TEST_CASE_P(SliceTests, SliceTestF, ::testing::ValuesIn(inputsf));
 
 INSTANTIATE_TEST_CASE_P(SliceTests, SliceTestD, ::testing::ValuesIn(inputsd));
 
 }  // end namespace matrix
 }  // end namespace raft
 