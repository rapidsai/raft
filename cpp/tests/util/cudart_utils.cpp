/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <regex>
#include <vector>

namespace raft {

#define TEST_ADD_FILENAME(s)    \
  {                             \
    s += std::string{__FILE__}; \
  }

std::string reg_escape(const std::string& s)
{
  static const std::regex SPECIAL_CHARS{R"([-[\]{}()*+?.,\^$|#\s])"};
  return std::regex_replace(s, SPECIAL_CHARS, R"(\$&)");
}

TEST(Raft, Utils)
{
  ASSERT_NO_THROW(ASSERT(1 == 1, "Should not assert!"));
  ASSERT_THROW(ASSERT(1 != 1, "Should assert!"), exception);
  ASSERT_THROW(THROW("Should throw!"), exception);
  ASSERT_NO_THROW(RAFT_CUDA_TRY(cudaFree(nullptr)));

  // test for long error message strings
  std::string test{"This is a test string repeated many times. "};
  for (size_t i = 0; i < 6; ++i)
    test += test;
  EXPECT_TRUE(test.size() > 2048) << "size of test string is: " << test.size();
  auto test_format    = test + "%d";
  auto* test_format_c = test_format.c_str();

  std::string file{};
  TEST_ADD_FILENAME(file);
  std::string reg_file = reg_escape(file);

  // THROW has to convert the test string into an exception string
  try {
    ASSERT(1 != 1, test_format_c, 121);
  } catch (const raft::exception& e) {
    std::string msg_full{e.what()};
    // only use first line
    std::string msg = msg_full.substr(0, msg_full.find('\n'));
    std::string re_exp{"^exception occurred! file="};
    re_exp += reg_file;
    // test code must be at line >10 (copyright), assume line is never >9999
    re_exp += " line=\\d{2,4}: ";
    re_exp += reg_escape(test);
    re_exp += "121$";
    EXPECT_TRUE(std::regex_match(msg, std::regex(re_exp))) << "message:'" << msg << "'" << std::endl
                                                           << "expected regex:'" << re_exp << "'";
  }

  // Now we test SET_ERROR_MSG instead of THROW
  std::string msg{"prefix:"};
  ASSERT_NO_THROW(SET_ERROR_MSG(msg, "location prefix:", test_format_c, 123));

  std::string re_exp{"^prefix:location prefix:file="};
  re_exp += reg_file;
  // test code must be at line >10 (copyright), assume line is never >9999
  re_exp += " line=\\d{2,4}: ";
  re_exp += reg_escape(test);
  re_exp += "123$";
  EXPECT_TRUE(std::regex_match(msg, std::regex(re_exp))) << "message:'" << msg << "'" << std::endl
                                                         << "expected regex:'" << re_exp << "'";
}

TEST(Raft, GetDeviceForAddress)
{
  resources handle;
  std::vector<int> h(1);
  ASSERT_EQ(-1, raft::get_device_for_address(h.data()));

  rmm::device_uvector<int> d(1, resource::get_cuda_stream(handle));
  ASSERT_EQ(0, raft::get_device_for_address(d.data()));
}

TEST(Raft, Copy2DAsync)
{
  using DType = float;

  constexpr size_t rows      = 4;
  constexpr size_t cols      = 5;
  constexpr size_t pitch     = 8;
  constexpr size_t elem_size = sizeof(DType);
  constexpr size_t width     = cols;
  constexpr size_t height    = rows;

  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<DType> d_src(pitch * elem_size * rows, stream);
  rmm::device_uvector<DType> d_dst(pitch * elem_size * rows, stream);

  std::vector<DType> h_src(rows * pitch, -1.0f);
  std::vector<DType> h_dst(rows * pitch, 0.0f);
  std::vector<DType> h_dst_baseline(rows * pitch, 0.0f);

  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < pitch; ++c) {
      h_src[r * pitch + c] = static_cast<DType>(r * pitch + c);
      if (r < height && c < cols) {
        h_dst_baseline[r * pitch + c] = static_cast<DType>(r * pitch + c);
      }
    }
  }
  RAFT_CUDA_TRY(
    cudaMemcpy(d_src.data(), h_src.data(), pitch * elem_size * rows, cudaMemcpyHostToDevice));
  RAFT_CUDA_TRY(
    cudaMemcpy(d_dst.data(), h_dst.data(), pitch * elem_size * rows, cudaMemcpyHostToDevice));

  raft::copy_matrix(d_dst.data(), pitch, d_src.data(), pitch, width, height, stream);
  RAFT_CUDA_TRY(
    cudaMemcpy(h_dst.data(), d_dst.data(), pitch * elem_size * rows, cudaMemcpyDeviceToHost));

  raft::resource::sync_stream(handle);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < pitch; ++c) {
      ASSERT_EQ(h_dst[r * pitch + c], h_dst_baseline[r * pitch + c])
        << "Mismatch at row " << r << " col " << c;
    }
  }
}

}  // namespace raft
