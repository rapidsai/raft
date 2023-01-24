/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <complex>
#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_serializer.hpp>
#include <string>
#include <vector>

namespace raft {

TEST(MDArraySerializer, Basic)
{
  raft::handle_t handle{};

  std::vector<float> vec{1.0, 2.0, 3.0, 4.0};

  using mdspan_matrix2d_c_layout =
    raft::host_mdspan<float, raft::dextents<std::size_t, 2>, raft::layout_c_contiguous>;
  auto span = mdspan_matrix2d_c_layout(vec.data(), 2, 2);

  std::ofstream of("/home/phcho/tmp/foobar.npy", std::ios::out | std::ios::binary);
  serialize_mdspan(handle, of, span);
}

TEST(MDArraySerializer, Tuple2String)
{
  {
    std::vector<int> tuple{};
    EXPECT_EQ(detail::numpy_serializer::tuple_to_string(tuple), "()");
  }
  {
    std::vector<int> tuple{2};
    EXPECT_EQ(detail::numpy_serializer::tuple_to_string(tuple), "(2,)");
  }
  {
    std::vector<int> tuple{2, 3};
    EXPECT_EQ(detail::numpy_serializer::tuple_to_string(tuple), "(2, 3)");
  }
  {
    std::vector<int> tuple{2, 3, 10, 20};
    EXPECT_EQ(detail::numpy_serializer::tuple_to_string(tuple), "(2, 3, 10, 20)");
  }
}

TEST(MDArraySerializer, NumPyDType)
{
  const char expected_endian_char = RAFT_SYSTEM_LITTLE_ENDIAN ? '<' : '>';
  {
    const detail::numpy_serializer::dtype_t expected_dtype{
      expected_endian_char, 'f', sizeof(float)};
    EXPECT_EQ(detail::numpy_serializer::get_numpy_dtype<float>(), expected_dtype);
  }
  {
    const detail::numpy_serializer::dtype_t expected_dtype{
      expected_endian_char, 'f', sizeof(long double)};
    EXPECT_EQ(detail::numpy_serializer::get_numpy_dtype<long double>(), expected_dtype);
  }
  {
    const detail::numpy_serializer::dtype_t expected_dtype{'|', 'i', sizeof(signed char)};
    EXPECT_EQ(detail::numpy_serializer::get_numpy_dtype<signed char>(), expected_dtype);
  }
  {
    const detail::numpy_serializer::dtype_t expected_dtype{
      expected_endian_char, 'i', sizeof(std::int64_t)};
    EXPECT_EQ(detail::numpy_serializer::get_numpy_dtype<std::int64_t>(), expected_dtype);
  }
  {
    const detail::numpy_serializer::dtype_t expected_dtype{'|', 'u', sizeof(unsigned char)};
    EXPECT_EQ(detail::numpy_serializer::get_numpy_dtype<unsigned char>(), expected_dtype);
  }
  {
    const detail::numpy_serializer::dtype_t expected_dtype{
      expected_endian_char, 'u', sizeof(std::uint64_t)};
    EXPECT_EQ(detail::numpy_serializer::get_numpy_dtype<std::uint64_t>(), expected_dtype);
  }
  {
    const detail::numpy_serializer::dtype_t expected_dtype{
      expected_endian_char, 'c', sizeof(std::complex<double>)};
    EXPECT_EQ(detail::numpy_serializer::get_numpy_dtype<std::complex<double>>(), expected_dtype);
  }
}

TEST(MDArraySerializer, WriteHeader)
{
  using namespace std::string_literals;
  std::ostringstream oss;
  detail::numpy_serializer::header_t header{{'<', 'f', 8}, false, {2, 10, 5}};
  detail::numpy_serializer::write_header(oss, header);
  EXPECT_EQ(oss.str(),
            "\x93NUMPY\x01\x00"s  // magic string + version (1.0)
            "\x46\x00"s           // HEADER_LEN = 70, in little endian
            "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 10, 5)}"s  // header
            "\x20\x20\x20\x20\x20\x20\x20\x20\n"s                             // padding
  );
}

}  // namespace raft
