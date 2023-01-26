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

#include <gtest/gtest.h>

#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/serialize.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <complex>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace raft {

template <typename MDSpanType, typename VectorType, typename... Args>
void test_mdspan_roundtrip(const raft::device_resources& handle, VectorType& vec, Args... dims)
{
  VectorType vec2(vec.size());

  auto span = MDSpanType(thrust::raw_pointer_cast(vec.data()), dims...);
  std::ostringstream oss;
  serialize_mdspan(handle, oss, span);

  auto span2 = MDSpanType(thrust::raw_pointer_cast(vec2.data()), dims...);
  std::istringstream iss(oss.str());
  deserialize_mdspan(handle, iss, span2);
  EXPECT_EQ(vec, vec2);
}

TEST(NumPySerializerMDSpan, E2ERoundTrip)
{
  raft::device_resources handle{};
  thrust::host_vector<float> vec = std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

  using mdspan_matrix2d_c_layout =
    raft::host_mdspan<float, raft::dextents<std::size_t, 2>, raft::layout_c_contiguous>;
  using mdspan_matrix2d_f_layout =
    raft::host_mdspan<float, raft::dextents<std::size_t, 2>, raft::layout_f_contiguous>;
  using mdspan_matrix3d_c_layout =
    raft::host_mdspan<float, raft::dextents<std::size_t, 3>, raft::layout_c_contiguous>;
  using mdspan_matrix3d_f_layout =
    raft::host_mdspan<float, raft::dextents<std::size_t, 3>, raft::layout_f_contiguous>;

  test_mdspan_roundtrip<mdspan_matrix2d_c_layout>(handle, vec, 2, 4);
  test_mdspan_roundtrip<mdspan_matrix2d_f_layout>(handle, vec, 2, 4);
  test_mdspan_roundtrip<mdspan_matrix2d_c_layout>(handle, vec, 1, 8);
  test_mdspan_roundtrip<mdspan_matrix2d_f_layout>(handle, vec, 1, 8);
  test_mdspan_roundtrip<mdspan_matrix3d_c_layout>(handle, vec, 2, 2, 2);
  test_mdspan_roundtrip<mdspan_matrix3d_f_layout>(handle, vec, 2, 2, 2);
  test_mdspan_roundtrip<mdspan_matrix3d_c_layout>(handle, vec, 1, 2, 4);
  test_mdspan_roundtrip<mdspan_matrix3d_f_layout>(handle, vec, 1, 2, 4);

  using device_mdspan_matrix2d_c_layout =
    raft::device_mdspan<float, raft::dextents<std::size_t, 2>, raft::layout_c_contiguous>;
  using device_mdspan_matrix2d_f_layout =
    raft::device_mdspan<float, raft::dextents<std::size_t, 2>, raft::layout_f_contiguous>;
  using device_mdspan_matrix3d_c_layout =
    raft::device_mdspan<float, raft::dextents<std::size_t, 3>, raft::layout_c_contiguous>;
  using device_mdspan_matrix3d_f_layout =
    raft::device_mdspan<float, raft::dextents<std::size_t, 3>, raft::layout_f_contiguous>;

  thrust::device_vector<float> d_vec(vec);
  test_mdspan_roundtrip<device_mdspan_matrix2d_c_layout>(handle, d_vec, 2, 4);
  test_mdspan_roundtrip<device_mdspan_matrix2d_f_layout>(handle, d_vec, 2, 4);
  test_mdspan_roundtrip<device_mdspan_matrix2d_c_layout>(handle, d_vec, 1, 8);
  test_mdspan_roundtrip<device_mdspan_matrix2d_f_layout>(handle, d_vec, 1, 8);
  test_mdspan_roundtrip<device_mdspan_matrix3d_c_layout>(handle, d_vec, 2, 2, 2);
  test_mdspan_roundtrip<device_mdspan_matrix3d_f_layout>(handle, d_vec, 2, 2, 2);
  test_mdspan_roundtrip<device_mdspan_matrix3d_c_layout>(handle, d_vec, 1, 2, 4);
  test_mdspan_roundtrip<device_mdspan_matrix3d_f_layout>(handle, d_vec, 1, 2, 4);
}

TEST(NumPySerializerMDSpan, HeaderRoundTrip)
{
  char byteorder = RAFT_NUMPY_HOST_ENDIAN_CHAR;
  for (char kind : std::vector<char>{'f', 'i', 'u', 'c'}) {
    for (unsigned int itemsize : std::vector<unsigned int>{1, 2, 4, 8, 16}) {
      for (bool fortran_order : std::vector<bool>{true, false}) {
        for (const auto& shape : std::vector<std::vector<detail::numpy_serializer::ndarray_len_t>>{
               {10}, {2, 2}, {10, 30, 100}, {}}) {
          detail::numpy_serializer::dtype_t dtype{byteorder, kind, itemsize};
          detail::numpy_serializer::header_t header{dtype, fortran_order, shape};
          std::ostringstream oss;
          detail::numpy_serializer::write_header(oss, header);
          std::istringstream iss(oss.str());
          auto header2 = detail::numpy_serializer::read_header(iss);
          EXPECT_EQ(header, header2);
        }
      }
    }
  }
}

TEST(NumPySerializerMDSpan, Tuple2String)
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

TEST(NumPySerializerMDSpan, NumPyDType)
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

TEST(NumPySerializerMDSpan, WriteHeader)
{
  using namespace std::string_literals;
  std::ostringstream oss;
  detail::numpy_serializer::header_t header{{'<', 'f', 8}, false, {2, 10, 5}};
  detail::numpy_serializer::write_header(oss, header);
  EXPECT_EQ(oss.str(),
            "\x93NUMPY\x01\x00"s  // magic string + version (1.0)
            "\x76\x00"s           // HEADER_LEN = 118, in little endian
            "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 10, 5)}"s  // header
            "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20"s                       // padding
            "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20"s
            "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20"s
            "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20"s
            "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20"s
            "\x20\x20\x20\x20\x20\x20\n"s);
}

TEST(NumPySerializerMDSpan, ParsePyDict)
{
  std::string dict{"{'apple': 2, 'pie': 'is', 'delicious': True, 'piece of': 'cake'}"};
  auto parse =
    detail::numpy_serializer::parse_pydict(dict, {"apple", "pie", "delicious", "piece of"});
  auto expected_parse = std::map<std::string, std::string>{
    {"apple", "2"}, {"pie", "'is'"}, {"delicious", "True"}, {"piece of", "'cake'"}};
  EXPECT_EQ(parse, expected_parse);
}

TEST(NumPySerializerMDSpan, ParsePyString)
{
  EXPECT_EQ(detail::numpy_serializer::parse_pystring("'foobar'"), "foobar");
}

TEST(NumPySerializerMDSpan, ParsePyTuple)
{
  {
    std::string tuple_str{"(2,)"};
    std::vector<std::string> expected_parse{"2"};
    EXPECT_EQ(detail::numpy_serializer::parse_pytuple(tuple_str), expected_parse);
  }
  {
    std::string tuple_str{"(2, 3)"};
    std::vector<std::string> expected_parse{"2", "3"};
    EXPECT_EQ(detail::numpy_serializer::parse_pytuple(tuple_str), expected_parse);
  }
  {
    std::string tuple_str{"(2, 3, 10, 20)"};
    std::vector<std::string> expected_parse{"2", "3", "10", "20"};
    EXPECT_EQ(detail::numpy_serializer::parse_pytuple(tuple_str), expected_parse);
  }
}

}  // namespace raft
