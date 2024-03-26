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

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#if defined(_RAFT_HAS_CUDA)
#include <cuda_fp16.h>
#endif

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace raft {

namespace detail {

namespace numpy_serializer {

/**
 * A small implementation of NumPy serialization format.
 * Reference: https://numpy.org/doc/1.23/reference/generated/numpy.lib.format.html
 *
 * Adapted from https://github.com/llohse/libnpy/blob/master/include/npy.hpp, using the following
 * license:
 *
 * MIT License
 *
 * Copyright (c) 2021 Leon Merten Lohse
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#define RAFT_NUMPY_LITTLE_ENDIAN_CHAR  '<'
#define RAFT_NUMPY_BIG_ENDIAN_CHAR     '>'
#define RAFT_NUMPY_NO_ENDIAN_CHAR      '|'
#define RAFT_NUMPY_MAGIC_STRING        "\x93NUMPY"
#define RAFT_NUMPY_MAGIC_STRING_LENGTH 6

#if RAFT_SYSTEM_LITTLE_ENDIAN == 1
#define RAFT_NUMPY_HOST_ENDIAN_CHAR RAFT_NUMPY_LITTLE_ENDIAN_CHAR
#else  // RAFT_SYSTEM_LITTLE_ENDIAN == 1
#define RAFT_NUMPY_HOST_ENDIAN_CHAR RAFT_NUMPY_BIG_ENDIAN_CHAR
#endif  // RAFT_SYSTEM_LITTLE_ENDIAN == 1

using ndarray_len_t = std::uint64_t;

struct dtype_t {
  const char byteorder;
  const char kind;
  const unsigned int itemsize;

  std::string to_string() const
  {
    char buf[16] = {0};
    std::sprintf(buf, "%c%c%u", byteorder, kind, itemsize);
    return std::string(buf);
  }

  bool operator==(const dtype_t& other) const
  {
    return (byteorder == other.byteorder && kind == other.kind && itemsize == other.itemsize);
  }
};

struct header_t {
  const dtype_t dtype;
  const bool fortran_order;
  const std::vector<ndarray_len_t> shape;

  bool operator==(const header_t& other) const
  {
    return (dtype == other.dtype && fortran_order == other.fortran_order && shape == other.shape);
  }
};

template <class T>
struct is_complex : std::false_type {};
template <class T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
inline dtype_t get_numpy_dtype()
{
  return {RAFT_NUMPY_HOST_ENDIAN_CHAR, 'f', sizeof(T)};
}

#if defined(_RAFT_HAS_CUDA)
template <typename T,
          typename std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, half>, bool> = true>
inline dtype_t get_numpy_dtype()
{
  return {RAFT_NUMPY_HOST_ENDIAN_CHAR, 'e', sizeof(T)};
}
#endif

template <typename T,
          typename std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T>, bool> = true>
inline dtype_t get_numpy_dtype()
{
  const char endian_char =
    (sizeof(T) == 1 ? RAFT_NUMPY_NO_ENDIAN_CHAR : RAFT_NUMPY_HOST_ENDIAN_CHAR);
  return {endian_char, 'i', sizeof(T)};
}

template <typename T,
          typename std::enable_if_t<std::is_integral_v<T> && std::is_unsigned_v<T>, bool> = true>
inline dtype_t get_numpy_dtype()
{
  const char endian_char =
    (sizeof(T) == 1 ? RAFT_NUMPY_NO_ENDIAN_CHAR : RAFT_NUMPY_HOST_ENDIAN_CHAR);
  return {endian_char, 'u', sizeof(T)};
}

template <typename T, typename std::enable_if_t<is_complex<T>{}, bool> = true>
inline dtype_t get_numpy_dtype()
{
  return {RAFT_NUMPY_HOST_ENDIAN_CHAR, 'c', sizeof(T)};
}

template <typename T, typename std::enable_if_t<std::is_enum_v<T>, bool> = true>
inline dtype_t get_numpy_dtype()
{
  return get_numpy_dtype<std::underlying_type_t<T>>();
}

template <typename T>
inline std::string tuple_to_string(const std::vector<T>& tuple)
{
  std::ostringstream oss;
  if (tuple.empty()) {
    oss << "()";
  } else if (tuple.size() == 1) {
    oss << "(" << tuple.front() << ",)";
  } else {
    oss << "(";
    for (std::size_t i = 0; i < tuple.size() - 1; ++i) {
      oss << tuple[i] << ", ";
    }
    oss << tuple.back() << ")";
  }
  return oss.str();
}

inline std::string header_to_string(const header_t& header)
{
  std::ostringstream oss;
  oss << "{'descr': '" << header.dtype.to_string()
      << "', 'fortran_order': " << (header.fortran_order ? "True" : "False")
      << ", 'shape': " << tuple_to_string(header.shape) << "}";
  return oss.str();
}

inline std::string trim(const std::string& str)
{
  const std::string whitespace = " \t";
  auto begin                   = str.find_first_not_of(whitespace);
  if (begin == std::string::npos) { return ""; }
  auto end = str.find_last_not_of(whitespace);

  return str.substr(begin, end - begin + 1);
}

// A poor man's parser for Python dictionary
// TODO(hcho3): Consider writing a proper parser
// Limitation: can only parse a flat dictionary; all values are assumed to non-objects
// Limitation: must know all the keys ahead of time; you get undefined behavior if you omit any key
inline std::map<std::string, std::string> parse_pydict(std::string str,
                                                       const std::vector<std::string>& keys)
{
  std::map<std::string, std::string> result;

  // Unwrap dictionary
  str = trim(str);
  RAFT_EXPECTS(str.front() == '{' && str.back() == '}', "Expected a Python dictionary");
  str = str.substr(1, str.length() - 2);

  // Get the position of each key and put it in the list
  std::vector<std::pair<std::size_t, std::string>> positions;
  for (auto const& key : keys) {
    std::size_t pos = str.find("'" + key + "'");
    RAFT_EXPECTS(pos != std::string::npos, "Missing '%s' key.", key.c_str());
    positions.emplace_back(pos, key);
  }
  // Sort the list
  std::sort(positions.begin(), positions.end());

  // Extract each key-value pair
  for (std::size_t i = 0; i < positions.size(); ++i) {
    std::string key = positions[i].second;

    std::size_t begin     = positions[i].first;
    std::size_t end       = (i + 1 < positions.size() ? positions[i + 1].first : std::string::npos);
    std::string raw_value = trim(str.substr(begin, end - begin));
    if (raw_value.back() == ',') { raw_value.pop_back(); }
    std::size_t sep_pos = raw_value.find_first_of(":");
    if (sep_pos == std::string::npos) {
      result[key] = "";
    } else {
      result[key] = trim(raw_value.substr(sep_pos + 1));
    }
  }

  return result;
}

inline std::string parse_pystring(std::string str)
{
  RAFT_EXPECTS(str.front() == '\'' && str.back() == '\'', "Invalid Python string: %s", str.c_str());
  return str.substr(1, str.length() - 2);
}

inline bool parse_pybool(std::string str)
{
  if (str == "True") {
    return true;
  } else if (str == "False") {
    return false;
  } else {
    RAFT_FAIL("Invalid Python boolean: %s", str.c_str());
  }
}

inline std::vector<std::string> parse_pytuple(std::string str)
{
  std::vector<std::string> result;

  str = trim(str);
  RAFT_EXPECTS(str.front() == '(' && str.back() == ')', "Invalid Python tuple: %s", str.c_str());
  str = str.substr(1, str.length() - 2);

  std::istringstream iss(str);
  for (std::string token; std::getline(iss, token, ',');) {
    result.push_back(trim(token));
  }

  return result;
}

inline dtype_t parse_descr(std::string typestr)
{
  RAFT_EXPECTS(typestr.length() >= 3, "Invalid typestr: Too short");
  char byteorder_c       = typestr.at(0);
  char kind_c            = typestr.at(1);
  std::string itemsize_s = typestr.substr(2);

  const char endian_chars[] = {
    RAFT_NUMPY_LITTLE_ENDIAN_CHAR, RAFT_NUMPY_BIG_ENDIAN_CHAR, RAFT_NUMPY_NO_ENDIAN_CHAR};
  const char numtype_chars[] = {'f', 'i', 'u', 'c', 'e'};

  RAFT_EXPECTS(std::find(std::begin(endian_chars), std::end(endian_chars), byteorder_c) !=
                 std::end(endian_chars),
               "Invalid typestr: unrecognized byteorder %c",
               byteorder_c);
  RAFT_EXPECTS(std::find(std::begin(numtype_chars), std::end(numtype_chars), kind_c) !=
                 std::end(numtype_chars),
               "Invalid typestr: unrecognized kind %c",
               kind_c);
  unsigned int itemsize = std::stoul(itemsize_s);

  return {byteorder_c, kind_c, itemsize};
}

inline void write_magic(std::ostream& os)
{
  os.write(RAFT_NUMPY_MAGIC_STRING, RAFT_NUMPY_MAGIC_STRING_LENGTH);
  RAFT_EXPECTS(os.good(), "Error writing magic string");
  // Use version 1.0
  os.put(1);
  os.put(0);
  RAFT_EXPECTS(os.good(), "Error writing magic string");
}

inline void read_magic(std::istream& is)
{
  char magic_buf[RAFT_NUMPY_MAGIC_STRING_LENGTH + 2] = {0};
  is.read(magic_buf, RAFT_NUMPY_MAGIC_STRING_LENGTH + 2);
  RAFT_EXPECTS(is.good(), "Error reading magic string");

  RAFT_EXPECTS(std::memcmp(magic_buf, RAFT_NUMPY_MAGIC_STRING, RAFT_NUMPY_MAGIC_STRING_LENGTH) == 0,
               "The given stream does not have a valid NumPy format.");

  std::uint8_t version_major = magic_buf[RAFT_NUMPY_MAGIC_STRING_LENGTH];
  std::uint8_t version_minor = magic_buf[RAFT_NUMPY_MAGIC_STRING_LENGTH + 1];
  RAFT_EXPECTS(version_major == 1 && version_minor == 0,
               "Unsupported NumPy version: %d.%d",
               version_major,
               version_minor);
}

inline void write_header(std::ostream& os, const header_t& header)
{
  std::string header_dict     = header_to_string(header);
  std::size_t preamble_length = RAFT_NUMPY_MAGIC_STRING_LENGTH + 2 + 2 + header_dict.length() + 1;
  RAFT_EXPECTS(preamble_length < 255 * 255, "Header too long");
  // Enforce 64-byte alignment
  std::size_t padding_len = 64 - preamble_length % 64;
  std::string padding(padding_len, ' ');

  write_magic(os);

  // Write header length
  std::uint8_t header_len_le16[2];
  std::uint16_t header_len =
    static_cast<std::uint16_t>(header_dict.length() + padding.length() + 1);
  header_len_le16[0] = (header_len >> 0) & 0xff;
  header_len_le16[1] = (header_len >> 8) & 0xff;
  os.put(header_len_le16[0]);
  os.put(header_len_le16[1]);
  RAFT_EXPECTS(os.good(), "Error writing HEADER_LEN");

  os << header_dict << padding << "\n";
  RAFT_EXPECTS(os.good(), "Error writing header dict");
}

inline std::string read_header_bytes(std::istream& is)
{
  read_magic(is);

  // Read header length
  std::uint8_t header_len_le16[2];
  is.read(reinterpret_cast<char*>(header_len_le16), 2);
  RAFT_EXPECTS(is.good(), "Error while reading HEADER_LEN");
  const std::uint32_t header_length = (header_len_le16[0] << 0) | (header_len_le16[1] << 8);

  std::vector<char> header_bytes(header_length);
  is.read(header_bytes.data(), header_length);
  RAFT_EXPECTS(is.good(), "Error while reading the header");

  return std::string(header_bytes.data(), header_length);
}

inline header_t read_header(std::istream& is)
{
  std::string header_bytes = read_header_bytes(is);

  // remove trailing newline
  RAFT_EXPECTS(header_bytes.back() == '\n', "Invalid NumPy header");
  header_bytes.pop_back();

  // parse the header dict
  auto header_dict   = parse_pydict(header_bytes, {"descr", "fortran_order", "shape"});
  dtype_t descr      = parse_descr(parse_pystring(header_dict["descr"]));
  bool fortran_order = parse_pybool(header_dict["fortran_order"]);
  std::vector<ndarray_len_t> shape;
  auto shape_tup_str = parse_pytuple(header_dict["shape"]);
  for (const auto& e : shape_tup_str) {
    shape.push_back(static_cast<ndarray_len_t>(std::stoul(e)));
  }

  RAFT_EXPECTS(
    descr.byteorder == RAFT_NUMPY_HOST_ENDIAN_CHAR || descr.byteorder == RAFT_NUMPY_NO_ENDIAN_CHAR,
    "The mdspan was serialized on a %s machine but you're attempting to load it on "
    "a %s machine. This use case is not currently supported.",
    (RAFT_SYSTEM_LITTLE_ENDIAN ? "big-endian" : "little-endian"),
    (RAFT_SYSTEM_LITTLE_ENDIAN ? "little-endian" : "big-endian"));

  return {descr, fortran_order, shape};
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void serialize_host_mdspan(
  std::ostream& os,
  const raft::host_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  static_assert(std::is_same_v<LayoutPolicy, raft::layout_c_contiguous> ||
                  std::is_same_v<LayoutPolicy, raft::layout_f_contiguous>,
                "The serializer only supports row-major and column-major layouts");

  using obj_t = raft::host_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

  const auto dtype         = get_numpy_dtype<ElementType>();
  const bool fortran_order = std::is_same_v<LayoutPolicy, raft::layout_f_contiguous>;
  std::vector<ndarray_len_t> shape;
  for (typename obj_t::rank_type i = 0; i < obj.rank(); ++i) {
    shape.push_back(obj.extent(i));
  }
  const header_t header = {dtype, fortran_order, shape};
  write_header(os, header);

  // For contiguous layouts, size() == product of dimensions
  os.write(reinterpret_cast<const char*>(obj.data_handle()), obj.size() * sizeof(ElementType));
  RAFT_EXPECTS(os.good(), "Error writing content of mdspan");
}

template <typename T>
inline void serialize_scalar(std::ostream& os, const T& value)
{
  const auto dtype         = get_numpy_dtype<T>();
  const bool fortran_order = false;
  const std::vector<ndarray_len_t> shape{};
  const header_t header = {dtype, fortran_order, shape};
  write_header(os, header);
  os.write(reinterpret_cast<const char*>(&value), sizeof(T));
  RAFT_EXPECTS(os.good(), "Error serializing a scalar");
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void deserialize_host_mdspan(
  std::istream& is,
  const raft::host_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  static_assert(std::is_same_v<LayoutPolicy, raft::layout_c_contiguous> ||
                  std::is_same_v<LayoutPolicy, raft::layout_f_contiguous>,
                "The serializer only supports row-major and column-major layouts");

  using obj_t = raft::host_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

  // Check if given dtype and fortran_order are compatible with the mdspan
  const auto expected_dtype         = get_numpy_dtype<ElementType>();
  const bool expected_fortran_order = std::is_same_v<LayoutPolicy, raft::layout_f_contiguous>;
  header_t header                   = read_header(is);
  RAFT_EXPECTS(header.dtype == expected_dtype,
               "Expected dtype %s but got %s instead",
               header.dtype.to_string().c_str(),
               expected_dtype.to_string().c_str());
  RAFT_EXPECTS(header.fortran_order == expected_fortran_order,
               "Wrong matrix layout; expected %s but got a different layout",
               (expected_fortran_order ? "Fortran layout" : "C layout"));

  // Check if dimensions are correct
  RAFT_EXPECTS(obj.rank() == header.shape.size(),
               "Incorrect rank: expected %zu but got %zu",
               obj.rank(),
               header.shape.size());
  for (typename obj_t::rank_type i = 0; i < obj.rank(); ++i) {
    RAFT_EXPECTS(static_cast<ndarray_len_t>(obj.extent(i)) == header.shape[i],
                 "Incorrect dimension: expected %zu but got %zu",
                 static_cast<ndarray_len_t>(obj.extent(i)),
                 header.shape[i]);
  }

  // For contiguous layouts, size() == product of dimensions
  is.read(reinterpret_cast<char*>(obj.data_handle()), obj.size() * sizeof(ElementType));
  RAFT_EXPECTS(is.good(), "Error while reading mdspan content");
}

template <typename T>
inline T deserialize_scalar(std::istream& is)
{
  // Check if dtype is correct
  const auto expected_dtype = get_numpy_dtype<T>();
  header_t header           = read_header(is);
  RAFT_EXPECTS(header.dtype == expected_dtype,
               "Expected dtype %s but got %s instead",
               header.dtype.to_string().c_str(),
               expected_dtype.to_string().c_str());
  // Check if dimensions are correct; shape should be ()
  RAFT_EXPECTS(header.shape.empty(), "Incorrect rank: expected 0 but got %zu", header.shape.size());

  T value;
  is.read(reinterpret_cast<char*>(&value), sizeof(T));
  RAFT_EXPECTS(is.good(), "Error while deserializing scalar");
  return value;
}

}  // end namespace numpy_serializer
}  // end namespace detail
}  // end namespace raft
