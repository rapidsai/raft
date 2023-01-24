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

#pragma once

#include <algorithm>
#include <complex>
#include <cstdint>
#include <ostream>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>
#include <string>
#include <type_traits>
#include <vector>

namespace raft {

namespace detail {

namespace numpy_serializer {

/*
 * A small implementation of NumPy serialization format.
 * Reference: https://numpy.org/doc/1.13/neps/npy-format.html
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

const char little_endian_char = '<';
const char big_endian_char    = '>';
const char no_endian_char     = '|';
const char endian_chars[]     = {little_endian_char, big_endian_char, no_endian_char};
const char numtype_chars[]    = {'f', 'i', 'u', 'c'};

#if RAFT_SYSTEM_LITTLE_ENDIAN == 1
const char host_endian_char = little_endian_char;
#else   // RAFT_SYSTEM_LITTLE_ENDIAN == 1
const char host_endian_char = big_endian_char;
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
};

template <class T>
struct is_complex : std::false_type {
};
template <class T>
struct is_complex<std::complex<T>> : std::true_type {
};

template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
dtype_t get_numpy_dtype()
{
  return {host_endian_char, 'f', sizeof(T)};
}

template <typename T,
          typename std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T>, bool> = true>
dtype_t get_numpy_dtype()
{
  const char endian_char = (sizeof(T) == 1 ? no_endian_char : host_endian_char);
  return {endian_char, 'i', sizeof(T)};
}

template <typename T,
          typename std::enable_if_t<std::is_integral_v<T> && std::is_unsigned_v<T>, bool> = true>
dtype_t get_numpy_dtype()
{
  const char endian_char = (sizeof(T) == 1 ? no_endian_char : host_endian_char);
  return {endian_char, 'u', sizeof(T)};
}

template <typename T, typename std::enable_if_t<is_complex<T>{}, bool> = true>
dtype_t get_numpy_dtype()
{
  return {host_endian_char, 'c', sizeof(T)};
}

template <typename T>
std::string tuple_to_string(const std::vector<T>& tuple)
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

std::string header_to_string(const header_t& header)
{
  std::ostringstream oss;
  oss << "{'descr': '" << header.dtype.to_string()
      << "', 'fortran_order': " << (header.fortran_order ? "True" : "False")
      << ", 'shape': " << tuple_to_string(header.shape) << "}";
  return oss.str();
}

const char magic_string[]             = "\x93NUMPY";
const std::size_t magic_string_length = 6;

void write_magic(std::ostream& os)
{
  os.write(magic_string, magic_string_length);
  RAFT_EXPECTS(os.good(), "Error writing magic string");
  // Use version 1.0
  os.put(1);
  os.put(0);
  RAFT_EXPECTS(os.good(), "Error writing magic string");
}

void write_header(std::ostream& os, const header_t& header)
{
  std::string header_dict     = header_to_string(header);
  std::size_t preamble_length = magic_string_length + 2 + 2 + header_dict.length() + 1;
  RAFT_EXPECTS(preamble_length < 255 * 255, "Header too long");
  std::size_t padding_len = 16 - preamble_length % 16;
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

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void serialize(const raft::handle_t& handle,
               std::ostream& os,
               const raft::host_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  using obj_t               = raft::host_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
  using inner_accessor_type = typename obj_t::accessor_type::accessor_type;
  static_assert(
    std::is_same_v<inner_accessor_type, std::experimental::default_accessor<ElementType>>,
    "The serializer only supports serializing mdspans with default accessor");

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

}  // end namespace numpy_serializer
}  // end namespace detail
}  // end namespace raft
