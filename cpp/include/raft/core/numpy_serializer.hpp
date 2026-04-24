/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/mdspan_numpy_serializer.hpp>

namespace raft::numpy_serializer {

/**
 * @defgroup numpy_serializer NumPy serialization helpers
 * @{
 */

/**
 * @brief Integer type used for NumPy array shape extents.
 */
using ndarray_len_t = detail::numpy_serializer::ndarray_len_t;

/**
 * @brief NumPy dtype descriptor.
 */
using dtype_t = detail::numpy_serializer::dtype_t;

/**
 * @brief Parsed NumPy header metadata.
 */
using header_t = detail::numpy_serializer::header_t;

/**
 * @brief Return the NumPy dtype descriptor corresponding to a C++ element type.
 *
 * @tparam T C++ element type.
 * @return NumPy dtype descriptor for T.
 */
template <typename T>
inline dtype_t get_numpy_dtype()
{
  return detail::numpy_serializer::get_numpy_dtype<T>();
}

/**
 * @brief Write a NumPy `.npy` header to an output stream.
 *
 * @param os Output stream.
 * @param header Header metadata to write.
 */
inline void write_header(std::ostream& os, const header_t& header)
{
  detail::numpy_serializer::write_header(os, header);
}

/**
 * @brief Read and parse a NumPy `.npy` header from an input stream.
 *
 * The stream is left positioned immediately after the header.
 *
 * @param is Input stream.
 * @return Parsed NumPy header metadata.
 */
inline header_t read_header(std::istream& is) { return detail::numpy_serializer::read_header(is); }

/** @} */

}  // namespace raft::numpy_serializer
