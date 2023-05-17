/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/linalg_types.hpp>
#include <raft/matrix/detail/math.cuh>
#include <raft/util/input_validation.hpp>

namespace raft::linalg {

/**
 * @defgroup matrix_vector Matrix-Vector Operations
 * @{
 */

/**
 * @brief multiply each row or column of matrix with vector, skipping zeros in vector
 * @param [in] handle: raft handle for managing library resources
 * @param[inout] data: input matrix, results are in-place
 * @param[in] vec: input vector
 * @param[in] apply whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns using enum class raft::linalg::Apply
 */
template <typename math_t, typename idx_t, typename layout_t>
void binary_mult_skip_zero(raft::resources const& handle,
                           raft::device_matrix_view<math_t, idx_t, layout_t> data,
                           raft::device_vector_view<const math_t, idx_t> vec,
                           Apply apply)
{
  bool row_major        = raft::is_row_major(data);
  auto bcast_along_rows = apply == Apply::ALONG_ROWS;

  idx_t vec_size = bcast_along_rows ? data.extent(1) : data.extent(0);

  RAFT_EXPECTS(
    vec.extent(0) == vec_size,
    "If `bcast_along_rows==true`, vector size must equal number of columns in the matrix."
    "If `bcast_along_rows==false`, vector size must equal number of rows in the matrix.");

  matrix::detail::matrixVectorBinaryMultSkipZero(data.data_handle(),
                                                 vec.data_handle(),
                                                 data.extent(0),
                                                 data.extent(1),
                                                 row_major,
                                                 bcast_along_rows,
                                                 resource::get_cuda_stream(handle));
}

/**
 * @brief divide each row or column of matrix with vector
 * @param[in] handle: raft handle for managing library resources
 * @param[inout] data: input matrix, results are in-place
 * @param[in] vec: input vector
 * @param[in] apply whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns using enum class raft::linalg::Apply
 */
template <typename math_t, typename idx_t, typename layout_t>
void binary_div(raft::resources const& handle,
                raft::device_matrix_view<math_t, idx_t, layout_t> data,
                raft::device_vector_view<const math_t, idx_t> vec,
                Apply apply)
{
  bool row_major        = raft::is_row_major(data);
  auto bcast_along_rows = apply == Apply::ALONG_ROWS;

  idx_t vec_size = bcast_along_rows ? data.extent(1) : data.extent(0);

  RAFT_EXPECTS(
    vec.extent(0) == vec_size,
    "If `bcast_along_rows==true`, vector size must equal number of columns in the matrix."
    "If `bcast_along_rows==false`, vector size must equal number of rows in the matrix.");

  matrix::detail::matrixVectorBinaryDiv(data.data_handle(),
                                        vec.data_handle(),
                                        data.extent(0),
                                        data.extent(1),
                                        row_major,
                                        bcast_along_rows,
                                        resource::get_cuda_stream(handle));
}

/**
 * @brief divide each row or column of matrix with vector, skipping zeros in vector
 * @param[in] handle: raft handle for managing library resources
 * @param[inout] data: input matrix, results are in-place
 * @param[in] vec: input vector
 * @param[in] apply whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns using enum class raft::linalg::Apply
 * @param[in] return_zero: result is zero if true and vector value is below threshold, original
 * value if false
 */
template <typename math_t, typename idx_t, typename layout_t>
void binary_div_skip_zero(raft::resources const& handle,
                          raft::device_matrix_view<math_t, idx_t, layout_t> data,
                          raft::device_vector_view<const math_t, idx_t> vec,
                          Apply apply,
                          bool return_zero = false)
{
  bool row_major        = raft::is_row_major(data);
  auto bcast_along_rows = apply == Apply::ALONG_ROWS;

  idx_t vec_size = bcast_along_rows ? data.extent(1) : data.extent(0);

  RAFT_EXPECTS(
    vec.extent(0) == vec_size,
    "If `bcast_along_rows==true`, vector size must equal number of columns in the matrix."
    "If `bcast_along_rows==false`, vector size must equal number of rows in the matrix.");

  matrix::detail::matrixVectorBinaryDivSkipZero(data.data_handle(),
                                                vec.data_handle(),
                                                data.extent(0),
                                                data.extent(1),
                                                row_major,
                                                bcast_along_rows,
                                                resource::get_cuda_stream(handle),
                                                return_zero);
}

/**
 * @brief add each row or column of matrix with vector
 * @param[in] handle: raft handle for managing library resources
 * @param[inout] data: input matrix, results are in-place
 * @param[in] vec: input vector
 * @param[in] apply whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns using enum class raft::linalg::Apply
 */
template <typename math_t, typename idx_t, typename layout_t>
void binary_add(raft::resources const& handle,
                raft::device_matrix_view<math_t, idx_t, layout_t> data,
                raft::device_vector_view<const math_t, idx_t> vec,
                Apply apply)
{
  bool row_major        = raft::is_row_major(data);
  auto bcast_along_rows = apply == Apply::ALONG_ROWS;

  idx_t vec_size = bcast_along_rows ? data.extent(1) : data.extent(0);

  RAFT_EXPECTS(
    vec.extent(0) == vec_size,
    "If `bcast_along_rows==true`, vector size must equal number of columns in the matrix."
    "If `bcast_along_rows==false`, vector size must equal number of rows in the matrix.");

  matrix::detail::matrixVectorBinaryAdd(data.data_handle(),
                                        vec.data_handle(),
                                        data.extent(0),
                                        data.extent(1),
                                        row_major,
                                        bcast_along_rows,
                                        resource::get_cuda_stream(handle));
}

/**
 * @brief subtract each row or column of matrix with vector
 * @param[in] handle: raft handle for managing library resources
 * @param[inout] data: input matrix, results are in-place
 * @param[in] vec: input vector
 * @param[in] apply whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns using enum class raft::linalg::Apply
 */
template <typename math_t, typename idx_t, typename layout_t>
void binary_sub(raft::resources const& handle,
                raft::device_matrix_view<math_t, idx_t, layout_t> data,
                raft::device_vector_view<const math_t, idx_t> vec,
                Apply apply)
{
  bool row_major        = raft::is_row_major(data);
  auto bcast_along_rows = apply == Apply::ALONG_ROWS;

  idx_t vec_size = bcast_along_rows ? data.extent(1) : data.extent(0);

  RAFT_EXPECTS(
    vec.extent(0) == vec_size,
    "If `bcast_along_rows==true`, vector size must equal number of columns in the matrix."
    "If `bcast_along_rows==false`, vector size must equal number of rows in the matrix.");

  matrix::detail::matrixVectorBinarySub(data.data_handle(),
                                        vec.data_handle(),
                                        data.extent(0),
                                        data.extent(1),
                                        row_major,
                                        bcast_along_rows,
                                        resource::get_cuda_stream(handle));
}

/** @} */  // end of matrix_vector

}  // namespace raft::linalg