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
#ifndef __AXPY_H
#define __AXPY_H

#pragma once

#include "detail/axpy.cuh"

namespace raft::linalg {

/**
 * @brief the wrapper of cublas axpy function
 *  It computes the following equation: y = alpha * x + y
 *
 * @tparam T the element type
 * @tparam DevicePointerMode whether pointers alpha, beta point to device memory
 * @param [in] handle raft handle
 * @param [in] n number of elements in x and y
 * @param [in] alpha host or device scalar
 * @param [in] x vector of length n
 * @param [in] incx stride between consecutive elements of x
 * @param [inout] y vector of length n
 * @param [in] incy stride between consecutive elements of y
 * @param [in] stream
 */
template <typename T, bool DevicePointerMode = false>
void axpy(const raft::handle_t& handle,
          const int n,
          const T* alpha,
          const T* x,
          const int incx,
          T* y,
          const int incy,
          cudaStream_t stream)
{
  detail::axpy<T, DevicePointerMode>(handle, n, alpha, x, incx, y, incy, stream);
}

/**
 * @defgroup axpy cuBLAS axpy
 * @{
 */

/**
 * @brief the wrapper of cublas axpy function
 *  It computes the following equation: y = alpha * x + y
 *
 * @tparam MdspanType  Type raft::mdspan
 * @tparam DevicePointerMode whether pointers alpha, beta point to device memory
 * @param [in] handle raft::handle_t
 * @param [in] alpha raft::scalar_view in either host or device memory
 * @param [in] x Input vector
 * @param [in] incx stride between consecutive elements of x
 * @param [inout] y Output vector
 * @param [in] incy stride between consecutive elements of y
 */
template <typename MdspanType,
          bool DevicePointerMode = false,
          typename               = raft::enable_if_mdspan<MdspanType>>
void axpy(const raft::handle_t& handle,
          raft::scalar_view<typename MdspanType::element_type> alpha,
          const MdspanType x,
          const int incx,
          MdspanType y,
          const int incy)
{
  RAFT_EXPECTS(y.size() == x.size(), "Size mismatch between Output and Input")

  axpy<typename MdspanType::element_type, DevicePointerMode>(
    handle, y.size(), alpha, x.data_handle(), incx, y.data_handle(), incy, handle.get_stream());
}

/** @} */  // end of group axpy

}  // namespace raft::linalg

#endif