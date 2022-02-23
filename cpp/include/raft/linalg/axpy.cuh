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

}  // namespace raft::linalg
