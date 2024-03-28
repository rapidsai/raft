/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

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
void axpy(raft::resources const& handle,
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
 * @defgroup axpy axpy routine
 * @{
 */

/**
 * @brief axpy function
 *  It computes the following equation: y = alpha * x + y
 *
 * @param [in] handle raft::resources
 * @param [in] alpha raft::device_scalar_view
 * @param [in] x Input vector
 * @param [inout] y Output vector
 */
template <typename ElementType,
          typename IndexType,
          typename InLayoutPolicy,
          typename OutLayoutPolicy,
          typename ScalarIdxType>
void axpy(raft::resources const& handle,
          raft::device_scalar_view<const ElementType, ScalarIdxType> alpha,
          raft::device_vector_view<const ElementType, IndexType, InLayoutPolicy> x,
          raft::device_vector_view<ElementType, IndexType, OutLayoutPolicy> y)
{
  RAFT_EXPECTS(y.size() == x.size(), "Size mismatch between Output and Input");

  axpy<ElementType, true>(handle,
                          y.size(),
                          alpha.data_handle(),
                          x.data_handle(),
                          x.stride(0),
                          y.data_handle(),
                          y.stride(0),
                          resource::get_cuda_stream(handle));
}

/**
 * @brief axpy function
 *  It computes the following equation: y = alpha * x + y
 * @param [in] handle raft::resources
 * @param [in] alpha raft::device_scalar_view
 * @param [in] x Input vector
 * @param [inout] y Output vector
 */
template <typename ElementType,
          typename IndexType,
          typename InLayoutPolicy,
          typename OutLayoutPolicy,
          typename ScalarIdxType>
void axpy(raft::resources const& handle,
          raft::host_scalar_view<const ElementType, ScalarIdxType> alpha,
          raft::device_vector_view<const ElementType, IndexType, InLayoutPolicy> x,
          raft::device_vector_view<ElementType, IndexType, OutLayoutPolicy> y)
{
  RAFT_EXPECTS(y.size() == x.size(), "Size mismatch between Output and Input");

  axpy<ElementType, false>(handle,
                           y.size(),
                           alpha.data_handle(),
                           x.data_handle(),
                           x.stride(0),
                           y.data_handle(),
                           y.stride(0),
                           resource::get_cuda_stream(handle));
}

/** @} */  // end of group axpy

}  // namespace raft::linalg

#endif
