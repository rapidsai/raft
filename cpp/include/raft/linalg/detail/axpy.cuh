/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cublas_wrappers.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resources.hpp>

#include <cublas_v2.h>

namespace RAFT_EXPORT raft {
namespace linalg::detail {

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
  auto cublas_h = resource::get_cublas_handle(handle);
  cublas_device_pointer_mode<DevicePointerMode> pmode(cublas_h);
  RAFT_CUBLAS_TRY(cublasaxpy(cublas_h, n, alpha, x, incx, y, incy, stream));
}

}  // namespace linalg::detail
}  // namespace RAFT_EXPORT raft
