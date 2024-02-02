/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "detail/cublaslt_wrappers.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/input_validation.hpp>

namespace raft::linalg {

/**
 * @defgroup gemm Matrix-Matrix Multiplication
 * @{
 */

/**
 * @brief GEMM function designed for handling all possible
 * combinations of operand layouts (raft::row_major or raft::col_major)
 * with scalars alpha and beta on the host or device
 * It computes the following equation: Z = alpha . X * Y + beta . Z
 * If alpha is not provided, it is assumed to be 1.0
 * If beta is not provided, it is assumed to be 0.0
 * @tparam ValueType Data type of input/output matrices (float/double)
 * @tparam IndexType Type of index
 * @tparam LayoutPolicyX layout of X
 * @tparam LayoutPolicyY layout of Y
 * @tparam LayoutPolicyZ layout of Z
 * @param[in] res raft handle
 * @param[in] x input raft::device_matrix_view of size M rows x K columns
 * @param[in] y input raft::device_matrix_view of size K rows x N columns
 * @param[out] z output raft::device_matrix_view of size M rows x N columns
 * @param[in] alpha optional raft::host_scalar_view or raft::device_scalar_view, default 1.0
 * @param[in] beta optional raft::host_scalar_view or raft::device_scalar_view, default 0.0
 */
template <typename ValueType,
          typename IndexType,
          typename LayoutPolicyX,
          typename LayoutPolicyY,
          typename LayoutPolicyZ,
          typename ScalarIdxType  = std::uint32_t,
          typename ScalarViewType = raft::host_scalar_view<ValueType, ScalarIdxType>,
          typename                = std::enable_if_t<std::disjunction_v<
            std::is_same<ScalarViewType, raft::host_scalar_view<ValueType, ScalarIdxType>>,
            std::is_same<ScalarViewType, raft::device_scalar_view<ValueType, ScalarIdxType>>>>>
void gemm(raft::resources const& res,
          raft::device_matrix_view<ValueType, IndexType, LayoutPolicyX> x,
          raft::device_matrix_view<ValueType, IndexType, LayoutPolicyY> y,
          raft::device_matrix_view<ValueType, IndexType, LayoutPolicyZ> z,
          std::optional<ScalarViewType> alpha = std::nullopt,
          std::optional<ScalarViewType> beta  = std::nullopt)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(x), "X is not contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(y), "Y is not contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(z), "Z is not contiguous");

  RAFT_EXPECTS(x.extent(0) == z.extent(0), "Number of rows of X and Z should be equal");
  RAFT_EXPECTS(y.extent(1) == z.extent(1), "Number of columns of Y and Z should be equal");
  RAFT_EXPECTS(x.extent(1) == y.extent(0), "Number of columns of X and rows of Y should be equal");

  constexpr auto kXColMajor = std::is_same_v<typename decltype(x)::layout_type, raft::col_major>;
  constexpr auto kYColMajor = std::is_same_v<typename decltype(y)::layout_type, raft::col_major>;
  constexpr auto kZColMajor = std::is_same_v<typename decltype(z)::layout_type, raft::col_major>;

  constexpr auto kDeviceMode =
    std::is_same_v<ScalarViewType, raft::device_scalar_view<ValueType, ScalarIdxType>>;

  ValueType alpha_value = 1;
  ValueType beta_value  = 0;

  auto alpha_device = raft::make_device_scalar(res, alpha_value);
  auto beta_device  = raft::make_device_scalar(res, beta_value);

  auto alpha_host = raft::make_host_scalar(alpha_value);
  auto beta_host  = raft::make_host_scalar(beta_value);

  if constexpr (kDeviceMode) {
    if (!alpha) { alpha = alpha_device.view(); }
    if (!beta) { beta = beta_device.view(); }
  } else {
    if (!alpha) { alpha = alpha_host.view(); }
    if (!beta) { beta = beta_host.view(); }
  }
  if constexpr (kZColMajor) {
    return detail::matmul<kDeviceMode, ValueType, ValueType, ValueType, ValueType>(
      res,
      !kXColMajor,
      !kYColMajor,
      static_cast<uint64_t>(z.extent(0)),
      static_cast<uint64_t>(z.extent(1)),
      static_cast<uint64_t>(x.extent(1)),
      alpha.value().data_handle(),
      x.data_handle(),
      static_cast<uint64_t>(x.extent(kXColMajor ? 0 : 1)),
      y.data_handle(),
      static_cast<uint64_t>(y.extent(kYColMajor ? 0 : 1)),
      beta.value().data_handle(),
      z.data_handle(),
      static_cast<uint64_t>(z.extent(0)));
  } else {
    return detail::matmul<kDeviceMode, ValueType, ValueType, ValueType, ValueType>(
      res,
      kYColMajor,
      kXColMajor,
      static_cast<uint64_t>(z.extent(1)),
      static_cast<uint64_t>(z.extent(0)),
      static_cast<uint64_t>(x.extent(1)),
      alpha.value().data_handle(),
      y.data_handle(),
      static_cast<uint64_t>(y.extent(kYColMajor ? 0 : 1)),
      x.data_handle(),
      static_cast<uint64_t>(x.extent(kXColMajor ? 0 : 1)),
      beta.value().data_handle(),
      z.data_handle(),
      static_cast<uint64_t>(z.extent(1)));
  }
}

/** @} */  // end of gemm

}  // namespace raft::linalg
