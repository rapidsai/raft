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

#include "detail/interop.hpp"

namespace raft::core {

/**
 * @defgroup interop Interoperability between `mdspan` and `DLManagedTensor`
 * @{
 */

/**
 * @brief Check if DLTensor has device accessible memory.
 *        This function returns true for `DLDeviceType` of values
 *        `kDLCUDA`, `kDLCUDAHost`, or `kDLCUDAManaged`
 *
 * @param[in] tensor DLTensor object to check underlying memory type
 * @return bool
 */
bool is_dlpack_device_compatible(DLTensor tensor)
{
  return detail::is_dlpack_device_compatible(tensor);
}

/**
 * @brief Check if DLTensor has host accessible memory.
 *        This function returns true for `DLDeviceType` of values
 *        `kDLCPU`, `kDLCUDAHost`, or `kDLCUDAManaged`
 *
 * @param tensor DLTensor object to check underlying memory type
 * @return bool
 */
bool is_dlpack_host_compatible(DLTensor tensor)
{
  return detail::is_dlpack_host_compatible(tensor);
}

/**
 * @brief Convert a DLManagedTensor to an mdspan
 * NOTE: This function only supports compact row-major layouts.
 *
 * @code {.cpp}
 * #include <raft/core/device_mdspan.hpp>
 * #include <raft/core/interop.hpp>
 * // We have a `DLManagedTensor` with `DLDeviceType == kDLCUDA`,
 * // `DLDataType.code == kDLFloat` and `DLDataType.bits == 8`
 * DLManagedTensor tensor;
 * // declare the return type
 * using mdpsan_type = raft::device_mdspan<float, int64_t, raft::row_major>;
 * auto mds = raft::core::from_dlpack<mdspan_type>(&tensor);
 * @endcode
 *
 * @tparam MdspanType
 * @tparam typename
 * @param[in] managed_tensor
 * @return MdspanType
 */
template <typename MdspanType, typename = raft::is_mdspan_t<MdspanType>>
MdspanType from_dlpack(DLManagedTensor* managed_tensor)
{
  return detail::from_dlpack<MdspanType>(managed_tensor);
}

/**
 * @}
 */

}  // namespace raft::core
