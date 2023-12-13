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
#include <raft/core/detail/copy.hpp>
namespace raft {
/**
 * @brief Copy data from one mdspan to another with the same extents
 *
 * This function copies data from one mdspan to another, regardless of whether
 * or not the mdspans have the same layout, memory type (host/device/managed)
 * or data type. So long as it is possible to convert the data type from source
 * to destination, and the extents are equal, this function should be able to
 * perform the copy. Any necessary device operations will be stream-ordered via the CUDA stream
 * provided by the `raft::resources` argument.
 *
 * This header includes a custom kernel used for copying data between
 * completely arbitrary mdspans on device. To compile this function in a
 * non-CUDA translation unit, `raft/core/copy.hpp` may be used instead. The
 * pure C++ header will correctly compile even without a CUDA compiler.
 * Depending on the specialization, this CUDA header may invoke the kernel and
 * therefore require a CUDA compiler.
 *
 * Limitations: Currently this function does not support copying directly
 * between two arbitrary mdspans on different CUDA devices. It is assumed that the caller sets the
 * correct CUDA device. Furthermore, host-to-host copies that require a transformation of the
 * underlying memory layout are currently not performant, although they are supported.
 *
 * Note that when copying to an mdspan with a non-unique layout (i.e. the same
 * underlying memory is addressed by different element indexes), the source
 * data must contain non-unique values for every non-unique destination
 * element. If this is not the case, the behavior is undefined. Some copies
 * to non-unique layouts which are well-defined will nevertheless fail with an
 * exception to avoid race conditions in the underlying copy.
 *
 * @tparam DstType An mdspan type for the destination container.
 * @tparam SrcType An mdspan type for the source container
 * @param res raft::resources used to provide a stream for copies involving the
 * device.
 * @param dst The destination mdspan.
 * @param src The source mdspan.
 */
template <typename DstType, typename SrcType>
detail::mdspan_copyable_with_kernel_t<DstType, SrcType> copy(resources const& res,
                                                             DstType&& dst,
                                                             SrcType&& src)
{
  detail::copy(res, std::forward<DstType>(dst), std::forward<SrcType>(src));
}

#ifndef RAFT_NON_CUDA_COPY_IMPLEMENTED
#define RAFT_NON_CUDA_COPY_IMPLEMENTED
template <typename DstType, typename SrcType>
detail::mdspan_copyable_not_with_kernel_t<DstType, SrcType> copy(resources const& res,
                                                                 DstType&& dst,
                                                                 SrcType&& src)
{
  detail::copy(res, std::forward<DstType>(dst), std::forward<SrcType>(src));
}
#endif
}  // namespace raft
