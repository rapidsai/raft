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

#include <raft/core/cuda_support.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/resource/stream_view.hpp>
#include <execution>
#include <type_traits>
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/cudart_utils.hpp>
#endif

namespace raft {
namespace detail {
template <
  typename DstElementType,
  typename DstExtents,
  typename DstLayoutPolicy,
  typename DstAccessorPolicy,
  typename SrcElementType,
  typename SrcExtents,
  typename SrcLayoutPolicy,
  typename SrcAccessorPolicy,
  typename ExecutionPolicy,
  std::enable_if_t<std::conjunction_v<
    std::is_convertible_v<SrcElementType, DstElementType>,
    SrcExtents::rank() == DstExtents::rank()
  >>* = nullptr
>
void copy(
    resources const& res,
    mdspan<DstElementType, DstExtents, DstLayoutPolicy, DstAccessorPolicy> & dst,
    mdspan<SrcElementType, SrcExtents, SrcLayoutPolicy, SrcAccessorPolicy> const& src,
    ExecutionPolicy host_exec_policy = std::execution::unseq
) {
  // TODO(Check size match?)
  if constexpr (
    // Contiguous memory, no transpose required
    std::conjunction_v<
      std::is_same_v<DstLayoutPolicy, SrcLayoutPolicy>,
      std::disjunction_v<
        std::is_same_v<DstLayoutPolicy, layout_c_contiguous>,
        std::is_same_v<DstLayoutPolicy, layout_f_contiguous>
      >
    >
  ) {
    if constexpr (
      std::disjunction_v<
        std::conjunction_v<
          CUDA_ENABLED,
          ! DstAccessorPolicy::mem_type::is_device_accessible,
          ! SrcAccessorPolicy::mem_type::is_device_accessible
        >,
        std::conjunction_v<
          ! CUDA_ENABLED,
          DstAccessorPolicy::mem_type::is_host_accessible,
          SrcAccessorPolicy::mem_type::is_host_accessible
        >
      >
    ) {
      std::copy(
        host_exec_policy,
        src.data_handle(),
        src.data_handle() + src.size(),
        dst.data_handle()
      );
    } else {
#ifndef RAFT_DISABLE_CUDA
      if constexpr(std::is_same_v<DstElementType, std::remove_const_t<SrcElementType>>) {
        raft::copy(
          dst.data_handle(),
          src.data_handle(),
          src.size(),
          get_stream_view(res)
        );
      } else {
        // TODO(wphicks): Convert type on src device and then copy
      }
#else
      throw non_cuda_build_error{
        "Attempted copy to/from device in non-CUDA build"
      };
#endif
    }
  } else { // Non-contiguous memory or transpose required
    if constexpr (
      std::conjunction_v<
        DstAccessorPolicy::mem_type::is_device_accessible,
        SrcAccessorPolicy::mem_type::is_device_accessible
      >
    ) {
      // TODO(wphicks): Conversion/transpose kernel
    } else if constexpr (
      std::conjunction_v<
        DstAccessorPolicy::mem_type::is_host_accessible,
        SrcAccessorPolicy::mem_type::is_host_accessible
      >
    ) {
      // TODO(wphicks): CPU conversion
    } else {
      // TODO(wphicks): Copy to intermediate mdarray on dest device, then call
      // recursively for transpose/conversion
    }
  }
}
}  // namespace detail
}  // namespace raft
