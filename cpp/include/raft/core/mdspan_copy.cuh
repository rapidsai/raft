#pragma once
#include <raft/core/cuda_support.hpp>
#include <raft/core/detail/mdspan_copy.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_copy.hpp>
#include <type_traits>
namespace raft {
template <typename DstType, typename SrcType>
std::enable_if_t<
  detail::mdspan_copyable<true, DstType, SrcType>::custom_kernel_allowed,
  detail::mdspan_copyable_t<DstType, SrcType>
> copy(resources const& res, DstType&& dst, SrcType const& src) {
  detail::copy(res, dst, src);
}

}  // namespace raft

