#pragma once
#include <raft/core/detail/mdspan_copy.hpp>
namespace raft {
template <typename DstType, typename SrcType>
detail::mdspan_copyable_with_kernel_t<DstType, SrcType>
copy(resources const& res, DstType&& dst, SrcType const& src) {
  detail::copy(res, dst, src);
}
}  // namespace raft
