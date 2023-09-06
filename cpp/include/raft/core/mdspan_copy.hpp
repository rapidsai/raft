#pragma once
#include <raft/core/detail/mdspan_copy.hpp>
#include <type_traits>
namespace raft {

template <typename DstType, typename SrcType>
detail::mdspan_uncopyable_with_kernel_t<DstType, SrcType>
copy(resources const& res, DstType&& dst, SrcType const& src) {
  detail::copy(res, dst, src);
}

}  // namespace raft
