#pragma once
#include <raft/core/detail/copy.hpp>
namespace raft {

#ifndef RAFT_NON_CUDA_COPY_IMPLEMENTED
#define RAFT_NON_CUDA_COPY_IMPLEMENTED
template <typename DstType, typename SrcType>
detail::mdspan_uncopyable_with_kernel_t<DstType, SrcType> copy(resources const& res,
                                                               DstType&& dst,
                                                               SrcType const& src)
{
  detail::copy(res, dst, src);
}
#endif

}  // namespace raft
