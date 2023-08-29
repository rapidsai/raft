#pragma once
#include <raft/core/cuda_support.hpp>
#include <raft/core/detail/mdspan_copy.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <type_traits>
namespace raft {

template <typename DstType, typename SrcType>
std::enable_if_t<
  std::conjunction_v<
    std::bool_constant<is_mdspan_v<std::remove_reference_t<DstType>, SrcType>>,
    std::bool_constant<!detail::mdspan_copy_requires_custom_kernel_v<std::remove_reference_t<DstType>, SrcType>>,
    std::is_convertible<typename SrcType::value_type, typename std::remove_reference_t<DstType>::element_type>,
    std::bool_constant<std::remove_reference_t<DstType>::extents_type::rank() == SrcType::extents_type::rank()>
  >
> copy(resources const& res, DstType&& dst, SrcType const& src) {
  detail::copy(res, dst, src);
}

}  // namespace raft
