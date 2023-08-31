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
#include <execution>
#include <raft/core/cuda_support.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/stream_view.hpp>
#include <raft/core/resources.hpp>
#include <type_traits>
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/device_mdarray.hpp>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
  #ifdef __CUDACC__
#include <raft/core/mdspan_copy.hpp>
  #endif
#endif

namespace raft {
namespace detail {

template <bool B, typename DstType=void, typename SrcType=void, typename T=void>
struct mdspan_copyable{};

template <typename DstType, typename SrcType, typename T>
struct mdspan_copyable<true, DstType, SrcType, T> {
  using dst_type = std::remove_reference_t<DstType>;
  using src_type = std::remove_reference_t<SrcType>;

  // Dtype properties
  using dst_value_type = typename dst_type::value_type;
  using src_value_type = typename src_type::value_type;
  using dst_element_type = typename dst_type::element_type;
  using src_element_type = typename src_type::element_type;
  auto static constexpr const same_dtype = std::is_same_v<dst_value_type, src_value_type>;
  auto static constexpr const compatible_dtype = std::is_convertible_v<src_value_type, dst_element_type>;

  auto static constexpr const dst_float = std::is_same_v<dst_value_type, float>;
  auto static constexpr const src_float = std::is_same_v<src_value_type, float>;
  auto static constexpr const dst_double = std::is_same_v<dst_value_type, double>;
  auto static constexpr const src_double = std::is_same_v<src_value_type, double>;

  auto static constexpr const both_float = dst_float && src_float;
  auto static constexpr const both_double = dst_double && src_double;
  auto static constexpr const both_float_or_both_double = both_float || both_double;

  // Ranks
  auto static constexpr const dst_rank = dst_type::extents_type::rank();
  auto static constexpr const src_rank = src_type::extents_type::rank();
  auto static constexpr const compatible_rank = (dst_rank == src_rank);
  auto static constexpr const vector_rank = (dst_rank == 1);
  auto static constexpr const matrix_rank = (dst_rank == 2);

  // Layout properties
  using dst_layout_type = typename dst_type::layout_type;
  using src_layout_type = typename src_type::layout_type;

  auto static constexpr const same_layout = std::is_same_v<dst_layout_type, src_layout_type>;

  auto static constexpr const src_contiguous = std::disjunction_v<
    std::is_same_v<src_layout_type, layout_c_contiguous>,
    std::is_same_v<src_layout_type, layout_f_contiguous>
  >;

  auto static constexpr const dst_contiguous = std::disjunction_v<
    std::is_same_v<dst_layout_type, layout_c_contiguous>,
    std::is_same_v<dst_layout_type, layout_f_contiguous>
  >;

  auto static constexpr const both_contiguous = src_contiguous && dst_contiguous;

  auto static constexpr const same_underlying_layout = std::disjunction_v<
    std::bool_constant<same_layout>,
    std::bool_constant<vector_rank && both_contiguous>
  >;


  // Accessibility
  auto static constexpr const dst_device_accessible = is_device_mdspan_v<dst_type>;
  auto static constexpr const src_device_accessible = is_device_mdspan_v<src_type>;
  auto static constexpr const both_device_accessible = dst_device_accessible && src_device_accessible;

  auto static constexpr const dst_host_accessible = is_host_mdspan_v<dst_type>;
  auto static constexpr const src_host_accessible = is_host_mdspan_v<src_type>;
  auto static constexpr const both_host_accessible = dst_host_accessible && src_host_accessible;

  // Allowed copy codepaths
  auto static constexpr const can_use_device = std::conjunction_v<CUDA_ENABLED, both_device_accessible>;

  auto static constexpr const can_use_host = both_host_accessible;

#if (defined(__AVX__) || defined(__SSE__) || defined(__ARM_NEON))
  auto static constexpr const can_use_simd = both_host_accessible && both_contiguous;
# else
  auto static constexpr const can_use_simd = false;
#endif

  auto static constexpr const can_use_std_copy = std::conjunction_v<
    std::bool_constant<can_use_host>,
    std::bool_constant<compatible_dtype>,
    std::bool_constant<both_contiguous>,
    std::bool_constant<same_underlying_layout>
  >;
  auto static constexpr const can_use_raft_copy = std::conjunction_v<
    std::bool_constant<CUDA_ENABLED>,
    std::bool_constant<same_dtype>,
    std::bool_constant<both_contiguous>,
    std::bool_constant<same_underlying_layout>
  >;
  auto static constexpr const can_use_cublas = std::conjunction_v<
    std::bool_constant<can_use_device>,
    std::bool_constant<compatible_dtype>,
    std::bool_constant<both_contiguous>,
    std::bool_constant<!same_underlying_layout>,
    std::bool_constant<matrix_rank>,
    std::bool_constant<both_float_or_both_double>
  >;

  auto static constexpr const requires_intermediate = !both_host_accessible && !both_device_accessible && !can_use_raft_copy;

  auto static constexpr const use_intermediate_dst = std::conjunction_v<
    std::bool_constant<requires_intermediate>,
    std::bool_constant<src_device_accessible>
  >;

  auto static constexpr const use_intermediate_src = std::conjunction_v<
    std::bool_constant<requires_intermediate>,
    std::bool_constant<!use_intermediate_dst>
  >;

  auto static constexpr const custom_kernel_allowed = std::conjunction_v<
    std::bool_constant<can_use_device>,
    std::bool_constant<!requires_intermediate>,
    std::bool_constant<
      !(can_use_raft_copy || can_use_cublas)
    >
  >;

  auto static constexpr const custom_kernel_required = std::conjunction_v<
    std::bool_constant<!can_use_host>,
    std::bool_constant<!requires_intermediate>,
    std::bool_constant<
      !(can_use_raft_copy || can_use_cublas)
    >
  >;

  // Viable overload?
  // TODO(wphicks): Detect case where custom kernel would be required AFTER
  // transfer only
  auto static constexpr const value = std::conjunction_v<
    is_mdspan<dst_type>,
    is_mdspan<src_type>,
#ifndef __CUDACC__
    std::bool_constant<!custom_kernel_required>,
#endif
    std::bool_constant<compatible_dtype>,
    std::bool_constant<compatible_rank>
  >;
  using type = std::enable_if_t<value, T>;
};

template <typename DstType, typename SrcType, typename T=void>
using mdspan_copyable_t = typename mdspan_copyable<true, DstType, SrcType, T>::type;
template <typename DstType, typename SrcType>
using mdspan_copyable_v = typename mdspan_copyable<true, DstType, SrcType, void>::value;

template <typename DstType, typename SrcType>
mdspan_copyable_t<DstType, SrcType>
copy(resources const& res, DstType&& dst, SrcType const& src)
{
  using config = mdspan_copyable<true, DstType, SrcType>;
  for (auto i = std::size_t{}; i < SrcType::extents_type::rank(); ++i) {
    RAFT_EXPECTS(src.extents(i) == dst.extents(i), "Must copy between mdspans of the same shape");
  }

  if constexpr(config::use_intermediate_src) {
    // Copy to intermediate source on device, then perform necessary
    // changes in layout on device, directly into final destination
    auto intermediate = device_mdarray<
      typename config::src_value_type,
      typename config::src_extents_type,
      typename config::src_layout_type
    >(res, src.extents());
    copy(res, intermediate.view(), src);
    copy(res, dst, intermediate.view());

  } else if constexpr(config::use_intermediate_dst) {
    // Perform necessary changes in layout on device, then copy to final
    // destination on host
    auto intermediate = device_mdarray<
      typename config::dst_value_type,
      typename config::dst_extents_type,
      typename config::dst_layout_type
    >(res, dst.extents());
    copy(res, intermediate.view(), src);
    copy(res, dst, intermediate.view());
  } else if constexpr(config::can_use_raft_copy) {
#ifndef RAFT_DISABLE_CUDA
    raft::copy(
      dst.data_handle(),
      src.data_handle(),
      dst.size(),
      resource::get_cuda_stream(res)
    );
#endif
  } else if constexpr(config::can_use_cublas) {
    auto constexpr const alpha = typename std::remove_reference_t<DstType>::value_type{1};
    auto constexpr const beta  = typename std::remove_reference_t<DstType>::value_type{0};
    CUBLAS_TRY(cublasgeam(resource::get_cublas_handle(res),
                          CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          dst.extent(0),
                          dst.extent(1),
                          &alpha,
                          src.data_handle(),
                          src.stride(0),
                          &beta,
                          static_cast<typename std::remove_reference_t<DstType>::value_type*>(nullptr),
                          dst.stride(0),
                          dst.data_handle(),
                          dst.stride(0),
                          resource::get_cuda_stream(res)));
  } else if constexpr(config::can_use_std_copy) {
    std::copy(src.data_handle(), src.data_handle() + dst.size(), dst.data_handle());
  } else if constexpr(config::can_use_simd) {
  } else {
      auto indices = std::array<index_type, std::remove_reference_t<DstType>::extents_type::rank()>{};
      for (auto i = std::size_t{}; i < dst.size(); ++i) {
        if constexpr (std::is_same_v<typename std::remove_reference_t<DstType>::layout_type, layout_c_contiguous>) {
          // For layout_right/layout_c_contiguous, we iterate over the
          // rightmost extent fastest
          auto dim = std::remove_reference_t<DstType>::extents_type::rank();
          while ((indices[dim]++) == dst.extent(dim)) {
            indices[dim] = index_type{};
            --dim;
          }
        } else {
          // For layout_left/layout_f_contiguous (and currently all other
          // layouts), we iterate over the leftmost extent fastest

          // TODO(wphicks): Add additional specialization for non-C/F
          // arrays that have a stride of 1 in one dimension. This would
          // be a performance enhancement; it is not required for
          // correctness.
          auto dim = std::size_t{};
          while ((indices[dim]++) == dst.extent(dim)) {
            indices[dim] = index_type{};
            ++dim;
          }
        }
        std::apply(dst, indices) = std::apply(src, indices);
      }
  }

  if constexpr (config::can_use_device) {
#ifndef RAFT_DISABLE_CUDA
    if constexpr (same_dtype && (same_layout || std::remove_reference_t<DstType>::extents_type::rank() == 1) && both_contiguous) {
      raft::copy(
        dst.data_handle(),
        src.data_handle(),
        dst.size(),
        resource::get_cuda_stream(res)
      );
    } else if constexpr (same_dtype && both_float_or_double && both_contiguous &&
                         std::remove_reference_t<DstType>::extents_type::rank() == 2) {
      auto constexpr const alpha = typename std::remove_reference_t<DstType>::value_type{1};
      auto constexpr const beta  = typename std::remove_reference_t<DstType>::value_type{0};
      CUBLAS_TRY(cublasgeam(resource::get_cublas_handle(res),
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            dst.extent(0),
                            dst.extent(1),
                            &alpha,
                            src.data_handle(),
                            src.stride(0),
                            &beta,
                            static_cast<typename std::remove_reference_t<DstType>::value_type*>(nullptr),
                            dst.stride(0),
                            dst.data_handle(),
                            dst.stride(0),
                            resource::get_cuda_stream(res)));
    } else {
#ifdef __CUDACC__
      // TODO(wphicks): Call kernel here
#else
      // Ordinarily, we would just make this a .cuh file, but we do not want
      // to signal that it *must* be built with CUDA. Instead, if this header
      // is used in a way that requires a CUDA compiler, we fail with an
      // informative error message.
      static_assert(
        !mdspan_copy_requires_custom_kernel_v<std::remove_reference_t<DstType>, SrcType>,
        "Selected instantiation of raft::copy requires nvcc compilation. Use raft/core/mdspan_copy.cuh instead of raft/core/mdspan_copy.hpp and #include it in a .cu file. The corresponding 'detail' headers should not be included anywhere else directly."
      );
#endif
    }
#endif
  } else if constexpr (both_host_accessible) {
    if constexpr ((same_layout || std::remove_reference_t<DstType>::extents_type::rank() == 1) && both_contiguous) {
      // Use STL if possible; this should be well optimized
      std::copy(src.data_handle(), src.data_handle() + dst.size(), dst.data_handle());
    } else {
      // TODO (wphicks): Use SIMD for both_contiguous &&
      // both_float_or_double

      // Finally, copy elements one by one, trying at least to perform
      // cache-friendly reads

      auto indices = std::array<index_type, std::remove_reference_t<DstType>::extents_type::rank()>{};
      for (auto i = std::size_t{}; i < dst.size(); ++i) {
        if constexpr (std::is_same_v<typename std::remove_reference_t<DstType>::layout_type, layout_c_contiguous>) {
          // For layout_right/layout_c_contiguous, we iterate over the
          // rightmost extent fastest
          auto dim = std::remove_reference_t<DstType>::extents_type::rank();
          while ((indices[dim]++) == dst.extent(dim)) {
            indices[dim] = index_type{};
            --dim;
          }
        } else {
          // For layout_left/layout_f_contiguous (and currently all other
          // layouts), we iterate over the leftmost extent fastest

          // TODO(wphicks): Add additional specialization for non-C/F
          // arrays that have a stride of 1 in one dimension. This would
          // be a performance enhancement; it is not required for
          // correctness.
          auto dim = std::size_t{};
          while ((indices[dim]++) == dst.extent(dim)) {
            indices[dim] = index_type{};
            ++dim;
          }
        }
        std::apply(dst, indices) = std::apply(src, indices);
      }
    }
  } else {
#ifndef RAFT_DISABLE_CUDA
    if constexpr (same_dtype && same_layout && both_contiguous) {
      raft::copy(dst.data_handle(), src.data_handle(), dst.size());
    } else if constexpr (is_device_mdspan_v<std::remove_reference_t<DstType>>) {
      // Copy to device memory and call recursively
    } else {
      // Copy to host memory and call recursively
    }
#else
    RAFT_FAIL("mdspan copy required device access in non-CUDA build");
#endif
  }
}
}  // namespace detail
}  // namespace raft
