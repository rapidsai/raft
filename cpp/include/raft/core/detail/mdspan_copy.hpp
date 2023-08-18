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
#include <raft/core/cudart_utils.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#endif

namespace raft {
namespace detail {
template <typename DstType, typename SrcType>
std::enable_if_t<
  std::conjunction_v<is_mdspan_v<DstType>,
                     is_mdspan_v<SrcType>,
                     std::is_convertible_v<SrcType::value_type, DstType::element_type>,
                     DstType::extents::rank() == SrcType::extents::rank()>>
copy(resources const& res, DstType& dst, SrcType const& src)
{
  using index_type =
    std::conditional_t<(std::numeric_limits<typename DstType::extents::index_type>::max() >
                        std::numeric_limits<typename SrcType::extents::index_type>::max()),
                       typename DstType::extents::index_type,
                       typename SrcType::extents::index_type>;
  auto constexpr const both_contiguous = std::conjunction_v<
    std::disjunction_v<std::is_same_v<typename DstType::layout_type, layout_c_contiguous>,
                       std::is_same_v<typename DstType::layout_type, layout_f_contiguous>>,
    std::disjunction_v<std::is_same_v<typename SrcType::layout_type, layout_c_contiguous>,
                       std::is_same_v<typename SrcType::layout_type, layout_f_contiguous>>>;
  auto constexpr const same_dtype = std::is_same_v<DstType::value_type, SrcType::value_type>;
  auto constexpr const both_device_accessible =
    std::conjunction_v<is_device_mdspan_v<DstType>, is_device_mdspan_v<SrcType>>;
  auto constexpr const both_host_accessible =
    std::conjunction_v<is_host_mdspan_v<DstType>, is_host_mdspan_v<SrcType>>;
  auto constexpr const same_layout    = std::is_same_v<DstType::layout_type, SrcType::layout_type>;
  auto constexpr const can_use_device = std::conjunction_v<CUDA_ENABLED, both_device_accessible>;

  auto constexpr const both_float_or_double =
    std::conjunction_v<std::disjunction_v<std::is_same_v<DstType::value_type, float>,
                                          std::is_same_v<DstType::value_type, double>>,
                       std::disjunction_v<std::is_same_v<SrcType::value_type, float>,
                                          std::is_same_v<SrcType::value_type, double>>>;

  auto constexpr const simd_available = false;  // TODO(wphicks)
  // TODO(wphicks): Think about data on different devices

  if constexpr (!can_use_device) {
    RAFT_EXPECTS(both_host_accessible,
                 "Copying to/from non-host-accessible mdspan in non-CUDA-enabled build");
  }

  for (auto i = std::size_t{}; i < SrcType::extents::rank(); ++i) {
    RAFT_EXPECTS(src.extents(i) == dst.extents(i), "Must copy between mdspans of the same shape");
  }

  if constexpr (both_device_accessible && CUDA_ENABLED) {
#ifndef RAFT_DISABLE_CUDA
    if constexpr (same_dtype && same_layout && both_contiguous) {
      // TODO(wphicks): stream
      raft::copy(dst.data_handle(), src.data_handle(), dst.size());
    } else if constexpr (same_dtype && both_float_or_double && both_contiguous &&
                         DstType::extents::rank() == 2) {
      auto constexpr const alpha = typename DstType::value_type{1};
      auto constexpr const beta  = typename DstType::value_type{0};
      CUBLAS_TRY(cublasgeam(resource::get_cublas_handle(res),
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            dst.extent(0),
                            dst.extent(1),
                            &alpha,
                            src.data_handle(),
                            src.stride(0),
                            &beta,
                            static_cast<typename DstType::value_type*>(nullptr),
                            dst.stride(0),
                            dst.data_handle(),
                            dst.stride(0),
                            resource::get_cuda_stream(res)));
    } else {
#ifdef __CUDACC__
      // custom kernel
#else
      // Ordinarily, we would just make this a .cuh file, but we do not want
      // to signal that it *must* be built with CUDA. Instead, if this header
      // is used in a way that requires a CUDA compiler, we fail with an
      // informative error message.
      static_assert(
        !CUDA_ENABLED,
        "When used in a CUDA-enabled build for non-trivial copies on device, mdspan_copy.hpp "
        "includes a kernel launch and must be compiled with a CUDA-enabled compiler. Use this "
        "header in a '.cu' file to ensure it is correctly compiled.");
#endif
    }
#endif
  } else if constexpr (both_host_accessible) {
    if constexpr (same_layout && both_contiguous) {
      // Use STL if possible; this should be well optimized
      std::copy(src.data_handle(), src.data_handle() + dst.size(), dst.data_handle());
    } else if constexpr (both_contiguous && both_float_or_double && simd_available) {
      // Next, use SIMD intrinsics if possible, since generic one-by-one copy implementation is hard
      // for the compiler to vectorize

      // simd transpose, possibly with dtype conversion
    } else {
      // Finally, copy elements one by one, trying at least to perform
      // cache-friendly reads

      auto indices = std::array<index_type, DstType::extents::rank()>{};
      for (auto i = std::size_t{}; i < dst.size(); ++i) {
        if constexpr (std::is_same_v<typename DstType::layout_type, layout_c_contiguous>) {
          // For layout_right/layout_c_contiguous, we iterate over the
          // rightmost extent fastest
          auto dim = DstType::extents::rank();
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
    } else if constexpr (is_device_mdspan_v<DstType>) {
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
