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

  auto static constexpr const src_contiguous = std::disjunction_v<
    std::is_same_v<src_layout_type, layout_c_contiguous>,
    std::is_same_v<src_layout_type, layout_f_contiguous>
  >;

  auto static constexpr const dst_contiguous = std::disjunction_v<
    std::is_same_v<dst_layout_type, layout_c_contiguous>,
    std::is_same_v<dst_layout_type, layout_f_contiguous>
  >;

  auto static constexpr const both_contiguous = src_contiguous && dst_contiguous;

  // Accessibility
  auto static constexpr const dst_device_accessible = is_device_mdspan_v<dst_type>;
  auto static constexpr const src_device_accessible = is_device_mdspan_v<src_type>;
  auto static constexpr const both_device_accessible = dst_device_accessible && src_device_accessible;

  auto static constexpr const dst_host_accessible = is_host_mdspan_v<dst_type>;
  auto static constexpr const src_host_accessible = is_host_mdspan_v<src_type>;
  auto static constexpr const both_host_accessible = dst_host_accessible && src_host_accessible;

  auto static constexpr const can_use_device = std::conjunction_v<CUDA_ENABLED, both_device_accessible>;

  auto static constexpr const can_use_host = both_host_accessible;

#if (defined(__AVX__) || defined(__SSE__) || defined(__ARM_NEON))
  auto static constexpr const can_use_simd = both_host_accessible;
# else
  auto static constexpr const can_use_simd = false;
#endif

  // Viable overload?
  using type = std::enable_if_t<
    std::conjunction_v<
      is_mdspan<dst_type>,
      is_mdspan<src_type>,
      std::is_convertible<src_value_type, dst_element_type>,
      std::bool_constant<compatible_rank>,
      std::bool_constant<can_use_device || can_use_host>
    >, T
  >;
};

// Need custom kernel if...
template <typename DstType, typename SrcType>
struct mdspan_copy_requires_custom_kernel : std::conjunction<
  // CUDA build is enabled...
  std::bool_constant<CUDA_ENABLED>,
  // and both mdspans can be accessed on device...
  std::bool_constant<is_device_mdspan_v<std::remove_reference_t<DstType>, SrcType>>,
  // and we cannot use cudaMemcpyAsync or cuBLAS.
  std::bool_constant<!std::conjunction_v<
    // We CAN use cudaMemcpyAsync or cuBLAS if...
    // src and dst dtypes are the same...
    std::is_same<typename std::remove_reference_t<DstType>::value_type, typename SrcType::value_type>,
    // and layout is contiguous...
    std::conjunction<
      std::disjunction<
        std::is_same<typename std::remove_reference_t<DstType>::layout_type, layout_c_contiguous>,
        std::is_same<typename std::remove_reference_t<DstType>::layout_type, layout_f_contiguous>
      >,
      std::disjunction<
        std::is_same<typename SrcType::layout_type, layout_c_contiguous>,
        std::is_same<typename SrcType::layout_type, layout_f_contiguous>
      >
    >,
    // and EITHER...
    std::disjunction<
      // the mdspans have the same layout (cudaMemcpyAsync)...
      std::is_same<typename std::remove_reference_t<DstType>::layout_type, typename SrcType::layout_type>,
      // OR the mdspans are 1D (in which case the underlying memory layout
      // is actually the same...
      std::bool_constant<std::remove_reference_t<DstType>::extents_type::rank() == 1>,
      // OR the data are a 2D matrix of either floats or doubles, in which
      // case we can perform the transpose with cuBLAS
      std::conjunction<
        std::bool_constant<std::remove_reference_t<DstType>::extents_type::rank() == 2>,
        std::disjunction<
          std::is_same<typename std::remove_reference_t<DstType>::value_type, float>,
          std::is_same<typename std::remove_reference_t<DstType>::value_type, double>
        > // end float or double check
      > // end cuBLAS compatibility check
    > // end cudaMemcpy || cuBLAS check
  >>
> {};

template <typename DstType, typename SrcType>
auto constexpr mdspan_copy_requires_custom_kernel_v = mdspan_copy_requires_custom_kernel<std::remove_reference_t<DstType>, SrcType>{}();


template <typename DstType, typename SrcType>
std::enable_if_t<
  std::conjunction_v<is_mdspan_v<std::remove_reference_t<DstType>, SrcType>,
                     std::is_convertible_v<typename SrcType::value_type, typename std::remove_reference_t<DstType>::element_type>,
                     std::remove_reference_t<DstType>::extents_type::rank() == SrcType::extents_type::rank()>>
copy(resources const& res, DstType&& dst, SrcType const& src)
{
  using index_type =
    std::conditional_t<(std::numeric_limits<typename std::remove_reference_t<DstType>::extents_type::index_type>::max() >
                        std::numeric_limits<typename SrcType::extents_type::index_type>::max()),
                       typename std::remove_reference_t<DstType>::extents_type::index_type,
                       typename SrcType::extents_type::index_type>;
  auto constexpr const both_contiguous = std::conjunction_v<
    std::disjunction_v<std::is_same_v<typename std::remove_reference_t<DstType>::layout_type, layout_c_contiguous>,
                       std::is_same_v<typename std::remove_reference_t<DstType>::layout_type, layout_f_contiguous>>,
    std::disjunction_v<std::is_same_v<typename SrcType::layout_type, layout_c_contiguous>,
                       std::is_same_v<typename SrcType::layout_type, layout_f_contiguous>>>;
  auto constexpr const same_dtype = std::is_same_v<typename std::remove_reference_t<DstType>::value_type, typename SrcType::value_type>;
  auto constexpr const both_device_accessible = is_device_mdspan_v<std::remove_reference_t<DstType>, SrcType>;
  auto constexpr const both_host_accessible = is_host_mdspan_v<std::remove_reference_t<DstType>, SrcType>;
  auto constexpr const same_layout    = std::is_same_v<typename std::remove_reference_t<DstType>::layout_type, typename SrcType::layout_type>;
  auto constexpr const can_use_device = std::conjunction_v<CUDA_ENABLED, both_device_accessible>;

  auto constexpr const both_float_or_double =
    std::conjunction_v<std::disjunction_v<std::is_same_v<typename std::remove_reference_t<DstType>::value_type, float>,
                                          std::is_same_v<typename std::remove_reference_t<DstType>::value_type, double>>,
                       std::disjunction_v<std::is_same_v<typename SrcType::value_type, float>,
                                          std::is_same_v<typename SrcType::value_type, double>>>;

  auto constexpr const simd_available = false;  // TODO(wphicks)
  // TODO(wphicks): If data are on different devices, perform a
  // cudaMemcpyPeer and then call recursively

  if constexpr (!can_use_device) {
    static_assert(both_host_accessible,
                 "Copying to/from non-host-accessible mdspan in non-CUDA-enabled build");
  }

  for (auto i = std::size_t{}; i < SrcType::extents_type::rank(); ++i) {
    RAFT_EXPECTS(src.extents(i) == dst.extents(i), "Must copy between mdspans of the same shape");
  }

  if constexpr (can_use_device) {
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
