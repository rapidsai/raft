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
#include <cstdio>
#include <execution>
#include <raft/core/cuda_support.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/stream_view.hpp>
#include <raft/core/resources.hpp>
#include <type_traits>
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/cudart_utils.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#endif

namespace raft {
namespace detail {

template <bool B, typename DstType = void, typename SrcType = void, typename T = void>
struct mdspan_copyable {};

template <typename DstType, typename SrcType, typename T>
struct mdspan_copyable<true, DstType, SrcType, T> {
  using dst_type = std::remove_reference_t<DstType>;
  using src_type = std::remove_reference_t<SrcType>;

  // Extents properties
  using dst_extents_type = typename dst_type::extents_type;
  using src_extents_type = typename src_type::extents_type;
  using index_type =
    std::conditional_t<(std::numeric_limits<typename dst_extents_type::index_type>::max() >
                        std::numeric_limits<typename src_extents_type::index_type>::max()),
                       typename dst_extents_type::index_type,
                       typename src_extents_type::index_type>;

  // Dtype properties
  using dst_value_type                   = typename dst_type::value_type;
  using src_value_type                   = typename src_type::value_type;
  using dst_element_type                 = typename dst_type::element_type;
  using src_element_type                 = typename src_type::element_type;
  auto static constexpr const same_dtype = std::is_same_v<dst_value_type, src_value_type>;
  auto static constexpr const compatible_dtype =
    std::is_convertible_v<src_value_type, dst_element_type>;

  auto static constexpr const dst_float  = std::is_same_v<dst_value_type, float>;
  auto static constexpr const src_float  = std::is_same_v<src_value_type, float>;
  auto static constexpr const dst_double = std::is_same_v<dst_value_type, double>;
  auto static constexpr const src_double = std::is_same_v<src_value_type, double>;

  auto static constexpr const both_float                = dst_float && src_float;
  auto static constexpr const both_double               = dst_double && src_double;
  auto static constexpr const both_float_or_both_double = both_float || both_double;

  // Ranks
  auto static constexpr const dst_rank        = dst_extents_type::rank();
  auto static constexpr const src_rank        = src_extents_type::rank();
  auto static constexpr const compatible_rank = (dst_rank == src_rank);
  auto static constexpr const vector_rank     = (dst_rank == 1);
  auto static constexpr const matrix_rank     = (dst_rank == 2);

  // Layout properties
  using dst_layout_type = typename dst_type::layout_type;
  using src_layout_type = typename src_type::layout_type;

  auto static constexpr const same_layout = std::is_same_v<dst_layout_type, src_layout_type>;

  auto static constexpr const src_contiguous =
    std::disjunction_v<std::is_same<src_layout_type, layout_c_contiguous>,
                       std::is_same<src_layout_type, layout_f_contiguous>>;

  auto static constexpr const dst_contiguous =
    std::disjunction_v<std::is_same<dst_layout_type, layout_c_contiguous>,
                       std::is_same<dst_layout_type, layout_f_contiguous>>;

  auto static constexpr const both_contiguous = src_contiguous && dst_contiguous;

  auto static constexpr const same_underlying_layout =
    std::disjunction_v<std::bool_constant<same_layout>,
                       std::bool_constant<vector_rank && both_contiguous>>;
  // Layout for intermediate tile if copying through custom kernel
  using tile_layout_type =
    std::conditional_t<src_contiguous,
                       src_layout_type,
                       std::conditional_t<dst_contiguous, dst_layout_type, layout_c_contiguous>>;

  // Accessibility
  auto static constexpr const dst_device_accessible = is_device_mdspan_v<dst_type>;
  auto static constexpr const src_device_accessible = is_device_mdspan_v<src_type>;
  auto static constexpr const both_device_accessible =
    dst_device_accessible && src_device_accessible;

  auto static constexpr const dst_host_accessible  = is_host_mdspan_v<dst_type>;
  auto static constexpr const src_host_accessible  = is_host_mdspan_v<src_type>;
  auto static constexpr const both_host_accessible = dst_host_accessible && src_host_accessible;

  // Allowed copy codepaths
  auto static constexpr const can_use_host = both_host_accessible;

#if (defined(__AVX__) || defined(__SSE__) || defined(__ARM_NEON))
  auto static constexpr const can_use_simd = can_use_host && both_contiguous;
#else
  auto static constexpr const can_use_simd = false;
#endif

  auto static constexpr const can_use_std_copy =
    std::conjunction_v<std::bool_constant<can_use_host>,
                       std::bool_constant<compatible_dtype>,
                       std::bool_constant<both_contiguous>,
                       std::bool_constant<same_underlying_layout>>;
  auto static constexpr const can_use_raft_copy =
    std::conjunction_v<std::bool_constant<CUDA_ENABLED>,
                       std::bool_constant<same_dtype>,
                       std::bool_constant<both_contiguous>,
                       std::bool_constant<same_underlying_layout>>;

  auto static constexpr const requires_intermediate =
    !both_host_accessible && !both_device_accessible && !can_use_raft_copy;

  auto static constexpr const use_intermediate_dst =
    std::conjunction_v<std::bool_constant<requires_intermediate>,
                       std::bool_constant<src_device_accessible>>;

  auto static constexpr const use_intermediate_src =
    std::conjunction_v<std::bool_constant<requires_intermediate>,
                       std::bool_constant<!use_intermediate_dst>>;
  auto static constexpr const can_use_device =
    std::conjunction_v<std::bool_constant<CUDA_ENABLED>,
                       std::disjunction<
                         std::bool_constant<both_device_accessible>,
                         std::bool_constant<requires_intermediate>,
                         std::bool_constant<can_use_raft_copy>
                       >
                     >;

  auto static constexpr const can_use_cublas =
    std::conjunction_v<std::bool_constant<can_use_device>,
                       std::bool_constant<compatible_dtype>,
                       std::bool_constant<both_contiguous>,
                       std::bool_constant<!same_underlying_layout>,
                       std::bool_constant<matrix_rank>,
                       std::bool_constant<both_float_or_both_double>>;

  auto static constexpr const custom_kernel_allowed =
    std::conjunction_v<std::bool_constant<can_use_device>,
                       std::bool_constant<!(can_use_raft_copy || can_use_cublas)>>;

  auto static constexpr const custom_kernel_required =
    std::conjunction_v<std::bool_constant<!can_use_host>,
                       std::bool_constant<!(can_use_raft_copy || can_use_cublas)>>;

  // Viable overload?
  auto static constexpr const value = std::conjunction_v<
    std::bool_constant<is_mdspan_v<src_type>>,
    std::bool_constant<is_mdspan_v<dst_type>>,
    std::bool_constant<can_use_host || can_use_device>
  >;
  using type = std::enable_if_t<value, T>;
};

template <typename DstType, typename SrcType, typename T = void>
using mdspan_copyable_t = typename mdspan_copyable<true, DstType, SrcType, T>::type;
template <typename DstType, typename SrcType>
using mdspan_copyable_v = typename mdspan_copyable<true, DstType, SrcType, void>::value;

#ifdef __CUDACC__
template <typename LayoutPolicy, typename IdxType>
__device__ auto increment_indices(IdxType* indices,
                                  IdxType const* max_indices,
                                  int rank,
                                  int incr = 1)
{
  auto valid_index = true;
  auto dim         = std::is_same_v<LayoutPolicy, layout_c_contiguous> ? rank : 0;
  do {
    indices[dim] += incr;
    incr = 0;
    while (indices[dim] >= max_indices[dim]) {
      indices[dim] -= max_indices[dim];
      ++incr;
    }
    if constexpr (std::is_same_v<LayoutPolicy, layout_c_contiguous>) {
      --dim;
      valid_index = dim >= 0;
    } else {
      ++dim;
      valid_index = dim < rank;
    }
  } while (incr != 0);
  return valid_index;
}

template <typename MdspanType,
          typename IdxType,
          IdxType remaining = MdspanType::extents::rank(),
          typename... ResT>
__device__ auto& get_mdspan_elem(MdspanType& md, IdxType const* indices, ResT... resolved_indices)
{
  if constexpr (remaining == IdxType{}) {
    return md(resolved_indices...);
  } else {
    return get_mdspan_elem<MdspanType, IdxType, remaining - 1>(
      md, indices, indices[remaining - 1], &resolved_indices...);
  }
}

template <typename DstType, typename SrcType, int TileDim = 32>
__global__ std::enable_if_t<mdspan_copyable_v<DstType, SrcType>::custom_kernel_allowed>
mdspan_device_copy(DstType dst, SrcType src)
{
  using config = mdspan_copyable<true, DstType, SrcType>;

  __shared__ typename config::dst_value_type tile_buffer[TileDim][TileDim + 1];
  auto tile = mdspan <typename config::dst_value_type, extents<std::uint32_t, TileDim, TileDim + 1>>
  {
    tile_buffer
  };

  auto const constexpr tile_elements       = TileDim * TileDim;
  typename config::index_type src_indices[config::dst_rank] = {blockIdx.x * tile_elements};
  typename config::index_type dst_indices[config::dst_rank] = {blockIdx.x * tile_elements};
  typename config::index_type max_indices[config::dst_rank];
  for (auto i = typename config::index_type{}; i < config::dst_rank; ++i) {
    max_indices[i] = dst.extent(i);
  }

  auto valid_indices = true;
  for (auto i = blockIdx.x * tile_elements; i += tile_elements * blockDim.x; i < dst.size()) {
    for (auto tile_slow = threadIdx.y; tile_slow += gridDim.y; tile_slow < TileDim) {
      for (auto tile_quick = threadIdx.x; tile_quick += gridDim.x; tile_quick < TileDim) {
        if (valid_indices) {
          if constexpr (std::is_same_v<tile_layout_policy, layout_c_contiguous>) {
            tile(tile_slow, tile_quick) = get_mdspan_elem(src, src_indices);
          } else {
            tile(tile_quick, tile_slow) = get_mdspan_elem(src, src_indices);
          }
        }
        valid_indices &=
          increment_indices<SrcType::layout_policy>(src_indices, max_indices, gridDim.x);
      }
      valid_indices &=
        increment_indices<SrcType::layout_policy>(src_indices, max_indices, gridDim.y * TileDim);
    }
    if constexpr (!std::is_same_v<DstType::layout_policy, SrcType::layout_policy>) {
      __syncthreads();
    }
    for (auto tile_slow = threadIdx.y; tile_slow += gridDim.y; tile_slow < TileDim) {
      for (auto tile_quick = threadIdx.x; tile_quick += gridDim.x; tile_quick < TileDim) {
        if (valid_indices) {
          if constexpr (std::is_same_v<DstType::layout_policy, layout_c_contiguous>) {
            get_mdspan_elem(dst, dst_indices) = tile(tile_slow, tile_quick)
          } else {
            get_mdspan_elem(dst, dst_indices) = tile(tile_quick, tile_slow)
          }
        }
        increment_indices<SrcType::layout_policy>(dst_indices, max_indices, gridDim.x);
      }
      increment_indices<SrcType::layout_policy>(dst_indices, max_indices, gridDim.y * TileDim);
    }
    valid_indices &= increment_indices<SrcType::layout_policy>(
      src_indices, max_indices, blockDim.x * tile_elements);
    increment_indices<SrcType::layout_policy>(dst_indices, max_indices, blockDim.x * tile_elements);
    __syncthreads();
  }
}
#endif

template <typename DstType, typename SrcType>
mdspan_copyable_t<DstType, SrcType> copy(resources const& res, DstType&& dst, SrcType const& src)
{
  using config = mdspan_copyable<true, DstType, SrcType>;
  for (auto i = std::size_t{}; i < config::src_rank; ++i) {
    RAFT_EXPECTS(src.extent(i) == dst.extent(i), "Must copy between mdspans of the same shape");
  }

  if constexpr (config::use_intermediate_src) {
    RAFT_LOG_WARN("use_intermediate_src");
    // Copy to intermediate source on device, then perform necessary
    // changes in layout on device, directly into final destination
    auto intermediate = device_mdarray<typename config::src_value_type,
                                       typename config::src_extents_type,
                                       typename config::src_layout_type>(res, src.extents());
    copy(res, intermediate.view(), src);
    copy(res, dst, intermediate.view());

  } else if constexpr (config::use_intermediate_dst) {
    RAFT_LOG_WARN("use_intermediate_dst");
    // Perform necessary changes in layout on device, then copy to final
    // destination on host
    auto intermediate = device_mdarray<typename config::dst_value_type,
                                       typename config::dst_extents_type,
                                       typename config::dst_layout_type>(res, dst.extents());
    copy(res, intermediate.view(), src);
    copy(res, dst, intermediate.view());
  } else if constexpr (config::can_use_raft_copy) {
    RAFT_LOG_WARN("can_use_raft_copy");
#ifndef RAFT_DISABLE_CUDA
    raft::copy(dst.data_handle(), src.data_handle(), dst.size(), resource::get_cuda_stream(res));
#endif
  } else if constexpr (config::can_use_cublas) {
    RAFT_LOG_WARN("can_use_cublas");
    auto constexpr const alpha = typename std::remove_reference_t<DstType>::value_type{1};
    auto constexpr const beta  = typename std::remove_reference_t<DstType>::value_type{0};
    if constexpr (std::is_same_v<typename config::dst_layout_type, layout_c_contiguous>) {
      CUBLAS_TRY(
        linalg::detail::cublasgeam(resource::get_cublas_handle(res),
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   dst.extent(1),
                   dst.extent(0),
                   &alpha,
                   src.data_handle(),
                   src.extent(0),
                   &beta,
                   dst.data_handle(),
                   dst.extent(1),
                   dst.data_handle(),
                   dst.extent(1),
                   resource::get_cuda_stream(res)));
    } else {
      CUBLAS_TRY(
        linalg::detail::cublasgeam(resource::get_cublas_handle(res),
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   dst.extent(0),
                   dst.extent(1),
                   &alpha,
                   src.data_handle(),
                   src.extent(1),
                   &beta,
                   dst.data_handle(),
                   dst.extent(0),
                   dst.data_handle(),
                   dst.extent(0),
                   resource::get_cuda_stream(res)));
    }
  } else if constexpr (config::custom_kernel_allowed) {
    RAFT_LOG_WARN("custom_kernel_allowed");
#ifdef __CUDACC__
    // TODO(wphicks): Determine sensible kernel launch parameters
    mdspan_device_copy<<<32, 1024, 0, resource::get_cuda_stream(res)>>>(dst, src);
#else
    // Should never actually reach this because of enable_ifs
    RAFT_FAIL(
      "raft::copy called in a way that requires custom kernel. Please use "
      "raft/core/mdspan_copy.cuh and include the header in a .cu file");
#endif
  } else if constexpr (config::can_use_std_copy) {
    RAFT_LOG_WARN("can_use_std_copy");
    std::copy(src.data_handle(), src.data_handle() + dst.size(), dst.data_handle());
    // } else if constexpr(config::can_use_simd) {
    //   RAFT_LOG_WARN("can_use_simd");
  } else {
    RAFT_LOG_WARN("Default host copy");
    auto indices = std::array<typename config::index_type, config::dst_rank>{};
    for (auto i = std::size_t{}; i < dst.size(); ++i) {
      if (i != 0) {
        if constexpr (std::is_same_v<typename config::src_layout_type, layout_c_contiguous>) {
          // For layout_right/layout_c_contiguous, we iterate over the
          // rightmost extent fastest
          auto dim = config::src_rank - 1;
          while ((++indices[dim]) == src.extent(dim)) {
            indices[dim] = typename config::index_type{};
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
          while ((indices[dim]++) == src.extent(dim)) {
            indices[dim] = typename config::index_type{};
            ++dim;
          }
        }
      }
      std::apply(dst, indices) = std::apply(src, indices);
    }
  }
}
}  // namespace detail
}  // namespace raft
