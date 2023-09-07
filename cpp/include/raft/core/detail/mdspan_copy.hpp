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
#ifdef __CUDACC__
#include <raft/util/cuda_dev_essentials.cuh>
#endif
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
  auto static constexpr const has_vector_rank     = (dst_rank == 1);
  auto static constexpr const has_matrix_rank     = (dst_rank == 2);

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
                       std::bool_constant<has_vector_rank && both_contiguous>>;
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
  // TODO(wphicks): Following should be only necessary restrictions. Test if
  // perf actually improves once fully implemented.
  // auto static constexpr const can_use_simd = can_use_host && both_contiguous && both_float_or_both_double;
  auto static constexpr const can_use_simd = can_use_host && both_contiguous && both_float && has_matrix_rank;
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
                       std::bool_constant<has_matrix_rank>,
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
auto static constexpr const mdspan_copyable_v = mdspan_copyable<true, DstType, SrcType, void>::value;

template <typename DstType, typename SrcType>
auto static constexpr const mdspan_copyable_with_kernel_v = mdspan_copyable<true, DstType, SrcType, void>::custom_kernel_allowed;
template <typename DstType, typename SrcType>
auto static constexpr const mdspan_uncopyable_with_kernel_v = !mdspan_copyable<true, DstType, SrcType, void>::custom_kernel_allowed;


template <typename DstType, typename SrcType, typename T=void>
using mdspan_copyable_with_kernel_t = std::enable_if_t<mdspan_copyable_with_kernel_v<DstType, SrcType>, T>;

template <typename DstType, typename SrcType, typename T=void>
using mdspan_uncopyable_with_kernel_t = std::enable_if_t<mdspan_uncopyable_with_kernel_v<DstType, SrcType>, T>;

#ifdef __CUDACC__
auto static constexpr const mdspan_copy_tile_dim = 32;
auto static constexpr const mdspan_copy_tile_elems = mdspan_copy_tile_dim * mdspan_copy_tile_dim;

// Helper struct to work around lack of CUDA-native std::apply
template<typename IdxType, IdxType... Idx>
struct index_sequence {
};

template<typename IdxType, IdxType N, IdxType... Idx>
struct make_index_sequence : std::conditional_t<
  N == IdxType{},
  index_sequence<IdxType, Idx...>,
  make_index_sequence<IdxType, N - IdxType{1}, N - IdxType{1}, Idx...>> {};

/* template <typename LambdaT, typename ContainerT, typename IdxT, IdxT... Idx>
__host__ __device__ decltype(auto) apply(LambdaT&& lambda, ContainerT&& args, index_sequence<IdxT, Idx...>)
{
  return lambda(args[Idx]...);
}

template <typename LambdaT, typename ContainerT, typename IdxT, IdxT size>
__host__ __device__ decltype(auto) apply(LambdaT&& lambda, ContainerT&& args)
{
  return apply(std::forward<LambdaT>(lambda), std::forward<ContainerT>(args), make_index_sequence<IdxT, size>{});
} */


/*
 * Given an mdspan and an array of indices, return a reference to the
 * indicated element.
 */
template <typename MdspanType, typename IdxType, IdxType... Idx>
__device__ auto& get_mdspan_elem(MdspanType& md, IdxType const* indices, index_sequence<IdxType, Idx...>)
{
  return md(indices[Idx]...);
}

template <typename MdspanType, typename IdxType>
__device__ auto& get_mdspan_elem(MdspanType& md, IdxType const* indices) {
  return get_mdspan_elem(md, indices, make_index_sequence<IdxType, MdspanType::extents_type::rank()>{});
}

/* Advance old_indices forward by the number of mdspan elements specified
 * by increment. Store the result in indices. Return true if the new
 * indices are valid for the input mdspan.
 */
template <typename MdspanType,
          typename IdxType>
__device__ auto increment_indices(
    IdxType* indices,
    MdspanType const& md,
    IdxType const* old_indices,
    IdxType const* index_strides,
    IdxType increment
) {
#pragma unroll
  for (auto i = typename MdspanType::extents_type::rank_type{}; i < md.rank(); ++i) {
    increment += index_strides[i] * old_indices[i];
  }

#pragma unroll
  for (auto i = typename MdspanType::extents_type::rank_type{}; i < md.rank(); ++i) {
    // Iterate through dimensions in order from slowest to fastest varying
    auto const real_index = [](auto ind) {
      if constexpr (std::is_same_v<typename MdspanType::layout_type, layout_f_contiguous>) {
        return MdspanType::rank() - ind - 1;
      } else {
        return ind;
      }
    }(i);

    auto cur_index = IdxType{};

    // printf("pre-increment: %d %d %d: %d\n", old_indices[0], old_indices[1], old_indices[2], int(increment));
    while (cur_index < md.extent(real_index) - 1 && increment >= index_strides[real_index]) {
      increment -= index_strides[real_index];
      ++cur_index;
    }
    indices[real_index] = cur_index;
  }
    // printf("post-increment: %d %d %d: %d\n", old_indices[0], old_indices[1], old_indices[2], int(increment));

  return increment == IdxType{};
}

/*
 * WARNING: This kernel _must_ be launched with mdspan_copy_tile_dim x
 * mdspan_copy_tile_dim threads per block. This restriction allows for
 * additional optimizations at the expense of generalized launch
 * parameters.
 */
template <typename DstType, typename SrcType>
__global__ mdspan_copyable_with_kernel_t<DstType, SrcType>
mdspan_device_copy(DstType dst, SrcType src)
{
  using config = mdspan_copyable<true, DstType, SrcType>;

  // An intermediate storage location for the data to be copied.
  __shared__ typename config::dst_value_type tile[mdspan_copy_tile_dim][mdspan_copy_tile_dim + 1];

  // Compute the cumulative product of extents in order from fastest to
  // slowest varying extent
  typename config::index_type index_strides[config::dst_rank];
  auto cur_stride = typename config::index_type{1};
#pragma unroll
  for (auto i = typename SrcType::extents_type::rank_type{}; i < config::src_rank; ++i) {
    // Iterate through dimensions in order from fastest to slowest varying
    auto const real_index = [](auto ind) {
      if constexpr (std::is_same_v<typename config::src_layout_type, layout_c_contiguous>) {
        return config::src_rank - ind - 1;
      } else {
        return ind;
      }
    }(i);

    index_strides[real_index] = cur_stride;
    cur_stride *= src.extent(real_index);
  }

  // The index of the first element in the mdspan which will be copied via
  // the current tile for this block.
  typename config::index_type tile_offset[config::dst_rank] = {0};
  /* // 0 0 0
  increment_indices(
    tile_offset,
    src,
    tile_offset,
    index_strides,
    typename config::index_type{0}
  );
  // 1 0 0
  increment_indices(
    tile_offset,
    src,
    tile_offset,
    index_strides,
    typename config::index_type{1}
  );
  // 2 0 0
  increment_indices(
    tile_offset,
    src,
    tile_offset,
    index_strides,
    typename config::index_type{1}
  );
  // 3 0 0
  increment_indices(
    tile_offset,
    src,
    tile_offset,
    index_strides,
    typename config::index_type{1}
  );
  // 4 0 0
  increment_indices(
    tile_offset,
    src,
    tile_offset,
    index_strides,
    typename config::index_type{1}
  );
  // 0 1 0
  increment_indices(
    tile_offset,
    src,
    tile_offset,
    index_strides,
    typename config::index_type{1}
  ); */
  typename config::index_type cur_indices[config::dst_rank];
  auto valid_tile = increment_indices(
    tile_offset,
    src,
    tile_offset,
    index_strides,
    blockIdx.x * mdspan_copy_tile_elems
  );

  while (valid_tile) {
    auto tile_read_x = std::is_same_v<typename config::src_layout_type, layout_f_contiguous> ? threadIdx.x : threadIdx.y;
    auto tile_read_y = std::is_same_v<typename config::src_layout_type, layout_f_contiguous> ? threadIdx.y : threadIdx.x;

    auto valid_index = increment_indices(
      cur_indices,
      src,
      tile_offset,
      index_strides,
      tile_read_x * mdspan_copy_tile_dim + tile_read_y
    );

    if constexpr (config::same_underlying_layout || !config::dst_contiguous) {
      if (valid_index) {
        tile[tile_read_x][tile_read_y] = get_mdspan_elem(src, cur_indices);
        get_mdspan_elem(dst, cur_indices) = tile[tile_read_x][tile_read_y];
      }
    } else {
      if (valid_index) {
        // printf("read: %d %d %d -> %d %d: %d\n", cur_indices[0], cur_indices[1], cur_indices[2], tile_read_x, tile_read_y, int(get_mdspan_elem(src, cur_indices)));
        tile[tile_read_x][tile_read_y] = get_mdspan_elem(src, cur_indices);
      }
      __syncthreads();

      valid_index = increment_indices(
        cur_indices,
        src,
        tile_offset,
        index_strides,
        tile_read_y * mdspan_copy_tile_dim + tile_read_x
      );
      if (valid_index) {
        // printf("write: %d %d -> %d %d %d: %d\n", tile_read_x, tile_read_y, cur_indices[0], cur_indices[1], cur_indices[2], int(tile[tile_read_y][tile_read_x]));
        get_mdspan_elem(dst, cur_indices) = tile[tile_read_y][tile_read_x];
        // printf("final: %d %d -> %d %d %d: %d\n", tile_read_x, tile_read_y, cur_indices[0], cur_indices[1], cur_indices[2], int(get_mdspan_elem(dst, cur_indices)));
      }
      __syncthreads();
    }
    valid_tile = increment_indices(
      tile_offset,
      src,
      tile_offset,
      index_strides,
      blockDim.x * mdspan_copy_tile_elems
    ); 
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
    // Copy to intermediate source on device, then perform necessary
    // changes in layout on device, directly into final destination
    using mdarray_t = device_mdarray<
      typename config::src_value_type,
      typename config::src_extents_type,
      typename config::src_layout_type
    >;
    auto intermediate = mdarray_t(
      res,
      typename mdarray_t::mapping_type{src.extents()},
      typename mdarray_t::container_policy_type{}
    );
    detail::copy(res, intermediate.view(), src);
    detail::copy(res, dst, intermediate.view());

  } else if constexpr (config::use_intermediate_dst) {
    // Perform necessary changes in layout on device, then copy to final
    // destination on host
    using mdarray_t = device_mdarray<
      typename config::dst_value_type,
      typename config::dst_extents_type,
      typename config::dst_layout_type
    >;
    auto intermediate = mdarray_t(
      res,
      typename mdarray_t::mapping_type{dst.extents()},
      typename mdarray_t::container_policy_type{}
    );
    detail::copy(res, intermediate.view(), src);
    detail::copy(res, dst, intermediate.view());
  } else if constexpr (config::can_use_raft_copy) {
#ifndef RAFT_DISABLE_CUDA
    raft::copy(dst.data_handle(), src.data_handle(), dst.size(), resource::get_cuda_stream(res));
#endif
  } else if constexpr (config::can_use_cublas) {
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
#ifdef __CUDACC__
    auto const blocks = std::min(
      // This maximum is somewhat arbitrary. Could query the device to see
      // how many blocks we could reasonably allow, but this is probably
      // sufficient considering that this kernel will likely overlap with
      // real computations for most use cases.
      typename config::index_type{32},
      raft::ceildiv(
        typename config::index_type(dst.size()),
        typename config::index_type(mdspan_copy_tile_elems))
    );
    auto constexpr const threads = dim3{mdspan_copy_tile_dim, mdspan_copy_tile_dim, 1};
    mdspan_device_copy<<<blocks, threads, 0, resource::get_cuda_stream(res)>>>(dst, src);
#else
    // Should never actually reach this because of enable_ifs
    RAFT_FAIL(
      "raft::copy called in a way that requires custom kernel. Please use "
      "raft/core/mdspan_copy.cuh and include the header in a .cu file");
#endif
  } else if constexpr (config::can_use_std_copy) {
    std::copy(src.data_handle(), src.data_handle() + dst.size(), dst.data_handle());
  } else if constexpr(config::can_use_simd) {
    RAFT_LOG_WARN("can_use_simd");
#ifdef __SSE__
    constexpr auto elem_per_vector = 4;  // 4 floats per __m128

    for (auto i = 0; i < src.extent(0); i += elem_per_vector) {
      for (auto j = 0; j < src.extent(1); j += elem_per_vector) {
        // Load a row of 4 floats from src into row0
        __m128 row0 = _mm_loadu_ps(&src(i, j));
        // Load the next row of 4 floats from src into row1
        __m128 row1 = _mm_loadu_ps(&src(i + 1, j));
        // Load another row of 4 floats from src into row2
        __m128 row2 = _mm_loadu_ps(&src(i + 2, j));
        // Load the final row of 4 floats from src into row3
        __m128 row3 = _mm_loadu_ps(&src(i + 3, j));

        // Shuffle elements from row0 and row1. tmp0 holds elements (0,1) from both row0 and row1
        __m128 tmp0 = _mm_shuffle_ps(row0, row1, _MM_SHUFFLE(1, 0, 1, 0));
        // Shuffle elements from row0 and row1. tmp2 holds elements (2,3) from both row0 and row1
        __m128 tmp2 = _mm_shuffle_ps(row0, row1, _MM_SHUFFLE(3, 2, 3, 2));
        // Shuffle elements from row2 and row3. tmp1 holds elements (0,1) from both row2 and row3
        __m128 tmp1 = _mm_shuffle_ps(row2, row3, _MM_SHUFFLE(1, 0, 1, 0));
        // Shuffle elements from row2 and row3. tmp3 holds elements (2,3) from both row2 and row3
        __m128 tmp3 = _mm_shuffle_ps(row2, row3, _MM_SHUFFLE(3, 2, 3, 2));

        // Final shuffle and store. Shuffle elements from tmp0 and tmp1 into first row of dst.
        _mm_storeu_ps(&dst(j, i), _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(2, 0, 2, 0)));
        // Final shuffle and store. Shuffle elements from tmp0 and tmp1 into second row of dst.
        _mm_storeu_ps(&dst(j + 1, i), _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(3, 1, 3, 1)));
        // Final shuffle and store. Shuffle elements from tmp2 and tmp3 into third row of dst.
        _mm_storeu_ps(&dst(j + 2, i), _mm_shuffle_ps(tmp2, tmp3, _MM_SHUFFLE(2, 0, 2, 0)));
        // Final shuffle and store. Shuffle elements from tmp2 and tmp3 into fourth row of dst.
        _mm_storeu_ps(&dst(j + 3, i), _mm_shuffle_ps(tmp2, tmp3, _MM_SHUFFLE(3, 1, 3, 1)));
      }
    }
#endif
  } else {
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
