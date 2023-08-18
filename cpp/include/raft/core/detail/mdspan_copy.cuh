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
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan.hpp>

namespace raft {
namespace detail {

template <typename LayoutPolicy, typename IdxType>
auto increment_indices(IdxType* indices, IdxType const* max_indices, int rank, int incr = 1)
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
auto& get_mdspan_elem(MdspanType& md, IdxType const* indices, ResT... resolved_indices)
{
  if constexpr (remaining == IdxType{}) {
    return md(resolved_indices...);
  } else {
    return get_mdspan_elem<MdspanType, IdxType, remaining - 1>(
      md, indices, indices[remaining - 1], &resolved_indices...);
  }
}

template <typename DstType, typename SrcType, int TileDim = 32>
__global__ std::enable_if_t<
  std::conjunction_v<is_device_mdspan_v<DstType>,
                     is_device_mdspan_v<SrcType>,
                     std::is_convertible_v<SrcType::value_type, DstType::value_type>>>
mdspan_device_copy(DstType dst, SrcType src)
{
  // Lay out shmem tile in same layout as source if it is contiguous.
  // Otherwise, lay it out in same layout as destination if destination is
  // contiguous. If neither are contiguous, just fall back to
  // layout_right/layout_c_contiguous
  using tile_layout_policy = std::conditional_v<
    std::disjunction_v<std::is_same_v<typename SrcType::layout_type, layout_c_contiguous>,
                       std::is_same_v<typename SrcType::layout_type, layout_f_contiguous>>,
    SrcType::layout_type,
    std::conditional_v<
      std::disjunction_v<std::is_same_v<typename DstType::layout_type, layout_c_contiguous>,
                         std::is_same_v<typename DstType::layout_type, layout_f_contiguous>>,
      DstType::layout_type,
      layout_c_contiguous>>;
  __shared__ DstType::value_type tile_buffer[TileDim][TileDim + 1];
  auto tile = mdspan<DstType::value_type, TileDim, TileDim + 1>(tile_buffer);

  using index_type =
    std::conditional_t<(std::numeric_limits<typename DstType::extents::index_type>::max() >
                        std::numeric_limits<typename SrcType::extents::index_type>::max()),
                       typename DstType::extents::index_type,
                       typename SrcType::extents::index_type>;
  auto const constexpr tile_elements               = TileDim * TileDim;
  index_type src_indices[DstType::extents::rank()] = {blockIdx.x * tile_elements};
  index_type dst_indices[DstType::extents::rank()] = {blockIdx.x * tile_elements};
  index_type max_indices[DstType::extents::rank()];
  for (auto i = index_type{}; i < DstType::extents::rank(); ++i) {
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
        increment_indices<DstType::layout_policy>(dst_indices, max_indices, gridDim.x);
      }
      increment_indices<DstType::layout_policy>(dst_indices, max_indices, gridDim.y * TileDim);
    }
    valid_indices &= increment_indices<DstType::layout_policy>(
      src_indices, max_indices, blockDim.x * tile_elements);
    increment_indices<SrcType::layout_policy>(dst_indices, max_indices, blockDim.x * tile_elements);
    __syncthreads();
  }
}

}  // namespace detail
}  // namespace raft
