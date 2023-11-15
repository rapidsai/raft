/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cstdint>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/memory_type.hpp>

namespace raft {

template <typename AccessorPolicy>
using managed_accessor = host_device_accessor<AccessorPolicy, memory_type::managed>;

/**
 * @brief std::experimental::mdspan with managed tag to indicate host/device accessibility
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = std::experimental::default_accessor<ElementType>>
using managed_mdspan = mdspan<ElementType, Extents, LayoutPolicy, managed_accessor<AccessorPolicy>>;

template <typename T, bool B>
struct is_managed_mdspan : std::false_type {};
template <typename T>
struct is_managed_mdspan<T, true> : std::bool_constant<T::accessor_type::is_managed_accessible> {};

/**
 * @\brief Boolean to determine if template type T is either raft::managed_mdspan or a derived type
 */
template <typename T>
using is_managed_mdspan_t = is_managed_mdspan<T, is_mdspan_v<T>>;

template <typename T>
using is_input_managed_mdspan_t = is_managed_mdspan<T, is_input_mdspan_v<T>>;

template <typename T>
using is_output_managed_mdspan_t = is_managed_mdspan<T, is_output_mdspan_v<T>>;

/**
 * @\brief Boolean to determine if variadic template types Tn are either raft::managed_mdspan or a
 * derived type
 */
template <typename... Tn>
inline constexpr bool is_managed_mdspan_v = std::conjunction_v<is_managed_mdspan_t<Tn>...>;

template <typename... Tn>
inline constexpr bool is_input_managed_mdspan_v =
  std::conjunction_v<is_input_managed_mdspan_t<Tn>...>;

template <typename... Tn>
inline constexpr bool is_output_managed_mdspan_v =
  std::conjunction_v<is_output_managed_mdspan_t<Tn>...>;

template <typename... Tn>
using enable_if_managed_mdspan = std::enable_if_t<is_managed_mdspan_v<Tn...>>;

template <typename... Tn>
using enable_if_input_managed_mdspan = std::enable_if_t<is_input_managed_mdspan_v<Tn...>>;

template <typename... Tn>
using enable_if_output_managed_mdspan = std::enable_if_t<is_output_managed_mdspan_v<Tn...>>;

/**
 * @brief Create a raft::managed_mdspan
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param ptr Pointer to the data
 * @param exts dimensionality of the array (series of integers)
 * @return raft::managed_mdspan
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous,
          size_t... Extents>
auto make_managed_mdspan(ElementType* ptr, extents<IndexType, Extents...> exts)
{
  return make_mdspan<ElementType, IndexType, LayoutPolicy, true, true>(ptr, exts);
}
}  // end namespace raft
