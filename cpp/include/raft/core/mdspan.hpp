/*
 * Copyright (2019) Sandia Corporation
 *
 * The source code is licensed under the 3-clause BSD license found in the LICENSE file
 * thirdparty/LICENSES/mdarray.license
 */

/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/core/detail/accessor_mixin.hpp>
#include <raft/core/detail/mdspan_util.hpp>
#include <raft/core/error.hpp>
#include <raft/detail/span.hpp>  // dynamic_extent
#include <raft/thirdparty/mdspan/include/experimental/mdspan>

namespace raft {
/**
 * @brief Dimensions extents for raft::mdspan
 */
template <typename IndexType, size_t... ExtentsPack>
using extents = std::experimental::extents<IndexType, ExtentsPack...>;

namespace stdex = std::experimental;

/**
 * @defgroup C-Contiguous layout for mdarray and mdspan. Implies row-major and contiguous memory.
 * @{
 */
using stdex::layout_right;
using layout_c_contiguous = layout_right;
using row_major           = layout_right;
/** @} */

/**
 * @defgroup F-Contiguous layout for mdarray and mdspan. Implies column-major and contiguous memory.
 * @{
 */
using stdex::layout_left;
using layout_f_contiguous = layout_left;
using col_major           = layout_left;
/** @} */

template <typename IndexType>
using vector_extent = stdex::extents<IndexType, dynamic_extent>;

template <typename IndexType>
using matrix_extent = stdex::extents<IndexType, dynamic_extent, dynamic_extent>;

template <typename IndexType = std::uint32_t>
using scalar_extent = stdex::extents<IndexType, 1>;

/**
 * @brief Strided layout for non-contiguous memory.
 */
using stdex::layout_stride;

template <typename IndexType>
using extent_1d = vector_extent<IndexType>;

template <typename IndexType>
using extent_2d = matrix_extent<IndexType>;

template <typename IndexType>
using extent_3d = stdex::extents<IndexType, dynamic_extent, dynamic_extent, dynamic_extent>;

template <typename IndexType>
using extent_4d =
  stdex::extents<IndexType, dynamic_extent, dynamic_extent, dynamic_extent, dynamic_extent>;

template <typename IndexType>
using extent_5d = stdex::extents<IndexType,
                                 dynamic_extent,
                                 dynamic_extent,
                                 dynamic_extent,
                                 dynamic_extent,
                                 dynamic_extent>;
/** @} */

template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = stdex::default_accessor<ElementType>>
using mdspan = stdex::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

/**
 * Ensure all types listed in the parameter pack `Extents` are integral types.
 * Usage:
 *   put it as the last nameless template parameter of a function:
 *     `typename = ensure_integral_extents<Extents...>`
 */
template <typename... Extents>
using ensure_integral_extents = std::enable_if_t<std::conjunction_v<std::is_integral<Extents>...>>;

/**
 * @\brief Template checks and helpers to determine if type T is an std::mdspan
 *         or a derived type
 */

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void __takes_an_mdspan_ptr(mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>*);

template <typename T, typename = void>
struct is_mdspan : std::false_type {
};
template <typename T>
struct is_mdspan<T, std::void_t<decltype(__takes_an_mdspan_ptr(std::declval<T*>()))>>
  : std::true_type {
};

template <typename T>
using is_mdspan_t = is_mdspan<std::remove_const_t<T>>;

// template <typename T>
// inline constexpr bool is_mdspan_v = is_mdspan_t<T>::value;

/**
 * @\brief Boolean to determine if variadic template types Tn are either
 *          raft::host_mdspan/raft::device_mdspan or their derived types
 */
template <typename... Tn>
inline constexpr bool is_mdspan_v = std::conjunction_v<is_mdspan_t<Tn>...>;

template <typename... Tn>
using enable_if_mdspan = std::enable_if_t<is_mdspan_v<Tn...>>;

// uint division optimization inspired by the CIndexer in cupy.  Division operation is
// slow on both CPU and GPU, especially 64 bit integer.  So here we first try to avoid 64
// bit when the index is smaller, then try to avoid division when it's exp of 2.
template <typename I, typename IndexType, size_t... Extents>
MDSPAN_INLINE_FUNCTION auto unravel_index_impl(I idx, stdex::extents<IndexType, Extents...> shape)
{
  constexpr auto kRank = static_cast<int32_t>(shape.rank());
  std::size_t index[shape.rank()]{0};  // NOLINT
  static_assert(std::is_signed<decltype(kRank)>::value,
                "Don't change the type without changing the for loop.");
  for (int32_t dim = kRank; --dim > 0;) {
    auto s = static_cast<std::remove_const_t<std::remove_reference_t<I>>>(shape.extent(dim));
    if (s & (s - 1)) {
      auto t     = idx / s;
      index[dim] = idx - t * s;
      idx        = t;
    } else {  // exp of 2
      index[dim] = idx & (s - 1);
      idx >>= detail::popc(s - 1);
    }
  }
  index[0] = idx;
  return detail::arr_to_tup(index);
}

/**
 * @brief Create a raft::mdspan
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @tparam is_host_accessible whether the data is accessible on host
 * @tparam is_device_accessible whether the data is accessible on device
 * @param ptr Pointer to the data
 * @param exts dimensionality of the array (series of integers)
 * @return raft::mdspan
 */
template <typename ElementType,
          typename IndexType        = std::uint32_t,
          typename LayoutPolicy     = layout_c_contiguous,
          bool is_host_accessible   = false,
          bool is_device_accessible = true,
          size_t... Extents>
auto make_mdspan(ElementType* ptr, extents<IndexType, Extents...> exts)
{
  using accessor_type = detail::accessor_mixin<std::experimental::default_accessor<ElementType>,
                                               is_host_accessible,
                                               is_device_accessible>;

  return mdspan<ElementType, decltype(exts), LayoutPolicy, accessor_type>{ptr, exts};
}

/**
 * @brief Create raft::extents to specify dimensionality
 *
 * @tparam IndexType The type of each dimension of the extents
 * @tparam Extents Dimensions (a series of integers)
 * @param exts The desired dimensions
 * @return raft::extents
 */
template <typename IndexType, typename... Extents, typename = ensure_integral_extents<Extents...>>
auto make_extents(Extents... exts)
{
  return extents<IndexType, ((void)exts, dynamic_extent)...>{exts...};
}

/**
 * @brief Flatten raft::mdspan into a 1-dim array view
 *
 * @tparam mdspan_type Expected type raft::host_mdspan or raft::device_mdspan
 * @param mds raft::host_mdspan or raft::device_mdspan object
 * @return raft::host_mdspan or raft::device_mdspan with vector_extent
 *         depending on AccessoryPolicy
 */
template <typename mdspan_type, typename = enable_if_mdspan<mdspan_type>>
auto flatten(mdspan_type mds)
{
  RAFT_EXPECTS(mds.is_exhaustive(), "Input must be contiguous.");

  vector_extent<typename mdspan_type::size_type> ext{mds.size()};

  return stdex::mdspan<typename mdspan_type::element_type,
                       decltype(ext),
                       typename mdspan_type::layout_type,
                       typename mdspan_type::accessor_type>(mds.data_handle(), ext);
}

/**
 * @brief Reshape raft::host_mdspan or raft::device_mdspan
 *
 * @tparam mdspan_type Expected type raft::host_mdspan or raft::device_mdspan
 * @tparam IndexType the index type of the extents
 * @tparam Extents raft::extents for dimensions
 * @param mds raft::host_mdspan or raft::device_mdspan object
 * @param new_shape Desired new shape of the input
 * @return raft::host_mdspan or raft::device_mdspan, depending on AccessorPolicy
 */
template <typename mdspan_type,
          typename IndexType = std::uint32_t,
          size_t... Extents,
          typename = enable_if_mdspan<mdspan_type>>
auto reshape(mdspan_type mds, extents<IndexType, Extents...> new_shape)
{
  RAFT_EXPECTS(mds.is_exhaustive(), "Input must be contiguous.");

  size_t new_size = 1;
  for (size_t i = 0; i < new_shape.rank(); ++i) {
    new_size *= new_shape.extent(i);
  }
  RAFT_EXPECTS(new_size == mds.size(), "Cannot reshape array with size mismatch");

  return stdex::mdspan<typename mdspan_type::element_type,
                       decltype(new_shape),
                       typename mdspan_type::layout_type,
                       typename mdspan_type::accessor_type>(mds.data_handle(), new_shape);
}

/**
 * \brief Turns linear index into coordinate.  Similar to numpy unravel_index.
 *
 * \code
 *   auto m = make_host_matrix<float>(7, 6);
 *   auto m_v = m.view();
 *   auto coord = unravel_index(2, m.extents(), typename decltype(m)::layout_type{});
 *   std::apply(m_v, coord) = 2;
 * \endcode
 *
 * \param idx    The linear index.
 * \param shape  The shape of the array to use.
 * \param layout Must be `layout_c_contiguous` (row-major) in current implementation.
 *
 * \return A std::tuple that represents the coordinate.
 */
template <typename Idx, typename IndexType, typename LayoutPolicy, size_t... Exts>
MDSPAN_INLINE_FUNCTION auto unravel_index(Idx idx,
                                          extents<IndexType, Exts...> shape,
                                          LayoutPolicy const& layout)
{
  static_assert(std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(layout)>>,
                               layout_c_contiguous>,
                "Only C layout is supported.");
  static_assert(std::is_integral_v<Idx>, "Index must be integral.");
  auto constexpr kIs64 = sizeof(std::remove_cv_t<std::remove_reference_t<Idx>>) == sizeof(uint64_t);
  if (kIs64 && static_cast<uint64_t>(idx) > std::numeric_limits<uint32_t>::max()) {
    return unravel_index_impl<uint64_t>(static_cast<uint64_t>(idx), shape);
  } else {
    return unravel_index_impl<uint32_t>(static_cast<uint32_t>(idx), shape);
  }
}

}  // namespace raft