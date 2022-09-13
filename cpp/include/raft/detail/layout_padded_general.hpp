/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
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

#include <raft/cudart_utils.h>
#include <raft/thirdparty/mdspan/include/experimental/mdspan>

#include <algorithm>
#include <array>
#include <numeric>

namespace std {
namespace experimental {

namespace stdex = std::experimental;

enum class StorageOrderType { column_major_t, row_major_t };

// keeping ByteAlignment as optional to allow testing
template <class ValueType, size_t ByteAlignment = 128>
struct padding {
  static_assert(std::is_same<remove_cv_t<ValueType>, ValueType>::value,
                "std::experimental::padding ValueType has to be provided without "
                "const or volatile specifiers.");
  static_assert(ByteAlignment % sizeof(ValueType) == 0 || sizeof(ValueType) % ByteAlignment == 0,
                "std::experimental::padding sizeof(ValueType) has to be multiple or "
                "divider of ByteAlignment.");
  static constexpr size_t value = std::max(ByteAlignment / sizeof(ValueType), 1ul);
};

namespace details {

// offset_index_sequence idea comes from "offset_sequence" here:
// https://devblogs.microsoft.com/oldnewthing/20200625-00/?p=103903
//
// offset_index_sequence adds N to each element of the given IndexSequence.
// We can't just template on the parameter pack of indices directly;
// the pack needs to be contained in some type.
// We choose index_sequence because it stores no run-time data.
template <std::size_t N, class IndexSequence>
struct offset_index_sequence;

template <std::size_t N, std::size_t... Indices>
struct offset_index_sequence<N, std::index_sequence<Indices...>> {
  using type = std::index_sequence<(Indices + N)...>;
};

template <std::size_t N, typename IndexSequence>
using offset_index_sequence_t = typename offset_index_sequence<N, IndexSequence>::type;

static_assert(std::is_same<offset_index_sequence_t<3, std::make_index_sequence<4>>,
                           std::index_sequence<3, 4, 5, 6>>::value,
              "offset_index_sequence defined incorrectly.");

// iota_index_sequence defines the half-open sequence
// begin, begin+1, begin+2, ..., end-1.
// If end == begin, then the sequence is empty (we permit this).
//
// Defining the struct first, rather than going straight to the type alias,
// lets us check the template arguments.
template <std::size_t begin, std::size_t end>
struct iota_index_sequence {
  static_assert(end >= begin, "end must be >= begin.");
  using type = offset_index_sequence_t<begin, std::make_index_sequence<end - begin>>;
};

// iota_index_sequence_t is like make_index_sequence,
// except that it starts with begin instead of 0.
template <std::size_t begin, std::size_t end>
using iota_index_sequence_t = typename iota_index_sequence<begin, end>::type;

// The *_helper functions work around not having C++20
// templated lambdas: []<size_t... TrailingIndices>{} .

// The third argument should always be
// iota_index_sequence_t<1, ReturnExtents::rank()>.
template <class ReturnExtents,
          std::size_t PrePaddedExtent,
          class InnerExtents,
          std::size_t... TrailingIndices>
MDSPAN_INLINE_FUNCTION constexpr ReturnExtents layout_left_extents_helper(
  const stdex::extents<typename InnerExtents::index_type, PrePaddedExtent>& pre_padded_extent,
  const InnerExtents& inner_extents,
  std::index_sequence<TrailingIndices...>)
{
  static_assert(sizeof...(TrailingIndices) + 1 == ReturnExtents::rank(),
                "sizeof...(TrailingIndices) + 1 != ReturnExtents::rank()");
  static_assert(InnerExtents::rank() == ReturnExtents::rank(),
                "InnerExtents::rank() != ReturnExtents::rank()");
  using index_type = typename ReturnExtents::index_type;
  return ReturnExtents{pre_padded_extent.extent(0),
                       index_type(inner_extents.extent(TrailingIndices))...};
}

// The third argument should always be
// iota_index_sequence_t<0, ReturnExtents::rank() - 1>.
template <class ReturnExtents,
          std::size_t PrePaddedExtent,
          class InnerExtents,
          std::size_t... LeadingIndices>
MDSPAN_INLINE_FUNCTION constexpr ReturnExtents layout_right_extents_helper(
  const InnerExtents& inner_extents,
  const stdex::extents<typename InnerExtents::index_type, PrePaddedExtent>& pre_padded_extent,
  std::index_sequence<LeadingIndices...>)
{
  static_assert(sizeof...(LeadingIndices) + 1 == ReturnExtents::rank(),
                "sizeof...(LeadingIndices) + 1 != ReturnExtents::rank()");
  static_assert(InnerExtents::rank() == ReturnExtents::rank(),
                "InnerExtents::rank() != ReturnExtents::rank()");
  using index_type = typename ReturnExtents::index_type;
  return ReturnExtents{index_type(inner_extents.extent(LeadingIndices))...,
                       pre_padded_extent.extent(0)};
}

template <class ReturnExtents,
          std::size_t PrePaddedExtent,
          class IndexType,
          std::size_t... InnerExtents>
MDSPAN_INLINE_FUNCTION constexpr ReturnExtents layout_left_extents(
  const stdex::extents<IndexType, PrePaddedExtent>& pre_padded_extent,
  const stdex::extents<IndexType, InnerExtents...>& inner_extents)
{
  return layout_left_extents_helper<ReturnExtents>(
    pre_padded_extent, inner_extents, details::iota_index_sequence_t<1, ReturnExtents::rank()>{});
}

// Rank-0 pre_padded_extent means rank-0 input,
// but the latter turns out not to matter here.

template <class ReturnExtents, class IndexType, std::size_t... InnerExtents>
MDSPAN_INLINE_FUNCTION constexpr ReturnExtents layout_left_extents(
  const stdex::extents<IndexType>& /* pre_padded_extent */,
  const stdex::extents<IndexType, InnerExtents...>& inner_extents)
{
  return inner_extents;
}

template <class ReturnExtents,
          std::size_t PrePaddedExtent,
          class IndexType,
          std::size_t... InnerExtents>
MDSPAN_INLINE_FUNCTION constexpr ReturnExtents layout_right_extents(
  const stdex::extents<IndexType, InnerExtents...>& inner_extents,
  const stdex::extents<IndexType, PrePaddedExtent>& pre_padded_extent)
{
  // If rank() is zero, size_t(-1) would be a very large upper bound.
  static_assert(ReturnExtents::rank() != 0, "ReturnExtents::rank() must not be 0");
  return layout_right_extents_helper<ReturnExtents>(
    inner_extents,
    pre_padded_extent,
    details::iota_index_sequence_t<0, ReturnExtents::rank() - 1>{});
}

// Rank-0 pre_padded_extent means rank-0 input,
// but the latter turns out not to matter here.

template <class ReturnExtents, class IndexType, std::size_t... InnerExtents>
MDSPAN_INLINE_FUNCTION constexpr ReturnExtents layout_right_extents(
  const stdex::extents<IndexType, InnerExtents...>& inner_extents,
  const stdex::extents<IndexType>& /* pre_padded_extent */)
{
  return inner_extents;
}

template <class InputExtentsType, std::size_t PaddingExtent, std::size_t... Indices>
MDSPAN_INLINE_FUNCTION constexpr auto pad_extents_left_helper(
  const InputExtentsType& input,
  const stdex::extents<typename InputExtentsType::index_type, PaddingExtent>& padding,
  std::index_sequence<Indices...>)
{
  // NOTE (mfh 2022/09/04) This can be if constexpr,
  // if the compiler supports it.
  if /* constexpr */ (PaddingExtent == stdex::dynamic_extent) {
    assert(padding.extent(0) != stdex::dynamic_extent);
  }
  using input_type           = std::remove_cv_t<std::remove_reference_t<InputExtentsType>>;
  using index_type           = typename input_type::index_type;
  constexpr std::size_t rank = input_type::rank();
  static_assert(sizeof...(Indices) == std::size_t(rank - 1), "Indices pack has the wrong size.");
  using return_type =
    stdex::extents<index_type, PaddingExtent, input_type::static_extent(Indices)...>;
  return return_type{index_type(padding.extent(0)), input.extent(Indices)...};
}

template <class InputExtentsType, std::size_t PaddingExtent, std::size_t... Indices>
MDSPAN_INLINE_FUNCTION constexpr auto pad_extents_right_helper(
  const InputExtentsType& input,
  const stdex::extents<typename InputExtentsType::index_type, PaddingExtent>& padding,
  std::index_sequence<Indices...>)
{
  // NOTE (mfh 2022/09/04) This can be if constexpr,
  // if the compiler supports it.
  if /* constexpr */ (PaddingExtent == stdex::dynamic_extent) {
    assert(padding.extent(0) != stdex::dynamic_extent);
  }
  using input_type           = std::remove_cv_t<std::remove_reference_t<InputExtentsType>>;
  using index_type           = typename input_type::index_type;
  constexpr std::size_t rank = input_type::rank();
  static_assert(sizeof...(Indices) == std::size_t(rank - 1), "Indices pack has the wrong size.");
  using return_type =
    stdex::extents<index_type, input_type::static_extent(Indices)..., PaddingExtent>;
  return return_type{input.extent(Indices)..., index_type(padding.extent(0))};
}

// Rank-0 and rank-1 mdspan don't need extra padding from their layout.
// They rely on an "aligned_accessor" and on the data_handle's alignment.

MDSPAN_TEMPLATE_REQUIRES(class IndexType,
                         std::size_t PaddingExtent,
                         std::size_t... InputExtents,
                         /* requires */ (sizeof...(InputExtents) <= std::size_t(1)))
MDSPAN_INLINE_FUNCTION constexpr auto pad_extents_left(
  const stdex::extents<IndexType, InputExtents...>& input,
  const stdex::extents<IndexType, PaddingExtent> /* padding */)
{
  return input;
}

MDSPAN_TEMPLATE_REQUIRES(class IndexType,
                         std::size_t PaddingExtent,
                         std::size_t... InputExtents,
                         /* requires */ (sizeof...(InputExtents) <= std::size_t(1)))
MDSPAN_INLINE_FUNCTION constexpr auto pad_extents_right(
  const stdex::extents<IndexType, InputExtents...>& input,
  const stdex::extents<IndexType, PaddingExtent> /* padding */)
{
  return input;
}

// rank > 1 case follows.

MDSPAN_TEMPLATE_REQUIRES(class IndexType,
                         std::size_t PaddingExtent,
                         std::size_t... InputExtents,
                         /* requires */ (sizeof...(InputExtents) > std::size_t(1)))
MDSPAN_INLINE_FUNCTION constexpr auto pad_extents_left(
  const stdex::extents<IndexType, InputExtents...>& input,
  const stdex::extents<IndexType, PaddingExtent> padding)
{
  constexpr std::size_t rank = sizeof...(InputExtents);
  return details::pad_extents_left_helper(
    input, padding, details::iota_index_sequence_t<1, rank>{});
}

MDSPAN_TEMPLATE_REQUIRES(class IndexType,
                         std::size_t PaddingExtent,
                         std::size_t... InputExtents,
                         /* requires */ (sizeof...(InputExtents) > std::size_t(1)))
MDSPAN_INLINE_FUNCTION constexpr auto pad_extents_right(
  const stdex::extents<IndexType, InputExtents...>& input,
  const stdex::extents<IndexType, PaddingExtent> padding)
{
  constexpr std::size_t rank = sizeof...(InputExtents);
  return details::pad_extents_right_helper(
    input, padding, details::iota_index_sequence_t<0, rank - 1>{});
}

MDSPAN_TEMPLATE_REQUIRES(class IndexType,
                         std::size_t... InputExtents,
                         /* requires */ (sizeof...(InputExtents) != std::size_t(0)))
MDSPAN_INLINE_FUNCTION constexpr auto pre_padded_extent_left(
  const stdex::extents<IndexType, InputExtents...>& input)
{
  using input_type = stdex::extents<IndexType, InputExtents...>;
  return stdex::extents<IndexType, input_type::static_extent(0)>{input.extent(0)};
}

MDSPAN_TEMPLATE_REQUIRES(class IndexType,
                         std::size_t... InputExtents,
                         /* requires */ (sizeof...(InputExtents) != std::size_t(0)))
MDSPAN_INLINE_FUNCTION constexpr auto pre_padded_extent_right(
  const stdex::extents<IndexType, InputExtents...>& input)
{
  using input_type = stdex::extents<IndexType, InputExtents...>;
  const auto rank  = input_type::rank();
  return stdex::extents<IndexType, input_type::static_extent(rank - 1)>{input.extent(rank - 1)};
}

template <class IndexType>
MDSPAN_INLINE_FUNCTION constexpr auto pre_padded_extent_left(
  const stdex::extents<IndexType>& /* input */)
{
  return stdex::extents<IndexType>{};
}

template <class IndexType>
MDSPAN_INLINE_FUNCTION constexpr auto pre_padded_extent_right(
  const stdex::extents<IndexType>& /* input */)
{
  return stdex::extents<IndexType>{};
}
}  // end namespace details

// TODO (mfh 2022/08/30) Private inheritance from layout_left::mapping
// resp. layout_right::mapping would reduce inlining depth.

// layout_left_padded is like layout_left,
// except that stride(0) == 1 always,
// and the leftmost extent may be padded
// (so that stride(1) could possibly be greater than extent(0)).
//
// This layout exists for two reasons:
//
// 1. Appropriate choice of padding, plus use of overaligned memory,
//    can ensure any desired power-of-two overalignment of the
//    beginning of each contiguous segment of elements in an mdspan.
//    This is useful for hardware that optimizes for overaligned
//    access.
//
// 2. For rank-2 mdspan, this is exactly the layout supported by the
//    BLAS and LAPACK (where the "leading dimension" of the matrix
//    (LDA), i.e., the stride, is greater than or equal to the number
//    of rows).
//
// The padding can be either a compile-time value or a run-time value.
// It is a template parameter of layout_left_padded (the "tag type"),
// and NOT of the mapping, because mdspan requires that the mapping be
// a metafunction of the tag type and the extents specialization type.
template <std::size_t Padding = stdex::dynamic_extent>
struct layout_left_padded {
  static constexpr size_t padding = Padding;
  template <class Extents>
  class mapping {
   public:
    using extents_type = Extents;
    using index_type   = typename extents_type::index_type;
    using size_type    = typename extents_type::size_type;
    using rank_type    = typename extents_type::rank_type;
    using layout_type  = layout_left_padded<Padding>;

   private:
    using padding_extents_type = stdex::extents<index_type, Padding>;
    using inner_layout_type    = stdex::layout_left;
    using inner_extents_type   = decltype(
      details::pad_extents_left(std::declval<Extents>(), std::declval<padding_extents_type>()));
    using inner_mapping_type = typename inner_layout_type::template mapping<inner_extents_type>;
    using pre_padding_extents_type =
      decltype(details::pre_padded_extent_left(std::declval<extents_type>()));

    inner_mapping_type inner_mapping_;
    pre_padding_extents_type pre_padded_extent_;

    struct unusable_tag_t {
    };
    static constexpr unusable_tag_t unusable_tag{};

   public:
    // mapping constructor that takes ONLY an extents_type.
    //
    // This constructor ONLY EXISTS if padding is known
    // at compile time, that is, if (the template parameter)
    // padding != dynamic_extent.
    //
    // This constructor makes it possible to construct an mdspan
    // from a pointer and extents, since that requires that
    // the mapping be constructible from extents alone.
    template <class T = unusable_tag_t>
    MDSPAN_INLINE_FUNCTION constexpr mapping(
      const extents_type& ext,
      std::enable_if_t<Padding != stdex::dynamic_extent and std::is_same_v<T, unusable_tag_t>,
                       unusable_tag_t> = unusable_tag)
      : inner_mapping_(details::pad_extents_left(ext, padding_extents_type{Padding})),
        pre_padded_extent_(details::pre_padded_extent_left(ext))
    {
    }

    // mapping constructor that takes an extents_type,
    // AND an integral padding_value.
    //
    // This constructor always exists, even if padding is known at
    // compile time -- just like the extents constructor lets you pass
    // in all rank() extents, even if some of them are known at
    // compile time.
    //
    // If padding is NOT known at compile time
    // (i.e., the template parameter padding equals dynamic_extent),
    // then you MUST pass it into the constructor,
    // either as an integral value (the first constructor below),
    // or as an extents object of the right type
    // (the second constructor below).
    template <
      class Size,
      std::enable_if_t<Padding == stdex::dynamic_extent && std::is_integral_v<Size>, Size> = 0>
    MDSPAN_INLINE_FUNCTION constexpr mapping(const extents_type& ext, Size padding_value = 0)
      : inner_mapping_(details::pad_extents_left(ext, padding_extents_type{padding_value})),
        pre_padded_extent_(details::pre_padded_extent_left(ext))
    {
      // We don't have to check padding_value here, because the
      // padding_extents_type constructor already has a precondition.
    }

    // It's always legal to pass in the padding as an extents object.
    MDSPAN_INLINE_FUNCTION
    mapping(const extents_type& ext, const padding_extents_type& padding_extents)
      : inner_mapping_(details::pad_extents_left(ext, padding_extents)),
        pre_padded_extent_(details::pre_padded_extent_left(ext))
    {
    }

    // layout_stride::mapping deliberately only defines the copy
    // constructor and copy assignment operator, not the move
    // constructor or move assignment operator.  This is fine because
    // all the storage is std::array-like; there's no advantage to
    // move construction or move assignment.  We imitate this.
    MDSPAN_INLINE_FUNCTION_DEFAULTED
    constexpr mapping(const mapping&) noexcept                                       = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED mapping& operator=(
      const mapping&) noexcept = default;

    MDSPAN_INLINE_FUNCTION
    constexpr extents_type extents() const noexcept
    {
      return details::layout_left_extents<extents_type>(pre_padded_extent_,
                                                        inner_mapping_.extents());
    }

    MDSPAN_INLINE_FUNCTION
    constexpr std::array<index_type, extents_type::rank()> strides() const noexcept
    {
      return inner_mapping_.strides();
    }

    MDSPAN_INLINE_FUNCTION
    constexpr index_type required_span_size() const noexcept
    {
      return inner_mapping_.required_span_size();
    }

    MDSPAN_TEMPLATE_REQUIRES(
      class... Indices,
      /* requires */ (
        sizeof...(Indices) == Extents::rank() &&
        _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(std::is_convertible, Indices, index_type) /*&& ...*/) &&
        _MDSPAN_FOLD_AND(
          _MDSPAN_TRAIT(std::is_nothrow_constructible, index_type, Indices) /*&& ...*/)))
    MDSPAN_INLINE_FUNCTION
    constexpr size_t operator()(Indices... idxs) const noexcept
    {
      // TODO (mfh 2022/08/30) in debug mode, check precondition before forwarding to inner mapping.
      return inner_mapping_(std::forward<Indices>(idxs)...);
    }

    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept { return true; }
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_exhaustive() noexcept
    {
      return extents_type::rank() == 0
               ? true
               : (extents_type::static_extent(0) != stdex::dynamic_extent &&
                  extents_type::static_extent(0) == pre_padding_extents_type::static_extent(0));
    }
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept { return true; }

    MDSPAN_INLINE_FUNCTION static constexpr bool is_unique() noexcept { return true; }
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 bool is_exhaustive() const noexcept
    {
      return extents_type::rank() == 0 ? true
                                       : inner_mapping_.extent(0) == pre_padded_extent_.extent(0);
    }
    MDSPAN_INLINE_FUNCTION static constexpr bool is_strided() noexcept { return true; }

    MDSPAN_INLINE_FUNCTION
    constexpr index_type stride(rank_type r) const noexcept { return inner_mapping_.stride(r); }
  };
};

template <std::size_t Padding = stdex::dynamic_extent>
struct layout_right_padded {
  static constexpr size_t padding = Padding;
  template <class Extents>
  class mapping {
   public:
    using extents_type = Extents;
    using index_type   = typename extents_type::index_type;
    using size_type    = typename extents_type::size_type;
    using rank_type    = typename extents_type::rank_type;
    using layout_type  = layout_right_padded<Padding>;

   private:
    using padding_extents_type = stdex::extents<index_type, Padding>;
    using inner_layout_type    = stdex::layout_right;
    using inner_extents_type   = decltype(
      details::pad_extents_right(std::declval<Extents>(), std::declval<padding_extents_type>()));
    using inner_mapping_type = typename inner_layout_type::template mapping<inner_extents_type>;
    using pre_padding_extents_type =
      decltype(details::pre_padded_extent_right(std::declval<extents_type>()));

    inner_mapping_type inner_mapping_;
    pre_padding_extents_type pre_padded_extent_;

    struct unusable_tag_t {
    };
    static constexpr unusable_tag_t unusable_tag{};

   public:
    // mapping constructor that takes ONLY an extents_type.
    //
    // This constructor ONLY EXISTS if padding is known
    // at compile time, that is, if (the template parameter)
    // padding != dynamic_extent.
    //
    // This constructor makes it possible to construct an mdspan
    // from a pointer and extents, since that requires that
    // the mapping be constructible from extents alone.
    template <class T = unusable_tag_t>
    MDSPAN_INLINE_FUNCTION constexpr mapping(
      const extents_type& ext,
      std::enable_if_t<Padding != stdex::dynamic_extent and std::is_same_v<T, unusable_tag_t>,
                       unusable_tag_t> = unusable_tag)
      : inner_mapping_(details::pad_extents_right(ext, padding_extents_type{Padding})),
        pre_padded_extent_(details::pre_padded_extent_right(ext))
    {
    }

    // mapping constructor that takes an extents_type,
    // AND an integral padding_value.
    //
    // This constructor always exists, even if padding is known at
    // compile time -- just like the extents constructor lets you pass
    // in all rank() extents, even if some of them are known at
    // compile time.
    //
    // If padding is NOT known at compile time
    // (i.e., the template parameter padding equals dynamic_extent),
    // then you MUST pass it into the constructor,
    // either as an integral value (the first constructor below),
    // or as an extents object of the right type
    // (the second constructor below).

    template <
      class Size,
      std::enable_if_t<Padding == stdex::dynamic_extent && std::is_integral_v<Size>, Size> = 0>
    MDSPAN_INLINE_FUNCTION constexpr mapping(const extents_type& ext, Size padding_value = 0)
      : inner_mapping_(details::pad_extents_right(ext, padding_extents_type{padding_value})),
        pre_padded_extent_(details::pre_padded_extent_right(ext))
    {
      // We don't have to check padding_value here, because the
      // padding_extents_type constructor already has a precondition.
    }

    // It's always legal to pass in the padding as an extents object.
    MDSPAN_INLINE_FUNCTION
    mapping(const extents_type& ext, const padding_extents_type& padding_extents)
      : inner_mapping_(details::pad_extents_right(ext, padding_extents)),
        pre_padded_extent_(details::pre_padded_extent_right(ext))
    {
    }

    // layout_stride::mapping deliberately only defines the copy
    // constructor and copy assignment operator, not the move
    // constructor or move assignment operator.  This is fine because
    // all the storage is std::array-like; there's no advantage to
    // move construction or move assignment.  We imitate this.
    MDSPAN_INLINE_FUNCTION_DEFAULTED
    constexpr mapping(const mapping&) noexcept                                       = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED mapping& operator=(
      const mapping&) noexcept = default;

    MDSPAN_INLINE_FUNCTION
    constexpr extents_type extents() const noexcept
    {
      return details::layout_right_extents<extents_type>(inner_mapping_.extents(),
                                                         pre_padded_extent_);
    }

    MDSPAN_INLINE_FUNCTION
    constexpr std::array<index_type, extents_type::rank()> strides() const noexcept
    {
      return inner_mapping_.strides();
    }

    MDSPAN_INLINE_FUNCTION
    constexpr index_type required_span_size() const noexcept
    {
      return inner_mapping_.required_span_size();
    }

    MDSPAN_TEMPLATE_REQUIRES(
      class... Indices,
      /* requires */ (
        sizeof...(Indices) == Extents::rank() &&
        _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(std::is_convertible, Indices, index_type) /*&& ...*/) &&
        _MDSPAN_FOLD_AND(
          _MDSPAN_TRAIT(std::is_nothrow_constructible, index_type, Indices) /*&& ...*/)))
    MDSPAN_INLINE_FUNCTION
    constexpr size_t operator()(Indices... idxs) const noexcept
    {
      // TODO (mfh 2022/08/30) in debug mode, check precondition before forwarding to inner mapping.
      return inner_mapping_(std::forward<Indices>(idxs)...);
    }

    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept { return true; }
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_exhaustive() noexcept
    {
      return extents_type::rank() == 0
               ? true
               : (extents_type::static_extent(Extents::rank() - 1) != stdex::dynamic_extent &&
                  extents_type::static_extent(Extents::rank() - 1) ==
                    pre_padding_extents_type::static_extent(0));
    }
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept { return true; }

    MDSPAN_INLINE_FUNCTION static constexpr bool is_unique() noexcept { return true; }
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 bool is_exhaustive() const noexcept
    {
      return extents_type::rank() == 0
               ? true
               : inner_mapping_.extent(Extents::rank() - 1) == pre_padded_extent_.extent(0);
    }
    MDSPAN_INLINE_FUNCTION static constexpr bool is_strided() noexcept { return true; }

    MDSPAN_INLINE_FUNCTION
    constexpr index_type stride(rank_type r) const noexcept { return inner_mapping_.stride(r); }
  };
};

template <class ElementType, size_t Alignment = 128>
struct aligned_accessor {
  using offset_policy               = aligned_accessor;
  using element_type                = ElementType;
  using reference                   = ElementType&;
  using data_handle_type            = ElementType*;
  static constexpr size_t alignment = Alignment;

  constexpr aligned_accessor() noexcept = default;

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType,
    size_t OtherAlignment,
    /* requires */
    (_MDSPAN_TRAIT(is_convertible,
                   typename aligned_accessor<OtherElementType, OtherAlignment>::element_type (*)[],
                   element_type (*)[]) &&
     alignment == OtherAlignment))
  MDSPAN_INLINE_FUNCTION
  constexpr aligned_accessor(aligned_accessor<OtherElementType, OtherAlignment>) noexcept {}

  MDSPAN_INLINE_FUNCTION
  constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept { return p + i; }

  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference access(data_handle_type p, size_t i) const noexcept
  {
#if defined(__cpp_lib_assume_aligned)
    return (std::assume_aligned<alignment>(p))[i];
#elif defined(__GNUC__)
    return reinterpret_cast<data_handle_type>(__builtin_assume_aligned(p, alignment))[i];
#else
    return p[i];
#endif
  }
};

template <std::size_t padding, StorageOrderType order>
using layout_padded_general = std::conditional_t<order == StorageOrderType::row_major_t,
                                                 layout_right_padded<padding>,
                                                 layout_left_padded<padding>>;

}  // end namespace experimental
}  // end namespace std
