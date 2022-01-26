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

#include <cassert>
#include <cinttypes>  // size_t
#include <cstddef>    // std::byte
#include <limits>     // numeric_limits
#include <type_traits>

#include <thrust/functional.h>
#include <thrust/iterator/reverse_iterator.h>

namespace raft {

constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

template <class ElementType, bool is_device, std::size_t Extent>
class span;

namespace detail {
/*!
 * The extent E of the span returned by subspan is determined as follows:
 *
 *   - If Count is not dynamic_extent, Count;
 *   - Otherwise, if Extent is not dynamic_extent, Extent - Offset;
 *   - Otherwise, dynamic_extent.
 */
template <std::size_t Extent, std::size_t Offset, std::size_t Count>
struct extent_value_t
  : public std::integral_constant<
      std::size_t,
      Count != dynamic_extent ? Count : (Extent != dynamic_extent ? Extent - Offset : Extent)> {
};

/*!
 * If N is dynamic_extent, the extent of the returned span E is also
 * dynamic_extent; otherwise it is std::size_t(sizeof(T)) * N.
 */
template <typename T, std::size_t Extent>
struct extent_as_bytes_value_t
  : public std::integral_constant<std::size_t,
                                  Extent == dynamic_extent ? Extent : sizeof(T) * Extent> {
};

template <std::size_t From, std::size_t To>
struct is_allowed_extent_conversion_t
  : public std::integral_constant<bool,
                                  From == To || From == dynamic_extent || To == dynamic_extent> {
};

template <class From, class To>
struct is_allowed_element_type_conversion_t
  : public std::integral_constant<bool, std::is_convertible<From (*)[], To (*)[]>::value>{};

template <class T>
struct is_span_oracle_t : std::false_type {
};

template <class T, bool is_device, std::size_t Extent>
struct is_span_oracle_t<span<T, is_device, Extent>> : std::true_type {
};

template <class T>
struct is_span_t : public is_span_oracle_t<typename std::remove_cv<T>::type> {
};

template <class InputIt1, class InputIt2, class Compare>
__host__ __device__ constexpr auto lexicographical_compare(InputIt1 first1,
                                                           InputIt1 last1,
                                                           InputIt2 first2,
                                                           InputIt2 last2) -> bool
{
  Compare comp;
  for (; first1 != last1 && first2 != last2; ++first1, ++first2) {
    if (comp(*first1, *first2)) { return true; }
    if (comp(*first2, *first1)) { return false; }
  }
  return first1 == last1 && first2 != last2;
}
}  // namespace detail

/**
 * @brief The span class defined in ISO C++20.  Iterator is defined as plain pointer and
 *        most of the methods have bound check on debug build.
 *
 * @code
 *   rmm::device_uvector<float> uvec(10, rmm::cuda_stream_default);
 *   auto view = device_span<float>{uvec.data(), uvec.size()};
 * @endcode
 */
template <typename T, bool is_device, std::size_t Extent = dynamic_extent>
class span {
 public:
  using element_type    = T;
  using value_type      = typename std::remove_cv<T>::type;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer         = T*;
  using const_pointer   = T const*;
  using reference       = T&;
  using const_reference = T const&;

  using iterator               = pointer;
  using const_iterator         = const_pointer;
  using reverse_iterator       = thrust::reverse_iterator<iterator>;
  using const_reverse_iterator = thrust::reverse_iterator<const_iterator>;

  /**
   * @brief Default constructor that constructs a span with size 0 and nullptr.
   */
  constexpr span() noexcept = default;

  /**
   * @brief Constructs a span that is a view over the range [first, first + count);
   */
  constexpr span(pointer ptr, size_type count) noexcept : size_(count), data_(ptr)
  {
    assert(!(Extent != dynamic_extent && count != Extent));
    assert(ptr || count == 0);
  }
  /**
   * @brief Constructs a span that is a view over the range [first, last)
   */
  constexpr span(pointer first, pointer last) noexcept : size_(last - first), data_(first)
  {
    assert(data_ || size_ == 0);
  }
  /**
   * @brief Constructs a span that is a view over the array arr.
   */
  template <std::size_t N>
  constexpr span(element_type (&arr)[N]) noexcept : size_(N), data_(&arr[0])
  {
  }

  /**
   * @brief Initialize a span class from another one who's underlying type is convertible
   *        to element_type.
   */
  template <class U,
            std::size_t OtherExtent,
            class = typename std::enable_if<
              detail::is_allowed_element_type_conversion_t<U, T>::value &&
              detail::is_allowed_extent_conversion_t<OtherExtent, Extent>::value>>
  constexpr span(const span<U, is_device, OtherExtent>& other) noexcept
    : size_(other.size()), data_(other.data())
  {
  }

  constexpr span(const span& other) noexcept : size_(other.size()), data_(other.data()) {}

  constexpr auto operator=(const span& other) noexcept -> span&
  {
    size_ = other.size();
    data_ = other.data();
    return *this;
  }

  constexpr auto begin() const noexcept -> iterator { return data(); }

  constexpr auto end() const noexcept -> iterator { return data() + size(); }

  constexpr auto cbegin() const noexcept -> const_iterator { return data(); }

  constexpr auto cend() const noexcept -> const_iterator { return data() + size(); }

  __host__ __device__ constexpr auto rbegin() const noexcept -> reverse_iterator
  {
    return reverse_iterator{end()};
  }

  __host__ __device__ constexpr auto rend() const noexcept -> reverse_iterator
  {
    return reverse_iterator{begin()};
  }

  __host__ __device__ constexpr auto crbegin() const noexcept -> const_reverse_iterator
  {
    return const_reverse_iterator{cend()};
  }

  __host__ __device__ constexpr auto crend() const noexcept -> const_reverse_iterator
  {
    return const_reverse_iterator{cbegin()};
  }

  // element access
  constexpr auto front() const -> reference { return (*this)[0]; }

  constexpr auto back() const -> reference { return (*this)[size() - 1]; }

  template <typename Index>
  constexpr auto operator[](Index _idx) const -> reference
  {
    assert(static_cast<size_type>(_idx) < size());
    return data()[_idx];
  }

  constexpr auto data() const noexcept -> pointer { return data_; }

  // Observers
  [[nodiscard]] constexpr auto size() const noexcept -> size_type { return size_; }
  [[nodiscard]] constexpr auto size_bytes() const noexcept -> size_type
  {
    return size() * sizeof(T);
  }

  constexpr auto empty() const noexcept { return size() == 0; }

  // Subviews
  template <std::size_t Count>
  constexpr auto first() const -> span<element_type, is_device, Count>
  {
    assert(Count <= size());
    return {data(), Count};
  }

  constexpr auto first(std::size_t _count) const -> span<element_type, is_device, dynamic_extent>
  {
    assert(_count <= size());
    return {data(), _count};
  }

  template <std::size_t Count>
  constexpr auto last() const -> span<element_type, is_device, Count>
  {
    assert(Count <= size());
    return {data() + size() - Count, Count};
  }

  constexpr auto last(std::size_t _count) const -> span<element_type, is_device, dynamic_extent>
  {
    assert(_count <= size());
    return subspan(size() - _count, _count);
  }

  /*!
   * If Count is std::dynamic_extent, r.size() == this->size() - Offset;
   * Otherwise r.size() == Count.
   */
  template <std::size_t Offset, std::size_t Count = dynamic_extent>
  constexpr auto subspan() const
    -> span<element_type, is_device, detail::extent_value_t<Extent, Offset, Count>::value>
  {
    assert((Count == dynamic_extent) ? (Offset <= size()) : (Offset + Count <= size()));
    return {data() + Offset, Count == dynamic_extent ? size() - Offset : Count};
  }

  constexpr auto subspan(size_type _offset, size_type _count = dynamic_extent) const
    -> span<element_type, is_device, dynamic_extent>
  {
    assert((_count == dynamic_extent) ? (_offset <= size()) : (_offset + _count <= size()));
    return {data() + _offset, _count == dynamic_extent ? size() - _offset : _count};
  }

 private:
  size_type size_{0};
  pointer data_{nullptr};
};

/**
 * @brief A span class for host pointer.
 */
template <typename T, size_t extent = dynamic_extent>
using host_span = span<T, false, extent>;

/**
 * @brief A span class for device pointer.
 */
template <typename T, size_t extent = dynamic_extent>
using device_span = span<T, true, extent>;

template <class T, std::size_t X, class U, std::size_t Y, bool is_device>
constexpr auto operator==(span<T, is_device, X> l, span<U, is_device, Y> r) -> bool
{
  if (l.size() != r.size()) { return false; }
  for (auto l_beg = l.cbegin(), r_beg = r.cbegin(); l_beg != l.cend(); ++l_beg, ++r_beg) {
    if (*l_beg != *r_beg) { return false; }
  }
  return true;
}

template <class T, std::size_t X, class U, std::size_t Y, bool is_device>
constexpr auto operator!=(span<T, is_device, X> l, span<U, is_device, Y> r)
{
  return !(l == r);
}

template <class T, std::size_t X, class U, std::size_t Y, bool is_device>
constexpr auto operator<(span<T, is_device, X> l, span<U, is_device, Y> r)
{
  return detail::lexicographical_compare<
    typename span<T, is_device, X>::iterator,
    typename span<U, is_device, Y>::iterator,
    thrust::less<typename span<T, is_device, X>::element_type>>(
    l.begin(), l.end(), r.begin(), r.end());
}

template <class T, std::size_t X, class U, std::size_t Y, bool is_device>
constexpr auto operator<=(span<T, is_device, X> l, span<U, is_device, Y> r)
{
  return !(l > r);
}

template <class T, std::size_t X, class U, std::size_t Y, bool is_device>
constexpr auto operator>(span<T, is_device, X> l, span<U, is_device, Y> r)
{
  return detail::lexicographical_compare<
    typename span<T, is_device, X>::iterator,
    typename span<U, is_device, Y>::iterator,
    thrust::greater<typename span<T, is_device, X>::element_type>>(
    l.begin(), l.end(), r.begin(), r.end());
}

template <class T, std::size_t X, class U, std::size_t Y, bool is_device>
constexpr auto operator>=(span<T, is_device, X> l, span<U, is_device, Y> r)
{
  return !(l < r);
}

/**
 * @brief Converts a span into a view of its underlying bytes
 */
template <class T, bool is_device, std::size_t E>
auto as_bytes(span<T, is_device, E> s) noexcept
  -> span<const std::byte, is_device, detail::extent_as_bytes_value_t<T, E>::value>
{
  return {reinterpret_cast<const std::byte*>(s.data()), s.size_bytes()};
}

/**
 * @brief Converts a span into a mutable view of its underlying bytes
 */
template <class T, bool is_device, std::size_t E>
auto as_writable_bytes(span<T, is_device, E> s) noexcept
  -> span<std::byte, is_device, detail::extent_as_bytes_value_t<T, E>::value>
{
  return {reinterpret_cast<std::byte*>(s.data()), s.size_bytes()};
}
}  // namespace raft
