/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/detail/macros.hpp>
#include <raft/core/mdspan_types.hpp>

#include <limits>  // numeric_limits
#include <type_traits>

namespace raft {

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
      Count != dynamic_extent ? Count : (Extent != dynamic_extent ? Extent - Offset : Extent)> {};

/*!
 * If N is dynamic_extent, the extent of the returned span E is also
 * dynamic_extent; otherwise it is std::size_t(sizeof(T)) * N.
 */
template <typename T, std::size_t Extent>
struct extent_as_bytes_value_t
  : public std::integral_constant<std::size_t,
                                  Extent == dynamic_extent ? Extent : sizeof(T) * Extent> {};

template <std::size_t From, std::size_t To>
struct is_allowed_extent_conversion_t
  : public std::integral_constant<bool,
                                  From == To || From == dynamic_extent || To == dynamic_extent> {};

template <class From, class To>
struct is_allowed_element_type_conversion_t
  : public std::integral_constant<bool, std::is_convertible<From (*)[], To (*)[]>::value> {};

template <class T>
struct is_span_oracle_t : std::false_type {};

template <class T, bool is_device, std::size_t Extent>
struct is_span_oracle_t<span<T, is_device, Extent>> : std::true_type {};

template <class T>
struct is_span_t : public is_span_oracle_t<typename std::remove_cv<T>::type> {};

template <class InputIt1, class InputIt2, class Compare>
_RAFT_HOST_DEVICE constexpr auto lexicographical_compare(InputIt1 first1,
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

template <typename T, std::size_t Extent>
struct span_storage {
 private:
  T* ptr_{nullptr};

 public:
  constexpr span_storage() noexcept = default;
  constexpr span_storage(T* ptr, std::size_t) noexcept : ptr_{ptr} {}
  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t { return Extent; }
  [[nodiscard]] constexpr auto data() const noexcept -> T* { return ptr_; }
};

template <typename T>
struct span_storage<T, dynamic_extent> {
 private:
  T* ptr_{nullptr};
  std::size_t size_{0};

 public:
  constexpr span_storage() noexcept = default;
  constexpr span_storage(T* ptr, std::size_t size) noexcept : ptr_{ptr}, size_{size} {}
  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t { return size_; }
  [[nodiscard]] constexpr auto data() const noexcept -> T* { return ptr_; }
};
}  // namespace detail
}  // namespace raft
