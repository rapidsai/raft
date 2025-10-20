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


// NOTE: This code is prematurely taken from an example based on
// https://github.com/kokkos/mdspan/pull/176

#pragma once

#include "macros.hpp"
#include "trait_backports.hpp"
#include "default_accessor.hpp"
#include "extents.hpp"
#include <cassert>
#include <iostream>
#include <type_traits>

namespace std {
namespace experimental {

namespace stdex = std::experimental;


// Prefer std::assume_aligned if available, as it is in the C++ Standard.
// Otherwise, use a compiler-specific equivalent if available.

// NOTE (mfh 2022/08/08) BYTE_ALIGNMENT must be unsigned and a power of 2.
#if defined(__cpp_lib_assume_aligned)
#  define _MDSPAN_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) (std::assume_aligned< BYTE_ALIGNMENT >( POINTER ))
  constexpr char assume_aligned_method[] = "std::assume_aligned";
#elif defined(__ICL)
#  define _MDSPAN_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) POINTER
  constexpr char assume_aligned_method[] = "(none)";
#elif defined(__ICC)
#  define _MDSPAN_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) POINTER
  constexpr char assume_aligned_method[] = "(none)";
#elif defined(__clang__)
#  define _MDSPAN_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) POINTER
  constexpr char assume_aligned_method[] = "(none)";
#elif defined(__GNUC__)
  // __builtin_assume_aligned returns void*
#  define _MDSPAN_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) reinterpret_cast< ELEMENT_TYPE* >(__builtin_assume_aligned( POINTER, BYTE_ALIGNMENT ))
  constexpr char assume_aligned_method[] = "__builtin_assume_aligned";
#else
#  define _MDSPAN_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) POINTER
  constexpr char assume_aligned_method[] = "(none)";
#endif

// Some compilers other than Clang or GCC like to define __clang__ or __GNUC__.
// Thus, we order the tests from most to least specific.
#if defined(__ICL)
#  define _MDSPAN_ALIGN_VALUE_ATTRIBUTE( BYTE_ALIGNMENT ) __declspec(align_value( BYTE_ALIGNMENT ));
  constexpr char align_attribute_method[] = "__declspec(align_value(BYTE_ALIGNMENT))";
#elif defined(__ICC)
#  define _MDSPAN_ALIGN_VALUE_ATTRIBUTE( BYTE_ALIGNMENT ) __attribute__((align_value( BYTE_ALIGNMENT )));
  constexpr char align_attribute_method[] = "__attribute__((align_value(BYTE_ALIGNMENT)))";
#elif defined(__clang__)
#  define _MDSPAN_ALIGN_VALUE_ATTRIBUTE( BYTE_ALIGNMENT ) __attribute__((align_value( BYTE_ALIGNMENT )));
  constexpr char align_attribute_method[] = "__attribute__((align_value(BYTE_ALIGNMENT)))";
#else
#  define _MDSPAN_ALIGN_VALUE_ATTRIBUTE( BYTE_ALIGNMENT )
  constexpr char align_attribute_method[] = "(none)";
#endif

constexpr bool
is_nonzero_power_of_two(const std::size_t x)
{
// Just checking __cpp_lib_int_pow2 isn't enough for some GCC versions.
// The <bit> header exists, but std::has_single_bit does not.
#if defined(__cpp_lib_int_pow2) && __cplusplus >= 202002L
  return std::has_single_bit(x);
#else
  return x != 0 && (x & (x - 1)) == 0;
#endif
}

template<class ElementType>
constexpr bool
valid_byte_alignment(const std::size_t byte_alignment)
{
  return is_nonzero_power_of_two(byte_alignment) && byte_alignment >= alignof(ElementType);
}

// We define aligned_pointer_t through a struct
// so we can check whether the byte alignment is valid.
// This makes it impossible to use the alias
// with an invalid byte alignment.
template<class ElementType, std::size_t byte_alignment>
struct aligned_pointer {
  static_assert(valid_byte_alignment<ElementType>(byte_alignment),
		"byte_alignment must be a power of two no less than "
		"the minimum required alignment of ElementType.");
  using type = ElementType* _MDSPAN_ALIGN_VALUE_ATTRIBUTE( byte_alignment );
};


template<class ElementType, std::size_t byte_alignment>
using aligned_pointer_t = typename aligned_pointer<ElementType, byte_alignment>::type;

template<class ElementType, std::size_t byte_alignment>
aligned_pointer_t<ElementType, byte_alignment>
bless(ElementType* ptr, std::integral_constant<std::size_t, byte_alignment> /* ba */ )
{
  return _MDSPAN_ASSUME_ALIGNED( ElementType, ptr, byte_alignment );
}


template<class ElementType, std::size_t byte_alignment>
struct aligned_accessor {
  using offset_policy = stdex::default_accessor<ElementType>;
  using element_type = ElementType;
  using reference = ElementType&;
  using data_handle_type = aligned_pointer_t<ElementType, byte_alignment>;

  constexpr aligned_accessor() noexcept = default;

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType,
    std::size_t other_byte_alignment,
    /* requires */ (std::is_convertible<OtherElementType(*)[], element_type(*)[]>::value && other_byte_alignment == byte_alignment)
    )
  constexpr aligned_accessor(aligned_accessor<OtherElementType, other_byte_alignment>) noexcept {}

  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    // This may declare alignment twice, depending on
    // if we have an attribute for marking pointer types.
    return _MDSPAN_ASSUME_ALIGNED( ElementType, p, byte_alignment )[i];
  }

  constexpr typename offset_policy::data_handle_type
  offset(data_handle_type p, size_t i) const noexcept {
    return p + i;
  }
};

template<class ElementType>
struct delete_raw {
  void operator()(ElementType* p) const {
    if (p != nullptr) {
      // All the aligned allocation methods below go with std::free.
      // If we implement a new method that uses a different
      // deallocation function, that function would go here.
      std::free(p);
    }
  }
};

}  // end namespace experimental
}  // end namespace std
