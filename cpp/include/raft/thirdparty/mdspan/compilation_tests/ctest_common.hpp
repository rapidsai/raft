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

#include "experimental/__p0009_bits/macros.hpp"

#include <type_traits>
#include <cassert>

#pragma once

#define MDSPAN_ENABLE_EXPENSIVE_COMPILATION_TESTS 0

#define MDSPAN_STATIC_TEST(...) \
  static_assert(__VA_ARGS__, "MDSpan compile time test failed at "  __FILE__ ":" MDSPAN_PP_STRINGIFY(__LINE__))

//==============================================================================
// <editor-fold desc="assert-like macros that don't break constexpr"> {{{1

// The basic idea: if we do something in a constexpr context that's not allowed
// (like access an array out of bounds), the compiler should give us an error.
// At least in the case of clang, that error is descriptive enough to see what's
// going on if we're careful with how we do things.

#if _MDSPAN_USE_CONSTEXPR_14

// A nice marker in the compiler output. Also, use std::is_constant_evaluated if we have it
#if __cpp_lib_is_constant_evaluated > 201811
#  define __________CONSTEXPR_ASSERTION_FAILED__________ \
     /* do something we can't do in constexpr like access an array out of bounds */ \
     if(std::is_constant_evaluated()) return __str[-!(checker(__val, _exp))]; \
     else { assert(checker(__val, _exp) && __str); return 0; }
#else
#  define __________CONSTEXPR_ASSERTION_FAILED__________  \
    /* try to protect from bad memory access... */ \
    char a[] = { 0 }; char b[] = { 0 }; char c[] = { 0 }; \
    return b[(a[0] + c[0] + -!(checker(__val, _exp)))];
#endif

// More sugar around the compiler output to print some values of things
struct _____constexpr_assertion_failed_ {
  const char* _expr_string;
  template <class T, class F>
  struct _expected_impl {
    T _exp;
    F checker;
    const char* __str;
    template <class U>
    constexpr char but_actual_value_was_(U __val) const {
      // Put this macro here so that failures are easy to find in compiler output
      __________CONSTEXPR_ASSERTION_FAILED__________
    }
  };
  struct _check_eq {
    template <class T, class U>
    constexpr bool operator()(T val, U exp) const { return val == exp; }
  };
  struct _check_not_eq {
    template <class T, class U>
    constexpr bool operator()(T val, U exp) const { return val != exp; }
  };
  struct _check_less {
    template <class T, class U>
    constexpr bool operator()(T val, U exp) const { return val != exp; }
  };
  struct _check_greater {
    template <class T, class U>
    constexpr bool operator()(T val, U exp) const { return val != exp; }
  };

  template <class T>
  constexpr auto _expected_to_be_true() const {
    return _expected_impl<T, _check_eq>{true, _check_eq{}, _expr_string};
  }
  template <class T>
  constexpr auto _rhs_expected_to_be_(T _exp) const {
    return _expected_impl<T, _check_eq>{_exp, _check_eq{}, _expr_string};
  }
  template <class T>
  constexpr auto _rhs_expected_to_not_be_(T _exp) const {
    return _expected_impl<T, _check_not_eq>{_exp, _check_not_eq{}, _expr_string};
  }
  template <class T>
  constexpr auto _expected_to_be_less_than_(T _exp) const {
    return _expected_impl<T, _check_less>{_exp, _check_less{}, _expr_string};
  }
  template <class T>
  constexpr auto _expected_to_be_greater_than_(T _exp) const {
    return _expected_impl<T, _check_greater>{_exp, _check_greater{}, _expr_string};
  }
};

// Macros for the assertions themselves
#define constexpr_assert(...) \
  _____constexpr_assertion_failed_{#__VA_ARGS__}._expected_to_be_(true).but_actual_value_was_(__VA_ARGS__);

#define constexpr_assert_equal(expr, ...) \
  _____constexpr_assertion_failed_{#__VA_ARGS__ "==" #expr}._rhs_expected_to_be_(expr).but_actual_value_was_((__VA_ARGS__));

#define constexpr_assert_not_equal(expr, ...) \
  _____constexpr_assertion_failed_{#__VA_ARGS__ "!=" #expr}._rhs_expected_to_not_be_(expr).but_actual_value_was_((__VA_ARGS__));

#define constexpr_assert_less_than(expr, ...) \
  _____constexpr_assertion_failed_{#__VA_ARGS__}._expected_to_be_less_than_(expr).but_actual_value_was_((__VA_ARGS__));

#define constexpr_assert_greater_than(expr, ...) \
  _____constexpr_assertion_failed_{#__VA_ARGS__}._expected_to_be_greater_than_(expr).but_actual_value_was_((__VA_ARGS__));

#endif // _MDSPAN_USE_CONSTEXPR_14

// </editor-fold> end assert-like macros that don't break constexpr }}}1
//==============================================================================

// All tests need a main so that they'll link
int main() { }
