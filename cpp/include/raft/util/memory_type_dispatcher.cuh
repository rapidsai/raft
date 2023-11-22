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
#include <raft/core/mdbuffer.cuh>
#include <raft/core/mdspan.hpp>
#include <raft/core/memory_type.hpp>
#include <type_traits>
#include <utility>

namespace raft {

namespace detail {

template <typename lambda_t, typename arg_t, typename = void>
struct is_callable : std::false_type {};

template <typename lambda_t, typename arg_t>
struct is_callable<lambda_t,
                   arg_t,
                   std::void_t<decltype(std::declval<lambda_t>()(std::declval<arg_t>()))>>
  : std::true_type {};

template <typename lambda_t,
          typename mdbuffer_type,
          memory_type mem_type,
          enable_if_mdbuffer<mdbuffer_type>* = nullptr>
auto static constexpr is_callable_for_memory_type =
  detail::is_callable<lambda_t,
                      decltype(std::declval<mdbuffer_type>().template view<mem_type>())>::value;

}  // namespace detail

inline void foo(host_matrix_view<double> view) { std::cout << view.extent(0) << std::endl; }

template <typename lambda_t, typename mdbuffer_type, enable_if_mdbuffer<mdbuffer_type>* = nullptr>
decltype(auto) memory_type_dispatcher(raft::resources const& res, lambda_t&& f, mdbuffer_type&& buf)
{
  if (is_host_device_accessible(buf.mem_type())) {
    // First see if functor has been specialized for this exact memory type
    if constexpr (detail::
                    is_callable_for_memory_type<lambda_t, mdbuffer_type, memory_type::managed>) {
      if (buf.mem_type() == memory_type::managed) {
        return f(buf.template view<memory_type::managed>());
      }
    } else if constexpr (detail::is_callable_for_memory_type<lambda_t,
                                                             mdbuffer_type,
                                                             memory_type::pinned>) {
      if (buf.mem_type() == memory_type::pinned) {
        return f(buf.template view<memory_type::pinned>());
      }
    }
  }
  // If the functor is specialized for device and the data are
  // device-accessible, use the device specialization
  if constexpr (detail::is_callable_for_memory_type<lambda_t, mdbuffer_type, memory_type::device>) {
    if (is_device_accessible(buf.mem_type())) {
      return f(mdbuffer{res, buf, memory_type::device}.template view<memory_type::device>());
    }
    // If there is no host specialization, still use the device specialization
    if constexpr (!detail::
                    is_callable_for_memory_type<lambda_t, mdbuffer_type, memory_type::host>) {
      return f(mdbuffer{res, buf, memory_type::device}.template view<memory_type::device>());
    }
  }

  // If nothing else has worked, use the host specialization
  if constexpr (detail::is_callable_for_memory_type<lambda_t, mdbuffer_type, memory_type::host>) {
    return f(mdbuffer{res, buf, memory_type::host}.template view<memory_type::host>());
  }

  // In the extremely rare case that the functor has been specialized _only_
  // for either pinned memory, managed memory, or both, and the input data are
  // neither pinned nor managed, we must perform a copy. In this situation, if
  // we have specializations for both pinned and managed memory, we arbitrarily
  // prefer the managed specialization. Note that if the data _are_ either
  // pinned or managed already, we will have already invoked the correct
  // specialization above.
  if constexpr (detail::
                  is_callable_for_memory_type<lambda_t, mdbuffer_type, memory_type::managed>) {
    return f(mdbuffer{res, buf, memory_type::managed}.template view<memory_type::managed>());
  } else if constexpr (detail::is_callable_for_memory_type<lambda_t,
                                                           mdbuffer_type,
                                                           memory_type::pinned>) {
    return f(mdbuffer{res, buf, memory_type::pinned}.template view<memory_type::pinned>());
  }

  // Suppress warning for unreachable loop. In general, it is a desirable thing
  // for this to be unreachable, but some functors may be specialized in such a
  // way that this is not the case.
#pragma nv_diag_suppress 128
  RAFT_FAIL("The given functor could not be invoked on the provided data");
#pragma nv_diag_default 128
}

template <typename lambda_t, typename mdspan_type, enable_if_mdspan<mdspan_type>* = nullptr>
decltype(auto) memory_type_dispatcher(raft::resources const& res, lambda_t&& f, mdspan_type md)
{
  return memory_type_dispatcher(res, std::forward<lambda_t>(f), mdbuffer{md});
}

template <typename mdbuffer_type,
          typename lambda_t,
          typename mdspan_type,
          enable_if_mdbuffer<mdbuffer_type>* = nullptr,
          enable_if_mdspan<mdspan_type>*     = nullptr>
decltype(auto) memory_type_dispatcher(raft::resources const& res, lambda_t&& f, mdspan_type md)
{
  return memory_type_dispatcher(res, std::forward<lambda_t>(f), mdbuffer_type{res, mdbuffer{md}});
}

}  // namespace raft
