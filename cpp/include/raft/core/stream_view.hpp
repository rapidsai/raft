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
#include <raft/core/cuda_support.hpp>
#include <raft/core/error.hpp>
#include <raft/core/logger.hpp>
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/interruptible.hpp>
#include <rmm/cuda_stream_view.hpp>
#endif

namespace raft {

namespace detail {
struct fail_stream_view {
  constexpr fail_stream_view()                                           = default;
  constexpr fail_stream_view(fail_stream_view const&)                    = default;
  constexpr fail_stream_view(fail_stream_view&&)                         = default;
  auto constexpr operator=(fail_stream_view const&) -> fail_stream_view& = default;
  auto constexpr operator=(fail_stream_view&&) -> fail_stream_view&      = default;
  auto value() { throw non_cuda_build_error{"Attempted to access CUDA stream in non-CUDA build"}; }
  [[nodiscard]] auto is_per_thread_default() const { return false; }
  [[nodiscard]] auto is_default() const { return false; }
  void synchronize() const
  {
    throw non_cuda_build_error{"Attempted to sync CUDA stream in non-CUDA build"};
  }
  void synchronize_no_throw() const
  {
    RAFT_LOG_ERROR("Attempted to sync CUDA stream in non-CUDA build");
  }
};
}  // namespace detail

/** A lightweight wrapper around rmm::cuda_stream_view that can be used in
 * CUDA-free builds
 *
 * While CUDA-free builds should never actually make use of a CUDA stream at
 * runtime, it is sometimes useful to have a symbol that can stand in place of
 * a CUDA stream to avoid excessive ifdef directives interspersed with other
 * logic. This struct's methods invoke the underlying rmm::cuda_stream_view in
 * CUDA-enabled builds but throw runtime exceptions if any non-trivial method
 * is called from a CUDA-free build */
struct stream_view {
#ifndef RAFT_DISABLE_CUDA
  using underlying_view_type = rmm::cuda_stream_view;
#else
  using underlying_view_type = detail::fail_stream_view;
#endif

  constexpr stream_view(
    underlying_view_type base_view = stream_view::get_underlying_per_thread_default())
    : base_view_{base_view}
  {
  }
  constexpr stream_view(stream_view const&)          = default;
  constexpr stream_view(stream_view&&)               = default;
  auto operator=(stream_view const&) -> stream_view& = default;
  auto operator=(stream_view&&) -> stream_view&      = default;
  auto value() { return base_view_.value(); }
  operator underlying_view_type() const noexcept { return base_view_; }
  [[nodiscard]] auto is_per_thread_default() const { return base_view_.is_per_thread_default(); }
  [[nodiscard]] auto is_default() const { return base_view_.is_default(); }
  void synchronize() const { base_view_.synchronize(); }
  void synchronize_no_throw() const { base_view_.synchronize_no_throw(); }
  void interruptible_synchronize() const
  {
#ifndef RAFT_DISABLE_CUDA
    interruptible::synchronize(base_view_);
#else
    synchronize();
#endif
  }

  auto underlying() { return base_view_; }
  void synchronize_if_cuda_enabled()
  {
    if constexpr (raft::CUDA_ENABLED) { base_view_.synchronize(); }
  }

 private:
  underlying_view_type base_view_;
  auto static get_underlying_per_thread_default() -> underlying_view_type
  {
#ifndef RAFT_DISABLE_CUDA
    return rmm::cuda_stream_per_thread;
#else
    auto static constexpr const default_fail_stream = underlying_view_type{};
    return default_fail_stream;
#endif
  }
};

auto static const stream_view_per_thread = stream_view{};

}  // namespace raft
