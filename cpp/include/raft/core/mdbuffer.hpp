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

#include <variant>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/core/logger-macros.hpp>

namespace raft {

namespace detail {
#ifdef RAFT_DISABLE_CUDA
using buffer_stream_view = rmm::cuda_stream_view;
#else
struct buffer_stream_view {
  auto value() const {
    throw non_cuda_build_error{
      "Attempted to access CUDA stream in non-CUDA build"
    };
  }
  [[nodiscard]] auto is_per_thread_default() const {
    throw non_cuda_build_error{
      "Attempted to access CUDA stream in non-CUDA build"
    };
    return false;
  }
  [[nodiscard]] auto is_default() const {
    throw non_cuda_build_error{
      "Attempted to access CUDA stream in non-CUDA build"
    };
    return false;
  }
  void synchronize() const {
    throw non_cuda_build_error{
      "Attempted to sync CUDA stream in non-CUDA build"
    };
  }

  void synchronize_no_throw() const {
    RAFT_LOG_ERROR(
      "Attempted to sync CUDA stream in non-CUDA build"
    );
  }
};
#endif
}  // namespace detail

inline auto constexpr variant_index_from_memory_type(raft::memory_type mem_type) {
  return static_cast<std::underlying_type_t<raft::memory_type>>(mem_type);
}

template <typename Variant, raft::memory_type MemType>
using alternate_from_mem_type = std::variant_alternative_t<variant_index_from_memory_type(MemType), Variant>;

template <
  typename ElementType
>
struct default_buffer_container_policy {
  using element_type = ElementType;
  using value_type      = std::remove_cv_t<element_type>;
  using container_policy_variant = std::variant<
    host_vector_policy<element_type>,
    device_uvector_policy<element_type>,
    managed_uvector_policy<element_type>,
    pinned_vector_policy<element_type>
  >;

  template <raft::memory_type MemType>
  using underlying_policy = alternate_from_mem_type<container_policy_variant, MemType>;
};

template <typename ContainerPolicy>
struct universal_buffer_reference {
  using value_type = typename ContainerPolicy::value_type;
  using pointer = typename ContainerPolicy::value_type*;
  using const_pointer = typename ContainerPolicy::value_type const*;

  using reference_variant = std::variant<
    typename ContainerPolicy::template underlying_policy<raft::memory_type::host>::reference,
    typename ContainerPolicy::template underlying_policy<raft::memory_type::device>::reference,
    typename ContainerPolicy::template underlying_policy<raft::memory_type::managed>::reference,
    typename ContainerPolicy::template underlying_policy<raft::memory_type::pinned>::reference
  >;
  using const_reference_variant = std::variant<
    typename ContainerPolicy::template underlying_policy<raft::memory_type::host>::const_reference,
    typename ContainerPolicy::template underlying_policy<raft::memory_type::device>::const_reference,
    typename ContainerPolicy::template underlying_policy<raft::memory_type::managed>::const_reference,
    typename ContainerPolicy::template underlying_policy<raft::memory_type::pinned>::const_reference
  >;

  universal_buffer_reference(pointer ptr, raft::memory_type mem_type)
    : ptr_{ptr}, mem_type_{mem_type}
  {
  }
 private:
  pointer ptr_;
  raft::memory_type mem_type_;

};

template <
  typename ElementType,
  typename Extents,
  typename LayoutPolicy = layout_c_contiguous,
  typename ContainerPolicy = default_buffer_container_policy<ElementType>
> struct mdbuffer {
  using extents_type = Extents;
  using layout_type  = LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;
  using element_type = ElementType;

  using value_type      = std::remove_cv_t<element_type>;
  using index_type      = typename extents_type::index_type;
  using difference_type = std::ptrdiff_t;
  using rank_type       = typename extents_type::rank_type;

  using owning_container_variant = std::variant<
    mdarray<ElementType
  >;
};

}  // namespace raft
