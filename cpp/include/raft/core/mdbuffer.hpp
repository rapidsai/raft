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
#include <raft/core/error.hpp>
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
}

template <typename T>
struct fail_container {
  using pointer = T*;
  using const_pointer = T const*;

  using reference = T&;
  using const_reference = T const&;

  using iterator = pointer;
  using const_iterator = const_pointer;

  explicit fail_container(size_t n=size_t{}) {
    if (n != size_t{}) {
      throw non_cuda_build_error{
        "Attempted to allocate device container in non-CUDA build"
      };
    }
  }
};

template <typename ElementType>
struct fail_container_policy {
  using element_type = ElementType;
  using container_type = fail_container<element_type>;
  using pointer         = typename container_type::pointer;
  using const_pointer   = typename container_type::const_pointer;
};

namespace detail {
template<typename ElementType>
using default_buffer_host_policy = host_vector_policy<ElementType>;

#ifdef RAFT_DISABLE_CUDA
#else
template<typename ElementType>
using default_buffer_device_policy = device_uvector_policy<ElementType>;
#endif
}

template <
  typename ElementType
>
struct default_buffer_container_policy {
  using element_type = ElementType;
  using container_policy_variant = std::variant<
    device_uvector_policy<element_type>,
    host_vector_policy<element_type>
  >;
};

template <
  typename ElementType,
  typename Extents,
  typename LayoutPolicy = layout_c_contiguous,
  typename ContainerPolicy
struct mdbuffer {
};

}
