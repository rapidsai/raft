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

#include "device_mdarray.hpp"
#include "device_mdspan.hpp"

#include <raft/util/cudart_utils.hpp>

#include <variant>

namespace raft {

/**
 * @brief An object to have temporary representation of either a host/device pointer in device
 * memory. This object provides a `view()` method that will provide a `raft::device_mdspan` that may
 * be read-only depending on const-qualified nature of the input pointer.
 *
 * @tparam ElementType type of the input
 * @tparam Extents raft::extents
 * @tparam LayoutPolicy layout of the input
 * @tparam ContainerPolicy container to be used to own device memory if needed
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = detail::device_uvector_policy<std::remove_cv_t<ElementType>>>
class temporary_device_buffer {
  using view_type            = device_mdspan<ElementType, Extents, LayoutPolicy>;
  using index_type           = typename Extents::index_type;
  using element_type         = std::remove_cv_t<ElementType>;
  using owning_device_buffer = device_mdarray<element_type, Extents, LayoutPolicy, ContainerPolicy>;
  using data_store           = std::variant<ElementType*, owning_device_buffer>;
  static constexpr bool is_const_pointer_ = std::is_const_v<ElementType>;

 public:
  temporary_device_buffer(temporary_device_buffer const&) = delete;
  temporary_device_buffer& operator=(temporary_device_buffer const&) = delete;

  constexpr temporary_device_buffer(temporary_device_buffer&&) = default;
  constexpr temporary_device_buffer& operator=(temporary_device_buffer&&) = default;

  /**
   * @brief Construct a new temporary device buffer object
   *
   * @param handle raft::device_resources
   * @param data input pointer
   * @param extents dimensions of input array
   * @param write_back if true, any writes to the `view()` of this object will be copid
   *                   back if the original pointer was in host memory
   */
  temporary_device_buffer(device_resources const& handle,
                          ElementType* data,
                          Extents extents,
                          bool write_back = false)
    : stream_(handle.get_stream()),
      original_data_(data),
      extents_{extents},
      write_back_(write_back),
      length_([this]() {
        std::size_t length = 1;
        for (std::size_t i = 0; i < extents_.rank(); ++i) {
          length *= extents_.extent(i);
        }
        return length;
      }()),
      device_id_{get_device_for_address(data)}
  {
    if (device_id_ == -1) {
      typename owning_device_buffer::mapping_type layout{extents_};
      typename owning_device_buffer::container_policy_type policy{handle.get_stream()};

      owning_device_buffer device_data{layout, policy};
      raft::copy(device_data.data_handle(), data, length_, handle.get_stream());
      data_ = data_store{std::in_place_index<1>, std::move(device_data)};
    } else {
      data_ = data_store{std::in_place_index<0>, data};
    }
  }

  ~temporary_device_buffer() noexcept(is_const_pointer_)
  {
    // only need to write data back for non const pointers
    // when write_back=true and original pointer is in
    // host memory
    if constexpr (not is_const_pointer_) {
      if (write_back_ && device_id_ == -1) {
        raft::copy(original_data_, std::get<1>(data_).data_handle(), length_, stream_);
      }
    }
  }

  /**
   * @brief Returns a `raft::device_mdspan`
   *
   * @return raft::device_mdspan
   */
  auto view() -> view_type
  {
    if (device_id_ == -1) {
      return std::get<1>(data_).view();
    } else {
      return make_mdspan<ElementType, index_type, LayoutPolicy, false, true>(original_data_,
                                                                             extents_);
    }
  }

 private:
  rmm::cuda_stream_view stream_;
  ElementType* original_data_;
  data_store data_;
  Extents extents_;
  bool write_back_;
  std::size_t length_;
  int device_id_;
};

/**
 * @brief Factory to create a `raft::temporary_device_buffer`
 *
 * @tparam ElementType type of the input
 * @tparam IndexType index type of `raft::extents`
 * @tparam LayoutPolicy layout of the input
 * @tparam ContainerPolicy container to be used to own device memory if needed
 * @tparam Extents variadic dimensions for `raft::extents`
 * @param handle raft::device_resources
 * @param data input pointer
 * @param extents dimensions of input array
 * @param write_back if true, any writes to the `view()` of this object will be copid
 *                   back if the original pointer was in host memory
 * @return raft::temporary_device_buffer
 */
template <typename ElementType,
          typename IndexType       = std::uint32_t,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = detail::device_uvector_policy<std::remove_cv_t<ElementType>>,
          size_t... Extents>
auto make_temporary_device_buffer(raft::device_resources const& handle,
                                  ElementType* data,
                                  raft::extents<IndexType, Extents...> extents,
                                  bool write_back = false)
{
  return temporary_device_buffer<ElementType, decltype(extents), LayoutPolicy, ContainerPolicy>(
    handle, data, extents, write_back);
}

/**
 * @brief Factory to create a `raft::temporary_device_buffer` which produces a
 *        read-only `raft::device_mdspan` from `view()` method with
 *        `write_back=false`
 *
 * @tparam ElementType type of the input
 * @tparam IndexType index type of `raft::extents`
 * @tparam LayoutPolicy layout of the input
 * @tparam ContainerPolicy container to be used to own device memory if needed
 * @tparam Extents variadic dimensions for `raft::extents`
 * @param handle raft::device_resources
 * @param data input pointer
 * @param extents dimensions of input array
 * @return raft::temporary_device_buffer
 */
template <typename ElementType,
          typename IndexType       = std::uint32_t,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = detail::device_uvector_policy<std::remove_cv_t<ElementType>>,
          size_t... Extents>
auto make_readonly_temporary_device_buffer(raft::device_resources const& handle,
                                           ElementType* data,
                                           raft::extents<IndexType, Extents...> extents)
{
  return temporary_device_buffer<const ElementType,
                                 decltype(extents),
                                 LayoutPolicy,
                                 ContainerPolicy>(handle, data, extents, false);
}

/**
 * @brief Factory to create a `raft::temporary_device_buffer` which produces a
 *        writeable `raft::device_mdspan` from `view()` method with
 *        `write_back=true`
 *
 * @tparam ElementType type of the input
 * @tparam IndexType index type of `raft::extents`
 * @tparam LayoutPolicy layout of the input
 * @tparam ContainerPolicy container to be used to own device memory if needed
 * @tparam Extents variadic dimensions for `raft::extents`
 * @param handle raft::device_resources
 * @param data input pointer
 * @param extents dimensions of input array
 * @return raft::temporary_device_buffer
 */
template <typename ElementType,
          typename IndexType       = std::uint32_t,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = detail::device_uvector_policy<ElementType>,
          size_t... Extents,
          typename = std::enable_if_t<not std::is_const_v<ElementType>>>
auto make_writeback_temporary_device_buffer(raft::device_resources const& handle,
                                            ElementType* data,
                                            raft::extents<IndexType, Extents...> extents)
{
  return temporary_device_buffer<ElementType, decltype(extents), LayoutPolicy, ContainerPolicy>(
    handle, data, extents, true);
}

}  // namespace raft
