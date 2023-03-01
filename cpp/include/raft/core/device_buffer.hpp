/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

template <typename ElementType,
          typename Extents,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = detail::device_uvector_policy<std::remove_cv_t<ElementType>>>
class device_buffer {
  using view_type            = device_mdspan<ElementType, Extents, LayoutPolicy>;
  using index_type           = typename Extents::index_type;
  using element_type         = std::remove_cv_t<ElementType>;
  using owning_device_buffer = device_mdarray<element_type, Extents, LayoutPolicy, ContainerPolicy>;
  using data_store           = std::variant<ElementType*, owning_device_buffer>;

 public:
  device_buffer(device_buffer const&) = delete;
  device_buffer& operator=(device_buffer const&) = delete;

  constexpr device_buffer(device_buffer&&) = default;
  constexpr device_buffer& operator=(device_buffer&&) = default;

  device_buffer(device_resources const& handle,
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

  ~device_buffer()
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
  static constexpr bool is_const_pointer_ = std::is_const_v<ElementType>;
  rmm::cuda_stream_view stream_;
  ElementType* original_data_;
  data_store data_;
  Extents extents_;
  bool write_back_;
  std::size_t length_;
  int device_id_;
};

}  // namespace raft
