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

#include <raft/core/detail/mdspan_numpy_serializer.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>
#include <vector>

namespace raft {

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void serialize_mdspan(
  const raft::handle_t& handle,
  std::ostream& os,
  const raft::host_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  static_assert(std::is_same_v<LayoutPolicy, raft::layout_c_contiguous> ||
                  std::is_same_v<LayoutPolicy, raft::layout_f_contiguous>,
                "The serializer only supports row-major and column-major layouts");
  detail::numpy_serializer::serialize(handle, os, obj);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void serialize_mdspan(
  const raft::handle_t& handle,
  std::ostream& os,
  const raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  static_assert(std::is_same_v<LayoutPolicy, raft::layout_c_contiguous> ||
                  std::is_same_v<LayoutPolicy, raft::layout_f_contiguous>,
                "The serializer only supports row-major and column-major layouts");
  using obj_t = raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

  // Copy to host before serializing
  // For contiguous layouts, size() == product of dimensions
  std::vector<typename obj_t::value_type> tmp(obj.size());
  cudaStream_t stream = handle.get_stream();
  raft::update_host(tmp.data(), obj.data_handle(), obj.size(), stream);
  using inner_accessor_type = typename obj_t::accessor_type::accessor_type;
  auto tmp_mdspan =
    raft::host_mdspan<ElementType, Extents, LayoutPolicy, raft::host_accessor<inner_accessor_type>>(
      tmp, obj.extents());
  detail::numpy_serializer::serialize(handle, os, tmp_mdspan);
}

}  // end namespace raft
