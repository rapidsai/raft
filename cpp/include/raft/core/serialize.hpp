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
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>

#include <iostream>
#include <vector>

/**
 * Collection of serialization functions for RAFT data types
 */

namespace raft {

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void serialize_mdspan(
  const raft::device_resources& handle,
  std::ostream& os,
  const raft::host_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  static_assert(std::is_same_v<LayoutPolicy, raft::layout_c_contiguous> ||
                  std::is_same_v<LayoutPolicy, raft::layout_f_contiguous>,
                "The serializer only supports row-major and column-major layouts");
  detail::numpy_serializer::serialize(handle, os, obj);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void serialize_mdspan(
  const raft::device_resources& handle,
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
      tmp.data(), obj.extents());
  detail::numpy_serializer::serialize(handle, os, tmp_mdspan);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void deserialize_mdspan(
  const raft::device_resources& handle,
  std::istream& is,
  raft::host_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  static_assert(std::is_same_v<LayoutPolicy, raft::layout_c_contiguous> ||
                  std::is_same_v<LayoutPolicy, raft::layout_f_contiguous>,
                "The serializer only supports row-major and column-major layouts");
  detail::numpy_serializer::deserialize(handle, is, obj);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void deserialize_mdspan(
  const raft::device_resources& handle,
  std::istream& is,
  raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  static_assert(std::is_same_v<LayoutPolicy, raft::layout_c_contiguous> ||
                  std::is_same_v<LayoutPolicy, raft::layout_f_contiguous>,
                "The serializer only supports row-major and column-major layouts");
  using obj_t = raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

  // Copy to device after serializing
  // For contiguous layouts, size() == product of dimensions
  std::vector<typename obj_t::value_type> tmp(obj.size());
  using inner_accessor_type = typename obj_t::accessor_type::accessor_type;
  auto tmp_mdspan =
    raft::host_mdspan<ElementType, Extents, LayoutPolicy, raft::host_accessor<inner_accessor_type>>(
      tmp.data(), obj.extents());
  detail::numpy_serializer::deserialize(handle, is, tmp_mdspan);

  cudaStream_t stream = handle.get_stream();
  raft::update_device(obj.data_handle(), tmp.data(), obj.size(), stream);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void deserialize_mdspan(
  const raft::device_resources& handle,
  std::istream& is,
  raft::host_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>&& obj)
{
  deserialize_mdspan(handle, is, obj);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void deserialize_mdspan(
  const raft::device_resources& handle,
  std::istream& is,
  raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>&& obj)
{
  deserialize_mdspan(handle, is, obj);
}

template <typename T>
void serialize_scalar(const raft::device_resources& handle, std::ostream& os, const T& value)
{
  using mdspan_1d_c_layout =
    raft::host_mdspan<const T, raft::dextents<std::size_t, 1>, raft::layout_c_contiguous>;
  auto tmp_mdspan = mdspan_1d_c_layout(&value, 1);
  serialize_mdspan(handle, os, tmp_mdspan);
}

template <typename T>
T deserialize_scalar(const raft::device_resources& handle, std::istream& is)
{
  T value;
  using mdspan_1d_c_layout =
    raft::host_mdspan<T, raft::dextents<std::size_t, 1>, raft::layout_c_contiguous>;
  auto tmp_mdspan = mdspan_1d_c_layout(&value, 1);
  deserialize_mdspan(handle, is, tmp_mdspan);
  return value;
}

}  // end namespace raft
