/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/detail/distance.cuh>
#include <raft/distance/distance_type.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace raft::spatial::knn::detail {

template <typename T>
void write_scalar(std::ofstream& of, const T& value)
{
  of.write((char*)&value, sizeof value);
  if (of.good()) {
    RAFT_LOG_DEBUG("Written %z bytes", (sizeof value));
  } else {
    RAFT_FAIL("error writing value to file");
  }
}

template <typename T>
T read_scalar(std::ifstream& file)
{
  T value;
  file.read((char*)&value, sizeof value);
  if (file.good()) {
    RAFT_LOG_DEBUG("Read %z bytes", (sizeof value));
  } else {
    RAFT_FAIL("error reading value from file");
  }
  return value;
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void write_mdspan(
  const raft::handle_t& handle,
  std::ofstream& of,
  const raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  using obj_t = raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
  write_scalar(of, obj.rank());
  if (obj.is_exhaustive() && obj.is_unique()) {
    write_scalar(of, obj.size());
  } else {
    RAFT_FAIL("Cannot serialize non exhaustive mdarray");
  }
  if (obj.size() > 0) {
    for (typename obj_t::rank_type i = 0; i < obj.rank(); i++)
      write_scalar(of, obj.extent(i));
    cudaStream_t stream = handle.get_stream();
    std::vector<
      typename raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>::value_type>
      tmp(obj.size());
    raft::update_host(tmp.data(), obj.data_handle(), obj.size(), stream);
    handle.sync_stream(stream);
    of.write(reinterpret_cast<char*>(tmp.data()), tmp.size() * sizeof(ElementType));
    if (of.good()) {
      RAFT_LOG_DEBUG("Written %zu bytes",
                     static_cast<size_t>(obj.size() * sizeof(obj.data_handle()[0])));
    } else {
      RAFT_FAIL("Error writing mdarray to file");
    }
  } else {
    RAFT_LOG_DEBUG("Skipping mdspand with zero size");
  }
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void read_mdspan(const raft::handle_t& handle,
                 std::ifstream& file,
                 raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  using obj_t = raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
  auto rank   = read_scalar<typename obj_t::rank_type>(file);
  if (obj.rank() != rank) { RAFT_FAIL("Incorrect rank while reading mdarray"); }
  auto size = read_scalar<typename obj_t::size_type>(file);
  if (obj.size() != size) {
    RAFT_FAIL("Incorrect rank while reading mdarray %zu vs %zu",
              static_cast<size_t>(size),
              static_cast<size_t>(obj.size()));
  }
  if (obj.size() > 0) {
    for (typename obj_t::rank_type i = 0; i < obj.rank(); i++) {
      auto ex = read_scalar<typename obj_t::index_type>(file);
      if (obj.extent(i) != ex) {
        RAFT_FAIL("Incorrect extent while reading mdarray %d vs %d at %d",
                  static_cast<int>(ex),
                  static_cast<int>(obj.extent(i)),
                  static_cast<int>(i));
      }
    }
    cudaStream_t stream = handle.get_stream();
    std::vector<typename obj_t::value_type> tmp(obj.size());
    file.read(reinterpret_cast<char*>(tmp.data()), tmp.size() * sizeof(ElementType));
    raft::update_device(obj.data_handle(), tmp.data(), tmp.size(), stream);
    handle.sync_stream(stream);
    if (file.good()) {
      RAFT_LOG_DEBUG("Read %zu bytes",
                     static_cast<size_t>(obj.size() * sizeof(obj.data_handle()[0])));
    } else {
      RAFT_FAIL("error reading mdarray from file");
    }
  } else {
    RAFT_LOG_DEBUG("Skipping mdspand with zero size");
  }
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void read_mdspan(const raft::handle_t& handle,
                 std::ifstream& file,
                 raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>&& obj)
{
  read_mdspan(handle, file, obj);
}
}  // namespace raft::spatial::knn::detail
