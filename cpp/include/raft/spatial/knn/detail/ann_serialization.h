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
    std::cout << "Written " << (sizeof value) << " bytes" << std::endl;
  } else {
    std::cerr << "error writing value to file" << std::endl;
  }
}

template <typename T>
T read_scalar(std::ifstream& file)
{
  T value;
  file.read((char*)&value, sizeof value);
  if (file.good()) {
    std::cout << "Read " << (sizeof value) << " bytes" << std::endl;
  } else {
    std::cerr << "error reading value from file" << std::endl;
  }
  return value;
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void write_mdspan(
    const raft::handle_t& handle,
    std::ofstream& of,
    const raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj) {
  using obj_t = raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
  write_scalar(of, obj.rank());
  if (obj.is_exhaustive()) {
    write_scalar(of, obj.size());
  } else {
    std::cerr << "Cannot serialize non exhaustive mdarray" << std::endl;
    write_scalar<size_t>(of, 0);
  }
  if (obj.size() > 0) {
    for (typename obj_t::rank_type i = 0; i < obj.rank(); i++) write_scalar(of, obj.extent(i));
    cudaStream_t stream = handle.get_stream();
    std::vector<
        typename raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>::
            value_type>
        tmp(obj.size());
    raft::update_host(tmp.data(), obj.data_handle(), obj.size(), stream);
    handle.sync_stream(stream);
    of.write(reinterpret_cast<char*>(tmp.data()), tmp.size() * sizeof(ElementType));
    if (of.good()) {
      std::cout << "Written " << obj.size() * sizeof(obj.data_handle()[0]) << " bytes"
                << std::endl;
    }
    else {
      std::cerr << "error writing mdarray to file" << std::endl;
    }
  } else {
    std::cout << "Skipping mdspand with zero size" << std::endl;
  }
}

template<typename ElementType,
         typename Extents,
         typename LayoutPolicy,
         typename AccessorPolicy>
void read_mdspan(
    const raft::handle_t& handle,
    std::ifstream& file,
    raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj) {
  using obj_t = raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
  auto rank = read_scalar<typename obj_t::rank_type>(file);
  if (obj.rank() != rank) {
    std::cerr << "Incorrect rank while reading mdarray " << rank << " vs. " << obj.rank()
              << std::endl;
  }
  auto size = read_scalar<typename obj_t::size_type>(file);
  if (obj.size() != size) {
    std::cerr << "Incorrect size while reading mdarray " << size << " vs. " << obj.size()
              << std::endl;
  }
  if (obj.size() > 0) {
    for (typename obj_t::rank_type i = 0; i < obj.rank(); i++) {
      auto ex = read_scalar<typename obj_t::index_type>(file);
      if (obj.extent(i) != ex) {
        std::cerr << "Incorrect extent while reading mdarray " << ex << " vs. "
                  << obj.extent(i) << " at " << i << std::endl;
      }
    }
    cudaStream_t stream = handle.get_stream();
    std::vector<typename obj_t::value_type> tmp(obj.size());
    file.read(reinterpret_cast<char*>(tmp.data()), tmp.size() * sizeof(ElementType));
    raft::update_device(obj.data_handle(), tmp.data(), tmp.size(), stream);
    handle.sync_stream(stream);
    if (file.good()) {
      std::cout << "read " << obj.size() * sizeof(obj.data_handle()[0]) << " bytes"
                << std::endl;
    }
    else {
      std::cerr << "error reading mdarray from file" << std::endl;
    }
  }
  else {
    std::cout << "Skipping mdspand with zero size" << std::endl;
  }
}

template<typename ElementType,
         typename Extents,
         typename LayoutPolicy,
         typename AccessorPolicy>
void read_mdspan(
    const raft::handle_t& handle,
    std::ifstream& file,
    raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>&& obj) {
  read_mdspan(handle, file, obj);
}
}  // namespace cuann
