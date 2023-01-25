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

}  // namespace raft::spatial::knn::detail
