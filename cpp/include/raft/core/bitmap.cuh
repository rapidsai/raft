/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/bitmap.hpp>
#include <raft/core/bitset.cuh>
#include <raft/core/detail/mdspan_util.cuh>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <type_traits>

namespace raft::core {

template <typename bitmap_t, typename index_t>
_RAFT_HOST_DEVICE inline bool bitmap_view<bitmap_t, index_t>::test(const index_t row,
                                                                   const index_t col) const
{
  return test(row * cols_ + col);
}

template <typename bitmap_t, typename index_t>
_RAFT_HOST_DEVICE void bitmap_view<bitmap_t, index_t>::set(const index_t row,
                                                           const index_t col,
                                                           bool new_value) const
{
  set(row * cols_ + col, new_value);
}

}  // end namespace raft::core
