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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>

#include <atomic>
#include <limits>
#include <type_traits>

namespace raft::neighbors::ivf {

/**
 * Default value filled in the `indices` array.
 * One may encounter it trying to access a record within a list that is outside of the
 * `size` bound or whenever the list is allocated but not filled-in yet.
 */
template <typename IdxT>
constexpr static IdxT kInvalidRecord = (std::is_signed_v<IdxT> ? IdxT{0}
                                                               : std::numeric_limits<IdxT>::max()) -
                                       1;

/** The data for a single IVF list. */
template <template <typename> typename SpecT, typename IdxT, typename SizeT = uint32_t>
struct list {
  using value_type   = typename SpecT<SizeT>::value_type;
  using list_extents = typename SpecT<SizeT>::list_extents;

  /** Possibly encoded data; it's layout is defined by `SpecT`. */
  device_mdarray<value_type, list_extents, row_major> data;
  /** Source indices. */
  device_mdarray<IdxT, extent_1d<SizeT>, row_major> indices;
  /** The actual size of the content. */
  std::atomic<SizeT> size;

  /** Allocate a new list capable of holding at least `n_rows` data records and indices. */
  list(raft::device_resources const& res, const SpecT<SizeT>& spec, SizeT n_rows);
};

}  // namespace raft::neighbors::ivf
