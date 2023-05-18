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
#include <raft/core/resources.hpp>

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
constexpr static IdxT kInvalidRecord =
  (std::is_signed_v<IdxT> ? IdxT{0} : std::numeric_limits<IdxT>::max()) - 1;

/** The data for a single IVF list. */
template <template <typename, typename...> typename SpecT,
          typename SizeT,
          typename... SpecExtraArgs>
struct list {
  using size_type    = SizeT;
  using spec_type    = SpecT<size_type, SpecExtraArgs...>;
  using value_type   = typename spec_type::value_type;
  using index_type   = typename spec_type::index_type;
  using list_extents = typename spec_type::list_extents;

  /** Possibly encoded data; it's layout is defined by `SpecT`. */
  device_mdarray<value_type, list_extents, row_major> data;
  /** Source indices. */
  device_mdarray<index_type, extent_1d<size_type>, row_major> indices;
  /** The actual size of the content. */
  std::atomic<size_type> size;

  /** Allocate a new list capable of holding at least `n_rows` data records and indices. */
  list(raft::resources const& res, const spec_type& spec, size_type n_rows);
};

template <typename ListT, class T = void>
struct enable_if_valid_list {};

template <class T,
          template <typename, typename...>
          typename SpecT,
          typename SizeT,
          typename... SpecExtraArgs>
struct enable_if_valid_list<list<SpecT, SizeT, SpecExtraArgs...>, T> {
  using type = T;
};

/**
 * Designed after `std::enable_if_t`, this trait is helpful in the instance resolution;
 * plug this in the return type of a function that has an instance of `ivf::list` as
 * a template parameter.
 */
template <typename ListT, class T = void>
using enable_if_valid_list_t = typename enable_if_valid_list<ListT, T>::type;

}  // namespace raft::neighbors::ivf
