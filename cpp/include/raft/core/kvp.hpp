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

#include <raft/core/detail/macros.hpp>

#ifdef _RAFT_HAS_CUDA
#include <cub/cub.cuh>
#endif
namespace raft {
/**
 * \brief A key identifier paired with a corresponding value
 *
 */
template <typename _Key, typename _Value>
struct KeyValuePair {
  typedef _Key Key;      ///< Key data type
  typedef _Value Value;  ///< Value data type

  Key key;      ///< Item key
  Value value;  ///< Item value

  /// Constructor
  RAFT_INLINE_FUNCTION KeyValuePair() {}

#ifdef _RAFT_HAS_CUDA
  /// Conversion Constructor to allow integration w/ cub
  RAFT_INLINE_FUNCTION KeyValuePair(cub::KeyValuePair<_Key, _Value> kvp)
    : key(kvp.key), value(kvp.value)
  {
  }

  RAFT_INLINE_FUNCTION operator cub::KeyValuePair<_Key, _Value>()
  {
    return cub::KeyValuePair(key, value);
  }
#endif

  /// Constructor
  RAFT_INLINE_FUNCTION KeyValuePair(Key const& key, Value const& value) : key(key), value(value) {}

  /// Inequality operator
  RAFT_INLINE_FUNCTION bool operator!=(const KeyValuePair& b)
  {
    return (value != b.value) || (key != b.key);
  }
};
}  // end namespace raft
