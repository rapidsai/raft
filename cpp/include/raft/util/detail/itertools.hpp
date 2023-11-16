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

#include <tuple>
#include <vector>

namespace raft::util::itertools::detail {

template <class S, typename... Args, size_t... Is>
inline std::vector<S> product(std::index_sequence<Is...> index, const std::vector<Args>&... vecs)
{
  size_t len = 1;
  ((len *= vecs.size()), ...);
  std::vector<S> out;
  out.reserve(len);
  for (size_t i = 0; i < len; i++) {
    std::tuple<Args...> tup;
    size_t mod = len, new_mod;
    ((new_mod = mod / vecs.size(), std::get<Is>(tup) = vecs[(i % mod) / new_mod], mod = new_mod),
     ...);
    out.push_back({std::get<Is>(tup)...});
  }
  return out;
}

}  // namespace raft::util::itertools::detail
