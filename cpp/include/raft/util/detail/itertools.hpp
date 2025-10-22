/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
