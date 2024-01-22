/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/detail/select_radix.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/matrix/select_k.cuh>

namespace raft::matrix::select {

struct params {
  size_t batch_size;
  size_t len;
  int k;
  bool select_min;
  bool use_index_input       = true;
  bool use_same_leading_bits = false;
  bool use_memory_pool       = true;
  double frac_infinities     = 0.0;
};

inline auto operator<<(std::ostream& os, const params& ss) -> std::ostream&
{
  os << "params{batch_size: " << ss.batch_size;
  os << ", len: " << ss.len;
  os << ", k: " << ss.k;
  os << (ss.select_min ? ", asc" : ", dsc");
  if (!ss.use_index_input) { os << ", no-input-index"; }
  if (ss.use_same_leading_bits) { os << ", same-leading-bits"; }
  if (ss.frac_infinities > 0) { os << ", infs: " << ss.frac_infinities; }
  os << "}";
  return os;
}
}  // namespace raft::matrix::select
