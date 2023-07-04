/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <raft/core/resources.hpp>

namespace raft {
namespace sparse {
namespace distance {
namespace detail {

template <typename value_idx, typename value_t>
struct distances_config_t {
  distances_config_t(raft::resources const& handle_) : handle(handle_) {}

  // left side
  value_idx a_nrows;
  value_idx a_ncols;
  value_idx a_nnz;
  value_idx* a_indptr;
  value_idx* a_indices;
  value_t* a_data;

  // right side
  value_idx b_nrows;
  value_idx b_ncols;
  value_idx b_nnz;
  value_idx* b_indptr;
  value_idx* b_indices;
  value_t* b_data;

  raft::resources const& handle;
};

template <typename value_t>
class distances_t {
 public:
  virtual void compute(value_t* out) {}
  virtual ~distances_t() = default;
};

};  // namespace detail
};  // namespace distance
};  // namespace sparse
};  // namespace raft