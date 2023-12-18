/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "../hnsw_types.hpp"
#include "hnsw_types.hpp"

#include <cstddef>
#include <cstdint>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/serialize.hpp>
#include <raft/neighbors/cagra_types.hpp>

#include <fstream>
#include <type_traits>

namespace raft::neighbors::hnsw::detail {

template <typename T>
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 index<T>*& index,
                 int dim,
                 raft::distance::DistanceType metric)
{
  index = new index_impl<T>(filename, dim, metric);
  RAFT_EXPECTS(index, "Could not set index pointer");
}

}  // namespace raft::neighbors::hnsw::detail
