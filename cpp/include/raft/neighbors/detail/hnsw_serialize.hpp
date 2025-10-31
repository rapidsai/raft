/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../hnsw_types.hpp"
#include "hnsw_types.hpp"

#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/serialize.hpp>
#include <raft/neighbors/cagra_types.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <type_traits>

namespace raft::neighbors::hnsw::detail {

template <typename T>
std::unique_ptr<index<T>> deserialize(raft::resources const& handle,
                                      const std::string& filename,
                                      int dim,
                                      raft::distance::DistanceType metric)
{
  return std::unique_ptr<index<T>>(new index_impl<T>(filename, dim, metric));
}

}  // namespace raft::neighbors::hnsw::detail
