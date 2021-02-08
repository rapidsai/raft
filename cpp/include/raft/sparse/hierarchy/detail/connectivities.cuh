/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>

#include <raft/linalg/distance_type.h>
#include <raft/mr/device/buffer.hpp>

#include <limits>

namespace raft {
namespace hierarchy {
namespace detail {

template <LinkageDistance dist_type, typename value_idx, typename value_t>
struct distance_graph_impl {
  void run(const raft::handle_t &handle, const value_t *X, size_t m, size_t n,
           raft::distance::DistanceType metric,
           raft::mr::device::buffer<value_idx> &indptr,
           raft::mr::device::buffer<value_idx> &indices,
           raft::mr::device::buffer<value_t> &data, int c);
};

template <typename value_idx, typename value_t, LinkageDistance dist_type>
void get_distance_graph(const raft::handle_t &handle, const value_t *X,
                        size_t m, size_t n, raft::distance::DistanceType metric,
                        raft::mr::device::buffer<value_idx> &indptr,
                        raft::mr::device::buffer<value_idx> &indices,
                        raft::mr::device::buffer<value_t> &data, int c) {
  auto stream = handle.get_stream();

  indptr.resize(m + 1, stream);

  distance_graph_impl<dist_type, value_idx, value_t> dist_graph;
  dist_graph.run(handle, X, m, n, metric, indptr, indices, data, c);

  // a little adjustment for distances of 0.
  // TODO: This will only need to be done when src_v==dst_v
  raft::linalg::unaryOp<value_t>(
    data.data(), data.data(), data.size(),
    [] __device__(value_t input) {
      if (input == 0)
        return std::numeric_limits<value_t>::max();
      else
        return input;
    },
    stream);
}

}  // namespace detail
}  // namespace hierarchy
}  // namespace raft