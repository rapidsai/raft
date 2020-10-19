/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <rmm/thrust_rmm_allocator.h>
#include <iostream>

#include <raft/handle.hpp>

#include <raft/sparse/mst.cuh>

namespace raft {

TEST(Raft, MST) {
  using namespace mst;
  using vertex_t = int;
  using edge_t = int;
  using value_t = float;

  handle_t h;
  ASSERT_EQ(0, h.get_num_internal_streams());

  edge_t* offsets{nullptr};
  vertex_t* indices{nullptr};
  value_t* weights{nullptr};
  edge_t e = 0;
  vertex_t v = 0;
  rmm::device_vector<vertex_t> mst_src;
  rmm::device_vector<vertex_t> mst_dst;

  MST_solver<vertex_t, edge_t, value_t> solver(h, offsets, indices, weights, v,
                                               e);

  //nullptr expected to trigger exceptions
  EXPECT_ANY_THROW(solver.solve(mst_src, mst_dst));
}
}  // namespace raft
