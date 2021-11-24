/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <iostream>
#include <limits>
#include <raft/handle.hpp>
#include <raft/sparse/coo.cuh>
#include <raft/sparse/op/reduce.cuh>
#include <rmm/device_uvector.hpp>
#include "../test_utils.h"

namespace raft {
namespace sparse {

template <typename value_t, typename value_idx>
struct SparseReduceInputs {
  std::vector<value_idx> in_rows;
  std::vector<value_idx> in_cols;
  std::vector<value_t> in_vals;
  std::vector<value_idx> out_rows;
  std::vector<value_idx> out_cols;
  std::vector<value_t> out_vals;

  size_t m;
  size_t n;
};

template <typename value_t, typename value_idx>
class SparseReduceTest
  : public ::testing::TestWithParam<SparseReduceInputs<value_t, value_idx>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<
      SparseReduceInputs<value_t, value_idx>>::GetParam();
  }

  void Run() {
    raft::handle_t handle;

    auto d_alloc = handle.get_device_allocator();
    auto stream = handle.get_stream();

    rmm::device_uvector<value_idx> in_rows(params.in_rows.size(), stream);
    rmm::device_uvector<value_idx> in_cols(params.in_cols.size(), stream);
    rmm::device_uvector<value_t> in_vals(params.in_vals.size(), stream);
    rmm::device_uvector<value_idx> out_rows(params.out_rows.size(), stream);
    rmm::device_uvector<value_idx> out_cols(params.out_cols.size(), stream);
    rmm::device_uvector<value_t> out_vals(params.out_vals.size(), stream);

    raft::update_device(in_rows.data(), params.in_rows.data(),
                        params.in_rows.size(), stream);
    raft::update_device(in_cols.data(), params.in_cols.data(),
                        params.in_cols.size(), stream);
    raft::update_device(in_vals.data(), params.in_vals.data(),
                        params.in_vals.size(), stream);
    raft::update_device(out_rows.data(), params.out_rows.data(),
                        params.out_rows.size(), stream);
    raft::update_device(out_cols.data(), params.out_cols.data(),
                        params.out_cols.size(), stream);
    raft::update_device(out_vals.data(), params.out_vals.data(),
                        params.out_vals.size(), stream);

    raft::sparse::COO<value_t, value_idx> out(d_alloc, stream);
    raft::sparse::op::max_duplicates(handle, out, in_rows.data(),
                                     in_cols.data(), in_vals.data(),
                                     params.in_rows.size(), params.m, params.n);

    ASSERT_TRUE(raft::devArrMatch<value_idx>(
      out_rows.data(), out.rows(), out.nnz, raft::Compare<value_idx>()));
    ASSERT_TRUE(raft::devArrMatch<value_idx>(
      out_cols.data(), out.cols(), out.nnz, raft::Compare<value_idx>()));
    ASSERT_TRUE(raft::devArrMatch<value_t>(out_vals.data(), out.vals(), out.nnz,
                                           raft::Compare<value_t>()));
  }

  void TearDown() override {}

 protected:
  SparseReduceInputs<value_t, value_idx> params;
  value_idx *in_rows, *in_cols, *out_rows, *out_cols;
  value_t *in_vals, *out_vals;
};

using SparseReduceTestF = SparseReduceTest<float, int>;
TEST_P(SparseReduceTestF, Result) { Run(); }

// Max reduce expects COO to be sorted already
const std::vector<SparseReduceInputs<float, int>> max_reduce_inputs_f = {
  {// input rows/cols/vals
   {0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 3},
   {1, 1, 1, 2, 0, 3, 3, 0, 2, 3, 3},
   {3.0, 50.0, 0.0, 2.0, 40.0, 2.0, 1.0, 4.0, 1.0, 0.0, 30.0},

   // output rows/cols/vals
   {0, 0, 1, 1, 2, 3, 3},
   {1, 2, 0, 3, 0, 2, 3},
   {50.0, 2.0, 40.0, 2.0, 4.0, 1.0, 30.0},
   4,
   4},
};

INSTANTIATE_TEST_CASE_P(SparseReduceTest, SparseReduceTestF,
                        ::testing::ValuesIn(max_reduce_inputs_f));

}  // namespace sparse
}  // namespace raft
