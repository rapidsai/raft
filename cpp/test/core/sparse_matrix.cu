/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <raft/core/device_resources.hpp>
#include <raft/core/device_sparse_matrix.hpp>

namespace raft {

// just simple integration test, main tests are in mdspan ref implementation.
void test_sparse_matrix()
{
  raft::device_resources handle;
  auto mat = raft::make_coo_matrix<float, int, int, int>(handle, 5, 5);

  auto structure_view = mat.structure_view();
  //  auto sparse_view = mat.view();

  ASSERT_EQ(structure_view.get_n_cols(), 5);
  ASSERT_EQ(structure_view.get_n_rows(), 5);
  ASSERT_EQ(structure_view.get_nnz(), 0);

  mat.initialize_sparsity(5);

  auto structure_view2 = mat.structure_view();
  //  auto sparse_view2 = mat.view();

  ASSERT_EQ(structure_view2.get_n_cols(), 5);
  ASSERT_EQ(structure_view2.get_n_rows(), 5);
  ASSERT_EQ(structure_view2.get_nnz(), 5);
}

TEST(SparseMatrix, Basic) { test_sparse_matrix(); }

}  // namespace raft
