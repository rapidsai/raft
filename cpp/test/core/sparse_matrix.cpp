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
#include <raft/core/host_coo_matrix.hpp>
#include <raft/core/host_csr_matrix.hpp>
#include <type_traits>

namespace raft {

/**
 * Example of accepting a value-owning matrix type which doesn't need to adjust sparsity
 */
template <typename S, typename = std::enable_if_t<is_host_csr_matrix_v<S>>>
bool test_host_csr_owning_ref(S& mat)
{
  using IndptrType = typename S::structure_type::indptr_type;

  IndptrType indptr_t = 50;

  std::cout << "csr owning: " << static_cast<void*>(mat.get_elements().data()) << indptr_t
            << std::endl;
  mat.structure_view();
  return true;
}

template <typename S, typename = std::enable_if_t<is_host_csr_sparsity_owning_v<S>>>
bool test_host_csr_sparsity_owning_ref(S& mat)
{
  std::cout << "Sparsity owning value address: " << static_cast<void*>(mat.get_elements().data())
            << std::endl;
  mat.structure_view();
  return true;
}

template <typename S, typename = std::enable_if_t<is_host_csr_sparsity_preserving_v<S>>>
bool test_host_csr_sparsity_preserving_ref(S& mat)
{
  std::cout << "Sparsity preserving value address: "
            << static_cast<void*>(mat.get_elements().data()) << std::endl;
  mat.structure_view();
  return true;
}

// just simple integration test, main tests are in mdspan ref implementation.
void test_host_coo_matrix()
{
  raft::resources handle;
  auto mat = raft::make_host_coo_matrix<float, int, int, int>(handle, 5, 5);

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

void test_host_csr_matrix()
{
  raft::resources handle;
  auto sparsity_owning = raft::make_host_csr_matrix<float, int, int, int>(handle, 5, 5);

  auto comp_struct = raft::make_host_compressed_structure(handle, 5, 5, 5);
  auto sparsity_preserving =
    raft::make_host_csr_matrix<float, int, int>(handle, comp_struct.view());

  auto structure_view = sparsity_owning.structure_view();
  //  auto sparse_view = mat.view();

  ASSERT_EQ(structure_view.get_n_cols(), 5);
  ASSERT_EQ(structure_view.get_n_rows(), 5);
  ASSERT_EQ(structure_view.get_nnz(), 0);

  sparsity_owning.initialize_sparsity(5);

  auto structure_view2 = sparsity_owning.structure_view();
  //  auto sparse_view2 = mat.view();

  std::cout << "Value address: " << static_cast<void*>(sparsity_owning.get_elements().data())
            << std::endl;

  test_host_csr_owning_ref(sparsity_owning);
  test_host_csr_owning_ref(sparsity_preserving);

  test_host_csr_sparsity_owning_ref(sparsity_owning);
  test_host_csr_sparsity_preserving_ref(sparsity_preserving);

  ASSERT_EQ(structure_view2.get_n_cols(), 5);
  ASSERT_EQ(structure_view2.get_n_rows(), 5);
  ASSERT_EQ(structure_view2.get_nnz(), 5);
}

TEST(HostSparseMatrix, Basic)
{
  test_host_csr_matrix();
  test_host_coo_matrix();
}
}  // namespace raft
