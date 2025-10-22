/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <raft/core/host_coo_matrix.hpp>
#include <raft/core/host_csr_matrix.hpp>
#include <raft/core/resources.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace raft {

/**
 * Example of accepting a value-owning matrix type which doesn't need to adjust sparsity
 */
template <typename S, typename = std::enable_if_t<is_host_csr_matrix_v<S>>>
void test_host_csr_owning_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a csr_matrix which also owns structure and can initialize sparsity
 */
template <typename S, typename = std::enable_if_t<is_host_csr_sparsity_owning_v<S>>>
void test_host_csr_sparsity_owning_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a csr_matrix which does not own the structure
 */
template <typename S, typename = std::enable_if_t<is_host_csr_sparsity_preserving_v<S>>>
void test_host_csr_sparsity_preserving_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a value-owning matrix type which doesn't need to adjust sparsity
 */
template <typename S, typename = std::enable_if_t<is_host_coo_matrix_v<S>>>
void test_host_coo_owning_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a coo_matrix which also owns structure and can initialize sparsity
 */
template <typename S, typename = std::enable_if_t<is_host_coo_sparsity_owning_v<S>>>
void test_host_coo_sparsity_owning_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a coo_matrix which does not own the structure
 */
template <typename S, typename = std::enable_if_t<is_host_coo_sparsity_preserving_v<S>>>
void test_host_coo_sparsity_preserving_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

void test_host_coo_matrix()
{
  raft::resources handle;
  auto sparsity_owning = raft::make_host_coo_matrix<float, int, int, int>(handle, 5, 5);

  auto structure_view = sparsity_owning.structure_view();

  ASSERT_EQ(structure_view.get_n_cols(), 5);
  ASSERT_EQ(structure_view.get_n_rows(), 5);
  ASSERT_EQ(structure_view.get_nnz(), 0);

  auto coord_struct = raft::make_host_coordinate_structure(handle, 5, 5, 5);
  auto sparsity_preserving =
    raft::make_host_coo_matrix<float, int, int>(handle, coord_struct.view());

  sparsity_owning.initialize_sparsity(5);

  auto structure_view2 = sparsity_owning.structure_view();

  ASSERT_EQ(structure_view2.get_n_cols(), 5);
  ASSERT_EQ(structure_view2.get_n_rows(), 5);
  ASSERT_EQ(structure_view2.get_nnz(), 5);

  void* d_owning     = static_cast<void*>(sparsity_owning.get_elements().data());
  void* d_preserving = static_cast<void*>(sparsity_preserving.get_elements().data());

  test_host_coo_owning_ref(sparsity_owning, d_owning);
  test_host_coo_owning_ref(sparsity_preserving, d_preserving);

  test_host_coo_sparsity_owning_ref(sparsity_owning, d_owning);
  test_host_coo_sparsity_preserving_ref(sparsity_preserving, d_preserving);
}

void test_host_csr_matrix()
{
  raft::resources handle;
  auto sparsity_owning = raft::make_host_csr_matrix<float, int, int, int>(handle, 5, 5);

  auto comp_struct = raft::make_host_compressed_structure(handle, 5, 5, 5);
  auto sparsity_preserving =
    raft::make_host_csr_matrix<float, int, int>(handle, comp_struct.view());

  auto structure_view = sparsity_owning.structure_view();

  ASSERT_EQ(structure_view.get_n_cols(), 5);
  ASSERT_EQ(structure_view.get_n_rows(), 5);
  ASSERT_EQ(structure_view.get_nnz(), 0);

  sparsity_owning.initialize_sparsity(5);

  auto structure_view2 = sparsity_owning.structure_view();

  ASSERT_EQ(structure_view2.get_n_cols(), 5);
  ASSERT_EQ(structure_view2.get_n_rows(), 5);
  ASSERT_EQ(structure_view2.get_nnz(), 5);

  void* d_owning     = static_cast<void*>(sparsity_owning.get_elements().data());
  void* d_preserving = static_cast<void*>(sparsity_preserving.get_elements().data());

  test_host_csr_owning_ref(sparsity_owning, d_owning);
  test_host_csr_owning_ref(sparsity_preserving, d_preserving);

  test_host_csr_sparsity_owning_ref(sparsity_owning, d_owning);
  test_host_csr_sparsity_preserving_ref(sparsity_preserving, d_preserving);
}

TEST(HostSparseCOOMatrix, Basic) { test_host_coo_matrix(); }

TEST(HostSparseCSRMatrix, Basic) { test_host_csr_matrix(); }

}  // namespace raft
