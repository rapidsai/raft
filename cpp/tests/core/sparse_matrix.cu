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
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/resources.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace raft {

/**
 * Example of accepting a value-owning matrix type which doesn't need to adjust sparsity
 */
template <typename S, typename = std::enable_if_t<is_device_csr_matrix_v<S>>>
void test_device_csr_owning_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a csr_matrix which also owns structure and can initialize sparsity
 */
template <typename S, typename = std::enable_if_t<is_device_csr_sparsity_owning_v<S>>>
void test_device_csr_sparsity_owning_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a csr_matrix which does not own the structure
 */
template <typename S, typename = std::enable_if_t<is_device_csr_sparsity_preserving_v<S>>>
void test_device_csr_sparsity_preserving_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a value-owning matrix type which doesn't need to adjust sparsity
 */
template <typename S, typename = std::enable_if_t<is_device_coo_matrix_v<S>>>
void test_device_coo_owning_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a coo_matrix which also owns structure and can initialize sparsity
 */
template <typename S, typename = std::enable_if_t<is_device_coo_sparsity_owning_v<S>>>
void test_device_coo_sparsity_owning_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a coo_matrix which does not own the structure
 */
template <typename S, typename = std::enable_if_t<is_device_coo_sparsity_preserving_v<S>>>
void test_device_coo_sparsity_preserving_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

void test_device_coo_matrix()
{
  raft::resources handle;
  auto sparsity_owning = raft::make_device_coo_matrix<float, int, int, int>(handle, 5, 5);

  auto structure_view = sparsity_owning.structure_view();

  ASSERT_EQ(structure_view.get_n_cols(), 5);
  ASSERT_EQ(structure_view.get_n_rows(), 5);
  ASSERT_EQ(structure_view.get_nnz(), 0);

  auto coord_struct = raft::make_device_coordinate_structure(handle, 5, 5, 5);
  auto sparsity_preserving =
    raft::make_device_coo_matrix<float, int, int>(handle, coord_struct.view());

  sparsity_owning.initialize_sparsity(5);

  auto structure_view2 = sparsity_owning.structure_view();

  ASSERT_EQ(structure_view2.get_n_cols(), 5);
  ASSERT_EQ(structure_view2.get_n_rows(), 5);
  ASSERT_EQ(structure_view2.get_nnz(), 5);

  void* d_owning     = static_cast<void*>(sparsity_owning.get_elements().data());
  void* d_preserving = static_cast<void*>(sparsity_preserving.get_elements().data());

  test_device_coo_owning_ref(sparsity_owning, d_owning);
  test_device_coo_owning_ref(sparsity_preserving, d_preserving);

  test_device_coo_sparsity_owning_ref(sparsity_owning, d_owning);
  test_device_coo_sparsity_preserving_ref(sparsity_preserving, d_preserving);
}

void test_device_csr_matrix()
{
  raft::resources handle;
  auto sparsity_owning = raft::make_device_csr_matrix<float, int, int, int>(handle, 5, 5);

  auto comp_struct = raft::make_device_compressed_structure(handle, 5, 5, 5);
  auto sparsity_preserving =
    raft::make_device_csr_matrix<float, int, int>(handle, comp_struct.view());

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

  test_device_csr_owning_ref(sparsity_owning, d_owning);
  test_device_csr_owning_ref(sparsity_preserving, d_preserving);

  test_device_csr_sparsity_owning_ref(sparsity_owning, d_owning);
  test_device_csr_sparsity_preserving_ref(sparsity_preserving, d_preserving);
}

TEST(DeviceSparseCOOMatrix, Basic) { test_device_coo_matrix(); }

TEST(DeviceSparseCSRMatrix, Basic) { test_device_csr_matrix(); }

}  // namespace raft
