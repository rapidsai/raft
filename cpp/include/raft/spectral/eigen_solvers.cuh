/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#ifndef __EIGEN_SOLVERS_H
#define __EIGEN_SOLVERS_H

#pragma once

#include <raft/core/mdspan.hpp>
#include <raft/sparse/solver/lanczos.cuh>
#include <raft/spectral/matrix_wrappers.hpp>

namespace raft {
namespace spectral {

// aggregate of control params for Eigen Solver:
//
template <typename index_type_t, typename value_type_t, typename size_type_t = index_type_t>
struct eigen_solver_config_t {
  size_type_t n_eigVecs;
  size_type_t maxIter;

  size_type_t restartIter;
  value_type_t tol;

  bool reorthogonalize{false};
  unsigned long long seed{
    1234567};  // CAVEAT: this default value is now common to all instances of using seed in
               // Lanczos; was not the case before: there were places where a default seed = 123456
               // was used; this may trigger slightly different # solver iterations

  raft::sparse::solver::LANCZOS_WHICH which{raft::sparse::solver::LANCZOS_WHICH::SA};
};

template <typename index_type_t, typename value_type_t, typename size_type_t = index_type_t>
struct lanczos_solver_t {
  explicit lanczos_solver_t(
    eigen_solver_config_t<index_type_t, value_type_t, size_type_t> const& config)
    : config_(config)
  {
  }

  auto solve_smallest_eigenvectors(
    raft::resources const& handle,
    matrix::sparse_matrix_t<index_type_t, value_type_t, size_type_t> const& A,
    value_type_t* __restrict__ eigVals,
    value_type_t* __restrict__ eigVecs) const
  {
    auto csr_structure =
      raft::make_device_compressed_structure_view<index_type_t, index_type_t, index_type_t>(
        const_cast<index_type_t*>(A.row_offsets_),
        const_cast<index_type_t*>(A.col_indices_),
        A.nrows_,
        A.ncols_,
        A.nnz_);

    auto csr_matrix =
      raft::make_device_csr_matrix_view<value_type_t, index_type_t, index_type_t, index_type_t>(
        const_cast<value_type_t*>(A.values_), csr_structure);
    return solve_smallest_eigenvectors(handle, csr_matrix, eigVals, eigVecs);
  }

  auto solve_smallest_eigenvectors(
    raft::resources const& handle,
    device_csr_matrix_view<value_type_t, index_type_t, index_type_t, index_type_t> input,
    value_type_t* __restrict__ eigVals,
    value_type_t* __restrict__ eigVecs) const
  {
    RAFT_EXPECTS(eigVals != nullptr, "Null eigVals buffer.");
    RAFT_EXPECTS(eigVecs != nullptr, "Null eigVecs buffer.");

    auto lanczos_config =
      raft::sparse::solver::lanczos_solver_config<value_type_t>{config_.n_eigVecs,
                                                                config_.maxIter,
                                                                config_.restartIter,
                                                                config_.tol,
                                                                config_.which,
                                                                config_.seed};
    auto v0_opt = std::optional<raft::device_vector_view<value_type_t, uint32_t, raft::row_major>>{
      std::nullopt};
    auto input_structure = input.structure_view();

    auto solver_iterations = sparse::solver::lanczos_compute_smallest_eigenvectors(
      handle,
      lanczos_config,
      input,
      v0_opt,
      raft::make_device_vector_view<value_type_t, uint32_t, raft::col_major>(eigVals,
                                                                             config_.n_eigVecs),
      raft::make_device_matrix_view<value_type_t, uint32_t, raft::col_major>(
        eigVecs, input_structure.get_n_rows(), config_.n_eigVecs));

    return index_type_t{solver_iterations};
  }

  index_type_t solve_largest_eigenvectors(
    raft::resources const& handle,
    matrix::sparse_matrix_t<index_type_t, value_type_t, size_type_t> const& A,
    value_type_t* __restrict__ eigVals,
    value_type_t* __restrict__ eigVecs) const
  {
    RAFT_EXPECTS(eigVals != nullptr, "Null eigVals buffer.");
    RAFT_EXPECTS(eigVecs != nullptr, "Null eigVecs buffer.");
    index_type_t iters{};
    sparse::solver::computeLargestEigenvectors(handle,
                                               A,
                                               config_.n_eigVecs,
                                               config_.maxIter,
                                               config_.restartIter,
                                               config_.tol,
                                               config_.reorthogonalize,
                                               iters,
                                               eigVals,
                                               eigVecs,
                                               config_.seed);
    return iters;
  }

  auto const& get_config(void) const { return config_; }

 private:
  eigen_solver_config_t<index_type_t, value_type_t, size_type_t> config_;
};

}  // namespace spectral
}  // namespace raft

#endif
