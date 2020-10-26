/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 * Copyright 2020 KETAN DATE & RAKESH NAGI
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
 *
 *      CUDA Implementation of O(n^3) alternating tree Hungarian Algorithm
 *      Authors: Ketan Date and Rakesh Nagi
 *
 *      Article reference:
 *          Date, Ketan, and Rakesh Nagi. "GPU-accelerated Hungarian algorithms
 *          for the Linear Assignment Problem." Parallel Computing 57 (2016): 52-72.
 *
 */
#pragma once

#include <raft/handle.hpp>

#include "d_structs.h"
#include "lap_functions.cuh"

namespace raft {
namespace lap {

template <typename vertex_t, typename weight_t>
class LinearAssignmentProblem {
  vertex_t size_;
  vertex_t batchsize_;

  weight_t const *d_costs_;

  Vertices<vertex_t, weight_t> d_vertices_dev;
  VertexData<vertex_t> d_row_data_dev, d_col_data_dev;

  raft::handle_t const &handle_;
  raft::mr::device::buffer<vertex_t> row_assignments_v;
  raft::mr::device::buffer<vertex_t> col_assignments_v;
  raft::mr::device::buffer<int> row_covers_v;
  raft::mr::device::buffer<int> col_covers_v;
  raft::mr::device::buffer<weight_t> row_duals_v;
  raft::mr::device::buffer<weight_t> col_duals_v;
  raft::mr::device::buffer<weight_t> col_slacks_v;
  raft::mr::device::buffer<int> row_is_visited_v;
  raft::mr::device::buffer<int> col_is_visited_v;
  raft::mr::device::buffer<vertex_t> row_parents_v;
  raft::mr::device::buffer<vertex_t> col_parents_v;
  raft::mr::device::buffer<vertex_t> row_children_v;
  raft::mr::device::buffer<vertex_t> col_children_v;
  raft::mr::device::buffer<weight_t> obj_val_primal_v;
  raft::mr::device::buffer<weight_t> obj_val_dual_v;

 public:
  LinearAssignmentProblem(raft::handle_t const &handle, vertex_t size, vertex_t batchsize)
    : handle_(handle),
      size_(size),
      batchsize_(batchsize),
      d_costs_(nullptr),
      row_assignments_v(handle_.get_device_allocator(), handle_.get_stream(),
                        0),
      col_assignments_v(handle_.get_device_allocator(), handle_.get_stream(),
                        0),
      row_covers_v(handle_.get_device_allocator(), handle_.get_stream(), 0),
      col_covers_v(handle_.get_device_allocator(), handle_.get_stream(), 0),
      row_duals_v(handle_.get_device_allocator(), handle_.get_stream(), 0),
      col_duals_v(handle_.get_device_allocator(), handle_.get_stream(), 0),
      col_slacks_v(handle_.get_device_allocator(), handle_.get_stream(), 0),
      row_is_visited_v(handle_.get_device_allocator(), handle_.get_stream(), 0),
      col_is_visited_v(handle_.get_device_allocator(), handle_.get_stream(), 0),
      row_parents_v(handle_.get_device_allocator(), handle_.get_stream(), 0),
      col_parents_v(handle_.get_device_allocator(), handle_.get_stream(), 0),
      row_children_v(handle_.get_device_allocator(), handle_.get_stream(), 0),
      col_children_v(handle_.get_device_allocator(), handle_.get_stream(), 0),
      obj_val_primal_v(handle_.get_device_allocator(), handle_.get_stream(), 0),
      obj_val_dual_v(handle_.get_device_allocator(), handle_.get_stream(), 0) {}

  // Executes Hungarian algorithm on the input cost matrix.
  void solve(weight_t const *d_cost_matrix) {
    initializeDevice();

    d_costs_ = d_cost_matrix;

    int step = 0;

    while (step != 100) {
      switch (step) {
        case 0:
          step = hungarianStep0();
          break;
        case 1:
          step = hungarianStep1();
          break;
        case 2:
          step = hungarianStep2();
          break;
        case 3:
          step = hungarianStep3();
          break;
        case 4:
          step = hungarianStep4();
          break;
        case 5:
          step = hungarianStep5();
          break;
        case 6:
          step = hungarianStep6();
          break;
      }
    }

    d_costs_ = nullptr;
  }

  // Function for getting optimal assignment vector for subproblem spId.
  std::pair<const vertex_t *, vertex_t> getAssignmentVector(int spId) const {
    return std::make_pair(row_assignments_v.data() + spId * size_, size_);
  }

  // Function for getting optimal row dual vector for subproblem spId.
  std::pair<const weight_t *, vertex_t> getRowDualVector(int spId) const {
    return std::make_pair(row_duals_v.data() + spId * size_, size_);
  }

  // Function for getting optimal col dual vector for subproblem spId.
  std::pair<const weight_t *, vertex_t> getColDualVector(int spId) {
    return std::make_pair(col_duals_v.data() + spId * size_, size_);
  }

  // Function for getting optimal primal objective value for subproblem spId.
  weight_t getPrimalObjectiveValue(int spId) {
    weight_t result;
    raft::update_host(&result, obj_val_primal_v.data() + spId, 1,
                      handle_.get_stream());
    CHECK_CUDA(handle_.get_stream());
    return result;
  }

  // Function for getting optimal dual objective value for subproblem spId.
  weight_t getDualObjectiveValue(int spId) {
    weight_t result;
    raft::update_host(&result, obj_val_dual_v.data() + spId, 1,
                      handle_.get_stream());
    CHECK_CUDA(handle_.get_stream());
    return result;
  }

 private:
  // Helper function for initializing global variables and arrays on a single host.
  void initializeDevice() {
    row_assignments_v.resize(batchsize_ * size_);
    col_assignments_v.resize(batchsize_ * size_);
    row_covers_v.resize(batchsize_ * size_);
    col_covers_v.resize(batchsize_ * size_);
    row_duals_v.resize(batchsize_ * size_);
    col_duals_v.resize(batchsize_ * size_);
    col_slacks_v.resize(batchsize_ * size_);
    row_is_visited_v.resize(batchsize_ * size_);
    col_is_visited_v.resize(batchsize_ * size_);
    row_parents_v.resize(batchsize_ * size_);
    col_parents_v.resize(batchsize_ * size_);
    row_children_v.resize(batchsize_ * size_);
    col_children_v.resize(batchsize_ * size_);
    obj_val_primal_v.resize(batchsize_);
    obj_val_dual_v.resize(batchsize_);

    d_vertices_dev.row_assignments = row_assignments_v.data();
    d_vertices_dev.col_assignments = col_assignments_v.data();
    d_vertices_dev.row_covers = row_covers_v.data();
    d_vertices_dev.col_covers = col_covers_v.data();

    d_vertices_dev.row_duals = row_duals_v.data();
    d_vertices_dev.col_duals = col_duals_v.data();
    d_vertices_dev.col_slacks = col_slacks_v.data();

    d_row_data_dev.is_visited = row_is_visited_v.data();
    d_col_data_dev.is_visited = col_is_visited_v.data();
    d_row_data_dev.parents = row_parents_v.data();
    d_row_data_dev.children = row_children_v.data();
    d_col_data_dev.parents = col_parents_v.data();
    d_col_data_dev.children = col_children_v.data();

    thrust::fill(thrust::device, row_assignments_v.begin(),
                 row_assignments_v.end(), int{-1});
    thrust::fill(thrust::device, col_assignments_v.begin(),
                 col_assignments_v.end(), int{-1});
    thrust::fill(thrust::device, row_covers_v.begin(), row_covers_v.end(),
                 int{0});
    thrust::fill(thrust::device, col_covers_v.begin(), col_covers_v.end(),
                 int{0});
    thrust::fill(thrust::device, row_duals_v.begin(), row_duals_v.end(),
                 weight_t{0});
    thrust::fill(thrust::device, col_duals_v.begin(), col_duals_v.end(),
                 weight_t{0});
  }

  // Function for calculating initial zeros by subtracting row and column minima from each element.
  int hungarianStep0() {
    detail::initialReduction(handle_, d_costs_, d_vertices_dev, batchsize_,
                             size_);

    return 1;
  }

  // Function for calculating initial zeros by subtracting row and column minima from each element.
  int hungarianStep1() {
    detail::computeInitialAssignments(handle_, d_costs_, d_vertices_dev,
                                      batchsize_, size_);

    int next = 2;

    while (true) {
      if ((next = hungarianStep2()) == 6) break;

      if ((next = hungarianStep3()) == 5) break;

      hungarianStep4();
    }

    return next;
  }

  // Function for checking optimality and constructing predicates and covers.
  int hungarianStep2() {
    int cover_count =
      detail::computeRowCovers(handle_, d_vertices_dev, d_row_data_dev,
                               d_col_data_dev, batchsize_, size_);

    int next = (cover_count == batchsize_ * size_) ? 6 : 3;

    return next;
  }

  // Function for building alternating tree rooted at unassigned rows.
  int hungarianStep3() {
    int next;

    raft::mr::device::buffer<bool> flag_v(handle_.get_device_allocator(),
                                          handle_.get_stream(), 1);

    bool h_flag = false;
    raft::update_device(flag_v.data(), &h_flag, 1, handle_.get_stream());

    detail::executeZeroCover(handle_, d_costs_, d_vertices_dev,
                             d_row_data_dev, d_col_data_dev, flag_v.data(),
                             batchsize_, size_);

    raft::update_host(&h_flag, flag_v.data(), 1, handle_.get_stream());

    next = h_flag ? 4 : 5;

    return next;
  }

  // Function for augmenting the solution along multiple node-disjoint alternating trees.
  int hungarianStep4() {
    detail::reversePass(handle_, d_row_data_dev, d_col_data_dev, batchsize_,
                        size_);

    detail::augmentationPass(handle_, d_vertices_dev, d_row_data_dev,
                             d_col_data_dev, batchsize_, size_);

    return 2;
  }

  // Function for updating dual solution to introduce new zero-cost arcs.
  int hungarianStep5() {
    detail::dualUpdate(handle_, d_vertices_dev, d_row_data_dev, d_col_data_dev,
                       batchsize_, size_);

    return 3;
  }

  // Function for calculating primal and dual objective values at optimality.
  int hungarianStep6() {
    detail::calcObjValPrimal(handle_, obj_val_primal_v.data(),
                             d_costs_, d_vertices_dev.row_assignments, batchsize_,
                             size_);

    detail::calcObjValDual(handle_, obj_val_dual_v.data(), d_vertices_dev,
                           batchsize_, size_);

    return 100;
  }
};

}  // namespace lap
}  // namespace raft
