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

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include "d_structs.h"

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <raft/mr/device/buffer.hpp>

#include <raft/lap/lap_kernels.cuh>

namespace raft {
namespace lap {
namespace detail {

const int BLOCKDIMX{64};
const int BLOCKDIMY{1};

// Function for calculating grid and block dimensions from the given input size.
inline void calculateLinearDims(dim3 &blocks_per_grid, dim3 &threads_per_block,
                                int &total_blocks, int size) {
  threads_per_block.x = BLOCKDIMX * BLOCKDIMY;

  int value = size / threads_per_block.x;
  if (size % threads_per_block.x > 0) value++;

  total_blocks = value;
  blocks_per_grid.x = value;
}

// Function for calculating grid and block dimensions from the given input size for square grid.
inline void calculateSquareDims(dim3 &blocks_per_grid, dim3 &threads_per_block,
                                int &total_blocks, int size) {
  threads_per_block.x = BLOCKDIMX;
  threads_per_block.y = BLOCKDIMY;

  int sq_size = (int)ceil(sqrt(size));

  int valuex = (int)ceil((float)(sq_size) / BLOCKDIMX);
  int valuey = (int)ceil((float)(sq_size) / BLOCKDIMY);

  total_blocks = valuex * valuey;
  blocks_per_grid.x = valuex;
  blocks_per_grid.y = valuey;
}

// Function for calculating grid and block dimensions from the given input size for rectangular grid.
inline void calculateRectangularDims(dim3 &blocks_per_grid,
                                     dim3 &threads_per_block, int &total_blocks,
                                     int xsize, int ysize) {
  threads_per_block.x = BLOCKDIMX;
  threads_per_block.y = BLOCKDIMY;

  int valuex = xsize / threads_per_block.x;
  if (xsize % threads_per_block.x > 0) valuex++;

  int valuey = ysize / threads_per_block.y;
  if (ysize % threads_per_block.y > 0) valuey++;

  total_blocks = valuex * valuey;
  blocks_per_grid.x = valuex;
  blocks_per_grid.y = valuey;
}

template <typename vertex_t, typename weight_t>
inline void initialReduction(raft::handle_t const &handle,
                             weight_t const *d_costs,
                             Vertices<vertex_t, weight_t> &d_vertices_dev,
                             int SP, vertex_t N) {
  dim3 blocks_per_grid;
  dim3 threads_per_block;
  int total_blocks = 0;

  raft::lap::detail::calculateRectangularDims(
    blocks_per_grid, threads_per_block, total_blocks, N, SP);

  kernel_rowReduction<<<blocks_per_grid, threads_per_block, 0,
                        handle.get_stream()>>>(
    d_costs, d_vertices_dev.row_duals, SP, N,
    std::numeric_limits<weight_t>::max());

  CHECK_CUDA(handle.get_stream());
  kernel_columnReduction<<<blocks_per_grid, threads_per_block, 0,
                           handle.get_stream()>>>(
    d_costs, d_vertices_dev.row_duals, d_vertices_dev.col_duals, SP, N,
    std::numeric_limits<weight_t>::max());
  CHECK_CUDA(handle.get_stream());
}

template <typename vertex_t, typename weight_t>
inline void computeInitialAssignments(raft::handle_t const &handle,
                                      weight_t const *d_costs,
                                      Vertices<vertex_t, weight_t> &d_vertices,
                                      int SP, vertex_t N, weight_t epsilon) {
  dim3 blocks_per_grid;
  dim3 threads_per_block;
  int total_blocks = 0;

  std::size_t size = SP * N;

  raft::mr::device::buffer<int> row_lock_v(handle.get_device_allocator(),
                                           handle.get_stream(), size);
  raft::mr::device::buffer<int> col_lock_v(handle.get_device_allocator(),
                                           handle.get_stream(), size);

  thrust::fill_n(thrust::device, d_vertices.row_assignments, size, -1);
  thrust::fill_n(thrust::device, d_vertices.col_assignments, size, -1);
  thrust::fill_n(thrust::device, row_lock_v.data(), size, 0);
  thrust::fill_n(thrust::device, col_lock_v.data(), size, 0);

  raft::lap::detail::calculateRectangularDims(
    blocks_per_grid, threads_per_block, total_blocks, N, SP);

  kernel_computeInitialAssignments<<<blocks_per_grid, threads_per_block, 0,
                                     handle.get_stream()>>>(
    d_costs, d_vertices.row_duals, d_vertices.col_duals,
    d_vertices.row_assignments, d_vertices.col_assignments, row_lock_v.data(),
    col_lock_v.data(), SP, N, epsilon);
  CHECK_CUDA(handle.get_stream());
}

// Function for finding row cover on individual devices.
template <typename vertex_t, typename weight_t>
inline int computeRowCovers(raft::handle_t const &handle,
                            Vertices<vertex_t, weight_t> &d_vertices,
                            VertexData<vertex_t> &d_row_data,
                            VertexData<vertex_t> &d_col_data, int SP,
                            vertex_t N) {
  dim3 blocks_per_grid;
  dim3 threads_per_block;
  int total_blocks = 0;

  std::size_t size = SP * N;

  thrust::fill_n(thrust::device, d_vertices.row_covers, size, int{0});
  thrust::fill_n(thrust::device, d_vertices.col_covers, size, int{0});
  thrust::fill_n(thrust::device, d_vertices.col_slacks, size,
                 std::numeric_limits<weight_t>::max());
  thrust::fill_n(thrust::device, d_row_data.is_visited, size, DORMANT);
  thrust::fill_n(thrust::device, d_col_data.is_visited, size, DORMANT);
  thrust::fill_n(thrust::device, d_row_data.parents, size, vertex_t{-1});
  thrust::fill_n(thrust::device, d_row_data.children, size, vertex_t{-1});
  thrust::fill_n(thrust::device, d_col_data.parents, size, vertex_t{-1});
  thrust::fill_n(thrust::device, d_col_data.children, size, vertex_t{-1});

  raft::lap::detail::calculateRectangularDims(
    blocks_per_grid, threads_per_block, total_blocks, N, SP);
  kernel_computeRowCovers<<<blocks_per_grid, threads_per_block, 0,
                            handle.get_stream()>>>(
    d_vertices.row_assignments, d_vertices.row_covers, d_row_data.is_visited,
    SP, N);

  CHECK_CUDA(handle.get_stream());

  return thrust::reduce(thrust::device, d_vertices.row_covers,
                        d_vertices.row_covers + size);
}

// Function for covering the zeros in uncovered rows and expanding the frontier.
template <typename vertex_t, typename weight_t>
inline void coverZeroAndExpand(
  raft::handle_t const &handle, weight_t const *d_costs_dev,
  vertex_t const *d_rows_csr_neighbors, vertex_t const *d_rows_csr_ptrs,
  Vertices<vertex_t, weight_t> &d_vertices_dev,
  VertexData<vertex_t> &d_row_data_dev, VertexData<vertex_t> &d_col_data_dev,
  bool *d_flag, int SP, vertex_t N, weight_t epsilon) {
  int total_blocks = 0;
  dim3 blocks_per_grid;
  dim3 threads_per_block;

  raft::lap::detail::calculateRectangularDims(
    blocks_per_grid, threads_per_block, total_blocks, N, SP);

  kernel_coverAndExpand<<<blocks_per_grid, threads_per_block, 0,
                          handle.get_stream()>>>(
    d_flag, d_rows_csr_ptrs, d_rows_csr_neighbors, d_costs_dev, d_vertices_dev,
    d_row_data_dev, d_col_data_dev, SP, N, epsilon);
}

template <typename vertex_t, typename weight_t>
inline vertex_t zeroCoverIteration(raft::handle_t const &handle,
                                   weight_t const *d_costs_dev,
                                   Vertices<vertex_t, weight_t> &d_vertices_dev,
                                   VertexData<vertex_t> &d_row_data_dev,
                                   VertexData<vertex_t> &d_col_data_dev,
                                   bool *d_flag, int SP, vertex_t N,
                                   weight_t epsilon) {
  vertex_t M;

  raft::mr::device::buffer<vertex_t> csr_ptrs_v(handle.get_device_allocator(),
                                                handle.get_stream(), 0);
  raft::mr::device::buffer<vertex_t> csr_neighbors_v(
    handle.get_device_allocator(), handle.get_stream(), 0);

  {
    dim3 blocks_per_grid;
    dim3 threads_per_block;
    int total_blocks = 0;

    raft::mr::device::buffer<bool> predicates_v(handle.get_device_allocator(),
                                                handle.get_stream(), SP * N);
    raft::mr::device::buffer<vertex_t> addresses_v(
      handle.get_device_allocator(), handle.get_stream(), SP * N);

    thrust::fill_n(thrust::device, predicates_v.data(), SP * N, false);
    thrust::fill_n(thrust::device, addresses_v.data(), SP * N, vertex_t{0});

    csr_ptrs_v.resize(SP + 1);

    thrust::fill_n(thrust::device, csr_ptrs_v.data(), (SP + 1), vertex_t{-1});

    raft::lap::detail::calculateRectangularDims(
      blocks_per_grid, threads_per_block, total_blocks, N, SP);

    // construct predicate matrix for edges.
    kernel_rowPredicateConstructionCSR<<<blocks_per_grid, threads_per_block, 0,
                                         handle.get_stream()>>>(
      predicates_v.data(), addresses_v.data(), d_row_data_dev.is_visited, SP,
      N);
    CHECK_CUDA(handle.get_stream());

    M = thrust::reduce(thrust::device, addresses_v.begin(), addresses_v.end());
    thrust::exclusive_scan(thrust::device, addresses_v.begin(),
                           addresses_v.end(), addresses_v.begin());

    if (M > 0) {
      csr_neighbors_v.resize(M);

      kernel_rowScatterCSR<<<blocks_per_grid, threads_per_block, 0,
                             handle.get_stream()>>>(
        predicates_v.data(), addresses_v.data(), csr_neighbors_v.data(),
        csr_ptrs_v.data(), M, SP, N);

      CHECK_CUDA(handle.get_stream());
    }
  }

  if (M > 0) {
    coverZeroAndExpand(handle, d_costs_dev, csr_neighbors_v.data(),
                       csr_ptrs_v.data(), d_vertices_dev, d_row_data_dev,
                       d_col_data_dev, d_flag, SP, N, epsilon);
  }

  return M;
}

// Function for executing recursive zero cover. Returns the next step (Step 4 or Step 5) depending on the presence of uncovered zeros.
template <typename vertex_t, typename weight_t>
inline void executeZeroCover(raft::handle_t const &handle,
                             weight_t const *d_costs_dev,
                             Vertices<vertex_t, weight_t> &d_vertices_dev,
                             VertexData<vertex_t> &d_row_data_dev,
                             VertexData<vertex_t> &d_col_data_dev, bool *d_flag,
                             int SP, vertex_t N, weight_t epsilon) {
  vertex_t M = 1;
  while (M > 0) {
    M = zeroCoverIteration(handle, d_costs_dev, d_vertices_dev, d_row_data_dev,
                           d_col_data_dev, d_flag, SP, N, epsilon);
  }
}

// Function for executing reverse pass of the maximum matching.
template <typename vertex_t>
inline void reversePass(raft::handle_t const &handle,
                        VertexData<vertex_t> &d_row_data_dev,
                        VertexData<vertex_t> &d_col_data_dev, int SP, int N) {
  int total_blocks = 0;
  dim3 blocks_per_grid;
  dim3 threads_per_block;

  std::size_t size = SP * N;

  raft::lap::detail::calculateLinearDims(blocks_per_grid, threads_per_block,
                                         total_blocks, size);

  raft::mr::device::buffer<bool> predicates_v(handle.get_device_allocator(),
                                              handle.get_stream(), size);
  raft::mr::device::buffer<vertex_t> addresses_v(handle.get_device_allocator(),
                                                 handle.get_stream(), size);

  thrust::fill_n(thrust::device, predicates_v.data(), size, false);
  thrust::fill_n(thrust::device, addresses_v.data(), size, vertex_t{0});

  // compact the reverse pass row vertices.
  kernel_augmentPredicateConstruction<<<blocks_per_grid, threads_per_block, 0,
                                        handle.get_stream()>>>(
    predicates_v.data(), addresses_v.data(), d_col_data_dev.is_visited, size);

  CHECK_CUDA(handle.get_stream());

  // calculate total number of vertices.
  std::size_t csr_size =
    thrust::reduce(thrust::device, addresses_v.begin(), addresses_v.end());
  // exclusive scan for calculating the scatter addresses.
  thrust::exclusive_scan(thrust::device, addresses_v.begin(), addresses_v.end(),
                         addresses_v.begin());

  if (csr_size > 0) {
    int total_blocks_1 = 0;
    dim3 blocks_per_grid_1;
    dim3 threads_per_block_1;
    raft::lap::detail::calculateLinearDims(
      blocks_per_grid_1, threads_per_block_1, total_blocks_1, csr_size);

    raft::mr::device::buffer<vertex_t> elements_v(
      handle.get_device_allocator(), handle.get_stream(), csr_size);

    kernel_augmentScatter<<<blocks_per_grid, threads_per_block, 0,
                            handle.get_stream()>>>(
      elements_v.data(), predicates_v.data(), addresses_v.data(), size);

    CHECK_CUDA(handle.get_stream());

    kernel_reverseTraversal<<<blocks_per_grid_1, threads_per_block_1, 0,
                              handle.get_stream()>>>(
      elements_v.data(), d_row_data_dev, d_col_data_dev, csr_size);
    CHECK_CUDA(handle.get_stream());
  }
}

// Function for executing augmentation pass of the maximum matching.
template <typename vertex_t, typename weight_t>
inline void augmentationPass(raft::handle_t const &handle,
                             Vertices<vertex_t, weight_t> &d_vertices_dev,
                             VertexData<vertex_t> &d_row_data_dev,
                             VertexData<vertex_t> &d_col_data_dev, int SP,
                             int N) {
  int total_blocks = 0;
  dim3 blocks_per_grid;
  dim3 threads_per_block;
  raft::lap::detail::calculateLinearDims(blocks_per_grid, threads_per_block,
                                         total_blocks, SP * N);

  raft::mr::device::buffer<bool> predicates_v(handle.get_device_allocator(),
                                              handle.get_stream(), SP * N);
  raft::mr::device::buffer<vertex_t> addresses_v(handle.get_device_allocator(),
                                                 handle.get_stream(), SP * N);

  thrust::fill_n(thrust::device, predicates_v.data(), SP * N, false);
  thrust::fill_n(thrust::device, addresses_v.data(), SP * N, vertex_t{0});

  // compact the reverse pass row vertices.
  kernel_augmentPredicateConstruction<<<blocks_per_grid, threads_per_block, 0,
                                        handle.get_stream()>>>(
    predicates_v.data(), addresses_v.data(), d_row_data_dev.is_visited, SP * N);

  CHECK_CUDA(handle.get_stream());

  // calculate total number of vertices.
  // TODO: should be vertex_t
  vertex_t row_ids_csr_size =
    thrust::reduce(thrust::device, addresses_v.begin(), addresses_v.end());
  // exclusive scan for calculating the scatter addresses.
  thrust::exclusive_scan(thrust::device, addresses_v.begin(), addresses_v.end(),
                         addresses_v.begin());

  if (row_ids_csr_size > 0) {
    int total_blocks_1 = 0;
    dim3 blocks_per_grid_1;
    dim3 threads_per_block_1;
    raft::lap::detail::calculateLinearDims(
      blocks_per_grid_1, threads_per_block_1, total_blocks_1, row_ids_csr_size);

    raft::mr::device::buffer<vertex_t> elements_v(
      handle.get_device_allocator(), handle.get_stream(), row_ids_csr_size);

    kernel_augmentScatter<<<blocks_per_grid, threads_per_block, 0,
                            handle.get_stream()>>>(
      elements_v.data(), predicates_v.data(), addresses_v.data(),
      vertex_t{SP * N});

    CHECK_CUDA(handle.get_stream());

    kernel_augmentation<<<blocks_per_grid_1, threads_per_block_1, 0,
                          handle.get_stream()>>>(
      d_vertices_dev.row_assignments, d_vertices_dev.col_assignments,
      elements_v.data(), d_row_data_dev, d_col_data_dev, vertex_t{N},
      row_ids_csr_size);

    CHECK_CUDA(handle.get_stream());
  }
}

template <typename vertex_t, typename weight_t>
inline void dualUpdate(raft::handle_t const &handle,
                       Vertices<vertex_t, weight_t> &d_vertices_dev,
                       VertexData<vertex_t> &d_row_data_dev,
                       VertexData<vertex_t> &d_col_data_dev, int SP, vertex_t N,
                       weight_t epsilon) {
  dim3 blocks_per_grid;
  dim3 threads_per_block;
  int total_blocks;

  raft::mr::device::buffer<weight_t> sp_min_v(handle.get_device_allocator(),
                                              handle.get_stream(), 1);

  raft::lap::detail::calculateLinearDims(blocks_per_grid, threads_per_block,
                                         total_blocks, SP);
  kernel_dualUpdate_1<<<blocks_per_grid, threads_per_block, 0,
                        handle.get_stream()>>>(
    sp_min_v.data(), d_vertices_dev.col_slacks, d_vertices_dev.col_covers, SP,
    N, std::numeric_limits<weight_t>::max());

  CHECK_CUDA(handle.get_stream());

  raft::lap::detail::calculateRectangularDims(
    blocks_per_grid, threads_per_block, total_blocks, N, SP);
  kernel_dualUpdate_2<<<blocks_per_grid, threads_per_block, 0,
                        handle.get_stream()>>>(
    sp_min_v.data(), d_vertices_dev.row_duals, d_vertices_dev.col_duals,
    d_vertices_dev.col_slacks, d_vertices_dev.row_covers,
    d_vertices_dev.col_covers, d_row_data_dev.is_visited,
    d_col_data_dev.parents, SP, N, std::numeric_limits<weight_t>::max(),
    epsilon);

  CHECK_CUDA(handle.get_stream());
}

// Function for calculating optimal objective function value using dual variables.
template <typename vertex_t, typename weight_t>
inline void calcObjValDual(raft::handle_t const &handle, weight_t *d_obj_val,
                           Vertices<vertex_t, weight_t> &d_vertices_dev, int SP,
                           int N) {
  dim3 blocks_per_grid;
  dim3 threads_per_block;
  int total_blocks = 0;

  raft::lap::detail::calculateLinearDims(blocks_per_grid, threads_per_block,
                                         total_blocks, SP);

  kernel_calcObjValDual<<<blocks_per_grid, threads_per_block, 0,
                          handle.get_stream()>>>(
    d_obj_val, d_vertices_dev.row_duals, d_vertices_dev.col_duals, SP, N);

  CHECK_CUDA(handle.get_stream());
}

// Function for calculating optimal objective function value using dual variables.
template <typename vertex_t, typename weight_t>
inline void calcObjValPrimal(raft::handle_t const &handle, weight_t *d_obj_val,
                             weight_t const *d_costs,
                             vertex_t const *d_row_assignments, int SP,
                             vertex_t N) {
  dim3 blocks_per_grid;
  dim3 threads_per_block;
  int total_blocks = 0;

  raft::lap::detail::calculateLinearDims(blocks_per_grid, threads_per_block,
                                         total_blocks, SP);

  kernel_calcObjValPrimal<<<blocks_per_grid, threads_per_block, 0,
                            handle.get_stream()>>>(d_obj_val, d_costs,
                                                   d_row_assignments, SP, N);

  CHECK_CUDA(handle.get_stream());
}

}  // namespace detail
}  // namespace lap
}  // namespace raft
