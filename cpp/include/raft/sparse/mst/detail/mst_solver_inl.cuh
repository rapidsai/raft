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

#pragma once

#include <curand.h>
#include <chrono>

#include "mst_kernels.cuh"
#include "utils.cuh"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <iostream>

#include <raft/cudart_utils.h>
#include <rmm/device_buffer.hpp>

namespace raft {
namespace mst {
typedef std::chrono::high_resolution_clock Clock;

// curand generator uniform
curandStatus_t curand_generate_uniformX(curandGenerator_t generator,
                                        float* outputPtr, size_t n) {
  return curandGenerateUniform(generator, outputPtr, n);
}
curandStatus_t curand_generate_uniformX(curandGenerator_t generator,
                                        double* outputPtr, size_t n) {
  return curandGenerateUniformDouble(generator, outputPtr, n);
}

template <typename vertex_t, typename edge_t, typename weight_t>
MST_solver<vertex_t, edge_t, weight_t>::MST_solver(
  const raft::handle_t& handle_, edge_t const* offsets_,
  vertex_t const* indices_, weight_t const* weights_, vertex_t const v_,
  edge_t const e_, vertex_t* color_, cudaStream_t stream_)
  : handle(handle_),
    offsets(offsets_),
    indices(indices_),
    weights(weights_),
    altered_weights(e_),
    v(v_),
    e(e_),
    color(color_),
    color_index(v_),
    next_color(v_),
    min_edge_color(v_),
    new_mst_edge(v_),
    mst_edge(e_, false),
    temp_src(2 * v_),
    temp_dst(2 * v_),
    temp_weights(2 * v_),
    mst_edge_count(1, 0),
    prev_mst_edge_count(1, 0),
    stream(stream_) {
  max_blocks = handle_.get_device_properties().maxGridSize[0];
  max_threads = handle_.get_device_properties().maxThreadsPerBlock;
  sm_count = handle_.get_device_properties().multiProcessorCount;

  //Initially, color holds the vertex id as color
  auto policy = rmm::exec_policy(stream);
  thrust::sequence(policy->on(stream), color, color + v, 0);
  thrust::sequence(policy->on(stream), color_index.begin(), color_index.end(),
                   0);
  thrust::sequence(policy->on(stream), next_color.begin(), next_color.end(), 0);

  //Initially, each edge is not in the mst
}

template <typename vertex_t, typename edge_t, typename weight_t>
raft::Graph_COO<vertex_t, edge_t, weight_t>
MST_solver<vertex_t, edge_t, weight_t>::solve() {
  RAFT_EXPECTS(v > 0, "0 vertices");
  RAFT_EXPECTS(e > 0, "0 edges");
  RAFT_EXPECTS(offsets != nullptr, "Null offsets.");
  RAFT_EXPECTS(indices != nullptr, "Null indices.");
  RAFT_EXPECTS(weights != nullptr, "Null weights.");
#ifdef MST_TIME
  double timer0 = 0, timer1 = 0, timer2 = 0, timer3 = 0, timer4 = 0, timer5 = 0;
  auto start = Clock::now();
#endif

  // Alterating the weights
  // this is done by identifying the lowest cost edge weight gap that is not 0, call this theta.
  // For each edge, add noise that is less than theta. That is, generate a random number in the range [0.0, theta) and add it to each edge weight.
  alteration();

#ifdef MST_TIME
  auto stop = Clock::now();
  timer0 = duration_us(stop - start);
#endif

  Graph_COO<vertex_t, edge_t, weight_t> mst_result(2 * v - 2, stream);

  // Boruvka original formulation says "while more than 1 supervertex remains"
  // Here we adjust it to support disconnected components (spanning forest)
  // track completion with mst_edge_found status.
  for (auto i = 0; i < v; i++) {
#ifdef MST_TIME
    start = Clock::now();
#endif
    // Finds the minimum outgoing edge from each supervertex to the lowest outgoing color
    // by working at each vertex of the supervertex
    min_edge_per_vertex();

#ifdef MST_TIME
    stop = Clock::now();
    timer1 += duration_us(stop - start);
    start = Clock::now();
#endif

    min_edge_per_supervertex();

#ifdef MST_TIME
    stop = Clock::now();
    timer2 += duration_us(stop - start);
    start = Clock::now();
#endif

    // check if msf/mst done, count new edges added
    check_termination();

#ifdef MST_TIME
    stop = Clock::now();
    timer3 += duration_us(stop - start);
#endif

    if (prev_mst_edge_count[0] == mst_edge_count[0]) {
#ifdef MST_TIME
      std::cout << "Iterations: " << i << std::endl;
      std::cout << timer0 << "," << timer1 << "," << timer2 << "," << timer3
                << "," << timer4 << "," << timer5 << std::endl;
#endif
      break;
    }

#ifdef MST_TIME
    start = Clock::now();
#endif
    // append the newly found MST edges to the final output
    append_src_dst_pair(mst_result.src.data(), mst_result.dst.data(),
                        mst_result.weights.data());
#ifdef MST_TIME
    stop = Clock::now();
    timer4 += duration_us(stop - start);
    start = Clock::now();
#endif

    // updates colors of supervertices by propagating the lower color to the higher
    label_prop(mst_result.src.data(), mst_result.dst.data());

#ifdef MST_TIME
    stop = Clock::now();
    timer5 += duration_us(stop - start);
#endif

    // copy this iteration's results and store
    prev_mst_edge_count = mst_edge_count;
  }

  thrust::host_vector<edge_t> host_mst_edge_count = mst_edge_count;
  mst_result.n_edges = host_mst_edge_count[0];
  mst_result.src.resize(mst_result.n_edges, stream);
  mst_result.dst.resize(mst_result.n_edges, stream);
  mst_result.weights.resize(mst_result.n_edges, stream);

  return mst_result;
}

// ||y|-|x||
template <typename weight_t>
struct alteration_functor {
  __host__ __device__ weight_t
  operator()(const thrust::tuple<weight_t, weight_t>& t) {
    auto x = thrust::get<0>(t);
    auto y = thrust::get<1>(t);
    x = x < 0 ? -x : x;
    y = y < 0 ? -y : y;
    return x < y ? y - x : x - y;
  }
};

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t MST_solver<vertex_t, edge_t, weight_t>::alteration_max() {
  auto policy = rmm::exec_policy(stream);
  rmm::device_vector<weight_t> tmp(e);
  thrust::device_ptr<const weight_t> weights_ptr(weights);
  thrust::copy(policy->on(stream), weights_ptr, weights_ptr + e, tmp.begin());
  //sort tmp weights
  thrust::sort(policy->on(stream), tmp.begin(), tmp.end());

  //remove duplicates
  auto new_end = thrust::unique(policy->on(stream), tmp.begin(), tmp.end());

  //min(a[i+1]-a[i])/2
  auto begin =
    thrust::make_zip_iterator(thrust::make_tuple(tmp.begin(), tmp.begin() + 1));
  auto end =
    thrust::make_zip_iterator(thrust::make_tuple(new_end - 1, new_end));
  auto init = tmp[1] - tmp[0];
  auto max = thrust::transform_reduce(policy->on(stream), begin, end,
                                      alteration_functor<weight_t>(), init,
                                      thrust::minimum<weight_t>());
  return max / static_cast<weight_t>(2);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::alteration() {
  auto nthreads = std::min(v, max_threads);
  auto nblocks = std::min((v + nthreads - 1) / nthreads, max_blocks);

  // maximum alteration that does not change realtive weights order
  weight_t max = alteration_max();

  // pool of rand values
  rmm::device_vector<weight_t> rand_values(v);

  // Random number generator
  curandGenerator_t randGen;
  curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(randGen, 1234567);

  // Initialize rand values
  auto curand_status = curand_generate_uniformX(
    randGen, thrust::raw_pointer_cast(rand_values.data()), v);
  RAFT_EXPECTS(curand_status == CURAND_STATUS_SUCCESS, "MST: CURAND failed");
  curand_status = curandDestroyGenerator(randGen);
  RAFT_EXPECTS(curand_status == CURAND_STATUS_SUCCESS,
               "MST: CURAND cleanup failed");

  //Alterate the weights, make all undirected edge weight unique while keeping Wuv == Wvu
  detail::alteration_kernel<<<nblocks, nthreads, 0, stream>>>(
    v, e, offsets, indices, weights, max,
    thrust::raw_pointer_cast(rand_values.data()),
    thrust::raw_pointer_cast(altered_weights.data()));
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::label_prop(vertex_t* mst_src,
                                                        vertex_t* mst_dst) {
  // update the colors of both ends its until there is no change in colors
  thrust::host_vector<vertex_t> curr_mst_edge_count = mst_edge_count;

  auto min_pair_nthreads = std::min(v, max_threads);
  auto min_pair_nblocks =
    std::min((v + min_pair_nthreads - 1) / min_pair_nthreads, max_blocks);

  rmm::device_vector<bool> done(1, false);

  edge_t* new_mst_edge_ptr = thrust::raw_pointer_cast(new_mst_edge.data());
  vertex_t* color_index_ptr = thrust::raw_pointer_cast(color_index.data());
  vertex_t* next_color_ptr = thrust::raw_pointer_cast(next_color.data());

  bool* done_ptr = thrust::raw_pointer_cast(done.data());

  auto i = 0;
  while (!done[0]) {
    done[0] = true;

    detail::min_pair_colors<<<min_pair_nblocks, min_pair_nthreads, 0, stream>>>(
      v, indices, new_mst_edge_ptr, color, color_index_ptr, next_color_ptr);

    detail::update_colors<<<min_pair_nblocks, min_pair_nthreads, 0, stream>>>(
      v, color, color_index_ptr, next_color_ptr, done_ptr);
    i++;
  }

  detail::
    final_color_indices<<<min_pair_nblocks, min_pair_nthreads, 0, stream>>>(
      v, color, color_index_ptr);
#ifdef MST_TIME
  std::cout << "Label prop iterations: " << i << std::endl;
#endif
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::min_edge_per_vertex() {
  thrust::fill(min_edge_color.begin(), min_edge_color.end(),
               std::numeric_limits<weight_t>::max());

  int n_threads = 32;

  vertex_t* color_index_ptr = thrust::raw_pointer_cast(color_index.data());
  edge_t* new_mst_edge_ptr = thrust::raw_pointer_cast(new_mst_edge.data());
  bool* mst_edge_ptr = thrust::raw_pointer_cast(mst_edge.data());
  weight_t* min_edge_color_ptr =
    thrust::raw_pointer_cast(min_edge_color.data());
  weight_t* altered_weights_ptr =
    thrust::raw_pointer_cast(altered_weights.data());

  detail::kernel_min_edge_per_vertex<<<v, n_threads, 0, stream>>>(
    offsets, indices, altered_weights_ptr, color, color_index_ptr,
    new_mst_edge_ptr, mst_edge_ptr, min_edge_color_ptr, v);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::min_edge_per_supervertex() {
  int nthreads = std::min(v, max_threads);
  int nblocks = std::min((v + nthreads - 1) / nthreads, max_blocks);

  thrust::fill(temp_src.begin(), temp_src.end(),
               std::numeric_limits<vertex_t>::max());

  vertex_t* color_index_ptr = thrust::raw_pointer_cast(color_index.data());
  edge_t* new_mst_edge_ptr = thrust::raw_pointer_cast(new_mst_edge.data());
  bool* mst_edge_ptr = thrust::raw_pointer_cast(mst_edge.data());
  weight_t* min_edge_color_ptr =
    thrust::raw_pointer_cast(min_edge_color.data());
  weight_t* altered_weights_ptr =
    thrust::raw_pointer_cast(altered_weights.data());
  vertex_t* temp_src_ptr = thrust::raw_pointer_cast(temp_src.data());
  vertex_t* temp_dst_ptr = thrust::raw_pointer_cast(temp_dst.data());
  weight_t* temp_weights_ptr = thrust::raw_pointer_cast(temp_weights.data());

  detail::min_edge_per_supervertex<<<nblocks, nthreads, 0, stream>>>(
    color, color_index_ptr, new_mst_edge_ptr, mst_edge_ptr, indices, weights,
    altered_weights_ptr, temp_src_ptr, temp_dst_ptr, temp_weights_ptr,
    min_edge_color_ptr, v);

  // the above kernel only adds directed mst edges in the case where
  // a pair of vertices don't pick the same min edge between them
  // so, now we add the reverse edge to make it undirected
  detail::add_reverse_edge<<<nblocks, nthreads, 0, stream>>>(
    new_mst_edge_ptr, indices, weights, temp_src_ptr, temp_dst_ptr,
    temp_weights_ptr, v);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::check_termination() {
  int nthreads = std::min(2 * v, max_threads);
  int nblocks = std::min((2 * v + nthreads - 1) / nthreads, max_blocks);

  // count number of new mst edges

  vertex_t* mst_edge_count_ptr =
    thrust::raw_pointer_cast(mst_edge_count.data());
  vertex_t* temp_src_ptr = thrust::raw_pointer_cast(temp_src.data());

  detail::kernel_count_new_mst_edges<<<nblocks, nthreads, 0, stream>>>(
    temp_src_ptr, mst_edge_count_ptr, 2 * v);
}

template <typename vertex_t, typename weight_t>
struct new_edges_functor {
  __host__ __device__ bool operator()(
    const thrust::tuple<vertex_t, vertex_t, weight_t>& t) {
    auto src = thrust::get<0>(t);

    return src != std::numeric_limits<vertex_t>::max() ? true : false;
  }
};

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::append_src_dst_pair(
  vertex_t* mst_src, vertex_t* mst_dst, weight_t* mst_weights) {
  auto policy = rmm::exec_policy(stream);

  auto curr_mst_edge_count = prev_mst_edge_count[0];

  // iterator to end of mst edges added to final output in previous iteration
  auto src_dst_zip_end = thrust::make_zip_iterator(thrust::make_tuple(
    mst_src + curr_mst_edge_count, mst_dst + curr_mst_edge_count,
    mst_weights + curr_mst_edge_count));

  // iterator to new mst edges found
  auto temp_src_dst_zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
    temp_src.begin(), temp_dst.begin(), temp_weights.begin()));
  auto temp_src_dst_zip_end = thrust::make_zip_iterator(
    thrust::make_tuple(temp_src.end(), temp_dst.end(), temp_weights.end()));

  // copy new mst edges to final output
  thrust::copy_if(policy->on(stream), temp_src_dst_zip_begin,
                  temp_src_dst_zip_end, src_dst_zip_end,
                  new_edges_functor<vertex_t, weight_t>());
}

}  // namespace mst
}  // namespace raft
