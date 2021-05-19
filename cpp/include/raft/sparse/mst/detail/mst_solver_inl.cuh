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

#include <raft/cudart_utils.h>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <iostream>

#include <raft/cudart_utils.h>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

namespace raft {
namespace mst {
typedef std::chrono::high_resolution_clock Clock;

// curand generator uniform
inline curandStatus_t curand_generate_uniformX(curandGenerator_t generator,
                                               float* outputPtr, size_t n) {
  return curandGenerateUniform(generator, outputPtr, n);
}
inline curandStatus_t curand_generate_uniformX(curandGenerator_t generator,
                                               double* outputPtr, size_t n) {
  return curandGenerateUniformDouble(generator, outputPtr, n);
}

template <typename vertex_t, typename edge_t, typename weight_t>
MST_solver<vertex_t, edge_t, weight_t>::MST_solver(
  const raft::handle_t& handle_, const edge_t* offsets_,
  const vertex_t* indices_, const weight_t* weights_, const vertex_t v_,
  const edge_t e_, vertex_t* color_, cudaStream_t stream_,
  bool symmetrize_output_, bool initialize_colors_, int iterations_, int alpha_)
  : handle(handle_),
    offsets(offsets_),
    indices(indices_),
    weights(weights_),
    altered_weights(e_),
    v(v_),
    e(e_),
    color_index(color_),
    color(v_),
    next_color(v_),
    min_edge_color(v_),
    new_mst_edge(v_),
    mst_edge(e_, false),
    temp_src(2 * v_),
    temp_dst(2 * v_),
    temp_weights(2 * v_),
    mst_edge_count(1, 0),
    prev_mst_edge_count(1, 0),
    stream(stream_),
    symmetrize_output(symmetrize_output_),
    initialize_colors(initialize_colors_),
    iterations(iterations_),
    alpha(alpha_) {
  max_blocks = handle_.get_device_properties().maxGridSize[0];
  max_threads = handle_.get_device_properties().maxThreadsPerBlock;
  sm_count = handle_.get_device_properties().multiProcessorCount;

  //Initially, color holds the vertex id as color
  auto policy = rmm::exec_policy(stream);
  if (initialize_colors_) {
    thrust::sequence(policy, color.begin(), color.end(), 0);
    thrust::sequence(policy, color_index, color_index + v, 0);
  } else {
    raft::copy(color.data().get(), color_index, v, stream);
  }
  thrust::sequence(policy, next_color.begin(), next_color.end(), 0);
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

  // int start_1507, end_1507;
  // raft::update_host(&start_1507, offsets + 1507, 1, stream);
  // raft::update_host(&end_1507, offsets + 1508, 1, stream);

  // raft::print_device_vector("1507 indices", indices + start_1507, end_1507 - start_1507, std::cout);
  // raft::print_device_vector("1507 weights", weights + start_1507, end_1507 - start_1507, std::cout);

  // Alterating the weights
  // this is done by identifying the lowest cost edge weight gap that is not 0, call this theta.
  // For each edge, add noise that is less than theta. That is, generate a random number in the range [0.0, theta) and add it to each edge weight.
  alteration();

#ifdef MST_TIME
  auto stop = Clock::now();
  timer0 = duration_us(stop - start);
#endif

  auto n_expected_edges = symmetrize_output ? 2 * v - 2 : v - 1;

  Graph_COO<vertex_t, edge_t, weight_t> mst_result(n_expected_edges, stream);

  // Boruvka original formulation says "while more than 1 supervertex remains"
  // Here we adjust it to support disconnected components (spanning forest)
  // track completion with mst_edge_found status and v as upper bound
  auto mst_iterations = iterations > 0 ? iterations : v;
  for (auto i = 0; i < mst_iterations; i++) {
#ifdef MST_TIME
    start = Clock::now();
#endif
    // Finds the minimum edge from each vertex to the lowest color
    // by working at each vertex of the supervertex
    min_edge_per_vertex();

#ifdef MST_TIME
    stop = Clock::now();
    timer1 += duration_us(stop - start);
    start = Clock::now();
#endif
    // Finds the minimum edge from each supervertex to the lowest color
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
    // raft::print_device_vector("altered_weights", altered_weights.data().get(), e, std::cout);
    // raft::print_device_vector("new_mst_edge", new_mst_edge.data().get(), v, std::cout);
    // raft::print_device_vector("min_edge_color", min_edge_color.data().get(), v, std::cout);

    // raft::print_device_vector("temp_src", temp_src.data().get(), v, std::cout);
    // raft::print_device_vector("temp_dst", temp_dst.data().get(), v, std::cout);
    // raft::print_device_vector("temp_weights", temp_weights.data().get(), v, std::cout);

    auto curr_mst_edge_count = mst_edge_count[0];
    std::cout << "edge count: " << curr_mst_edge_count
              << ", expected: " << n_expected_edges << std::endl;
    RAFT_EXPECTS(curr_mst_edge_count <= n_expected_edges,
                 "Number of edges found by MST is invalid. This may be due to "
                 "loss in precision. Try increasing precision of weights.");

    if (prev_mst_edge_count[0] == curr_mst_edge_count) {
#ifdef MST_TIME
      std::cout << "Iterations: " << i << std::endl;
      std::cout << timer0 << "," << timer1 << "," << timer2 << "," << timer3
                << "," << timer4 << "," << timer5 << std::endl;
#endif
      // exit here when reaching steady state
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

    // updates colors of vertices by propagating the lower color to the higher
    label_prop(mst_result.src.data(), mst_result.dst.data());

#ifdef MST_TIME
    stop = Clock::now();
    timer5 += duration_us(stop - start);
#endif

    // copy this iteration's results and store
    prev_mst_edge_count = mst_edge_count;
  }

  // result packaging
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

// Compute the uper bound for the alteration
template <typename vertex_t, typename edge_t, typename weight_t>
double MST_solver<vertex_t, edge_t, weight_t>::alteration_max() {
  auto policy = rmm::exec_policy(stream);
  rmm::device_vector<weight_t> tmp(e);
  thrust::device_ptr<const weight_t> weights_ptr(weights);
  thrust::copy(policy, weights_ptr, weights_ptr + e, tmp.begin());
  //sort tmp weights
  thrust::sort(policy, tmp.begin(), tmp.end());

  //remove duplicates
  auto new_end = thrust::unique(policy, tmp.begin(), tmp.end());

  //min(a[i+1]-a[i])/2
  auto begin =
    thrust::make_zip_iterator(thrust::make_tuple(tmp.begin(), tmp.begin() + 1));
  auto end =
    thrust::make_zip_iterator(thrust::make_tuple(new_end - 1, new_end));
  auto init = tmp[1] - tmp[0];
  auto max =
    thrust::transform_reduce(policy, begin, end, alteration_functor<weight_t>(),
                             init, thrust::minimum<weight_t>());
  return max / static_cast<weight_t>(2);
}

// Compute the alteration to make all undirected edge weight unique
// Preserves weights order
template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::alteration() {
  auto nthreads = std::min(v, max_threads);
  auto nblocks = std::min((v + nthreads - 1) / nthreads, max_blocks);

  // maximum alteration that does not change realtive weights order
  double max = alteration_max();

  // pool of rand values
  rmm::device_vector<double> rand_values(v);

  // Random number generator
  curandGenerator_t randGen;
  curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(randGen, 1234567);

  // Initialize rand values
  auto curand_status =
    curand_generate_uniformX(randGen, rand_values.data().get(), v);
  RAFT_EXPECTS(curand_status == CURAND_STATUS_SUCCESS, "MST: CURAND failed");
  curand_status = curandDestroyGenerator(randGen);
  RAFT_EXPECTS(curand_status == CURAND_STATUS_SUCCESS,
               "MST: CURAND cleanup failed");

  std::cout << "Max: " << max << std::endl;
  // bool use_alpha = max < 1e-3 && sizeof(weight_t) == 4;
  bool use_alpha = true;

  //Alterate the weights, make all undirected edge weight unique while keeping Wuv == Wvu
  detail::alteration_kernel<<<nblocks, nthreads, 0, stream>>>(
    v, e, offsets, indices, weights, max, rand_values.data().get(),
    altered_weights.data().get(), alpha, use_alpha);
}

// updates colors of vertices by propagating the lower color to the higher
template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::label_prop(vertex_t* mst_src,
                                                        vertex_t* mst_dst) {
  // update the colors of both ends its until there is no change in colors
  thrust::host_vector<edge_t> curr_mst_edge_count = mst_edge_count;

  auto min_pair_nthreads = std::min(v, (vertex_t)max_threads);
  auto min_pair_nblocks = std::min(
    (v + min_pair_nthreads - 1) / min_pair_nthreads, (vertex_t)max_blocks);

  rmm::device_vector<bool> done(1, false);

  edge_t* new_mst_edge_ptr = new_mst_edge.data().get();
  vertex_t* color_ptr = color.data().get();
  vertex_t* next_color_ptr = next_color.data().get();

  bool* done_ptr = done.data().get();

  auto i = 0;
  while (!done[0]) {
    done[0] = true;

    detail::min_pair_colors<<<min_pair_nblocks, min_pair_nthreads, 0, stream>>>(
      v, indices, new_mst_edge_ptr, color_ptr, color_index, next_color_ptr);

    detail::update_colors<<<min_pair_nblocks, min_pair_nthreads, 0, stream>>>(
      v, color_ptr, color_index, next_color_ptr, done_ptr);
    i++;
  }

  detail::
    final_color_indices<<<min_pair_nblocks, min_pair_nthreads, 0, stream>>>(
      v, color_ptr, color_index);
#ifdef MST_TIME
  std::cout << "Label prop iterations: " << i << std::endl;
#endif
}

// Finds the minimum edge from each vertex to the lowest color
template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::min_edge_per_vertex() {
  auto policy = rmm::exec_policy(stream);
  thrust::fill(policy, min_edge_color.begin(), min_edge_color.end(),
               std::numeric_limits<weight_t>::max());
  thrust::fill(policy, new_mst_edge.begin(), new_mst_edge.end(),
               std::numeric_limits<weight_t>::max());

  int n_threads = 32;

  vertex_t* color_ptr = color.data().get();
  edge_t* new_mst_edge_ptr = new_mst_edge.data().get();
  bool* mst_edge_ptr = mst_edge.data().get();
  double* min_edge_color_ptr = min_edge_color.data().get();
  double* altered_weights_ptr = altered_weights.data().get();

  detail::kernel_min_edge_per_vertex<<<v, n_threads, 0, stream>>>(
    offsets, indices, altered_weights_ptr, color_ptr, color_index,
    new_mst_edge_ptr, mst_edge_ptr, min_edge_color_ptr, v);
}

// Finds the minimum edge from each supervertex to the lowest color
template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::min_edge_per_supervertex() {
  auto nthreads = std::min(v, max_threads);
  auto nblocks = std::min((v + nthreads - 1) / nthreads, max_blocks);

  auto policy = rmm::exec_policy(stream);
  thrust::fill(policy, temp_src.begin(), temp_src.end(),
               std::numeric_limits<vertex_t>::max());

  vertex_t* color_ptr = color.data().get();
  edge_t* new_mst_edge_ptr = new_mst_edge.data().get();
  bool* mst_edge_ptr = mst_edge.data().get();
  double* min_edge_color_ptr = min_edge_color.data().get();
  double* altered_weights_ptr = altered_weights.data().get();
  vertex_t* temp_src_ptr = temp_src.data().get();
  vertex_t* temp_dst_ptr = temp_dst.data().get();
  weight_t* temp_weights_ptr = temp_weights.data().get();

  detail::min_edge_per_supervertex<<<nblocks, nthreads, 0, stream>>>(
    color_ptr, color_index, new_mst_edge_ptr, mst_edge_ptr, indices, weights,
    altered_weights_ptr, temp_src_ptr, temp_dst_ptr, temp_weights_ptr,
    min_edge_color_ptr, v, symmetrize_output);

  // the above kernel only adds directed mst edges in the case where
  // a pair of vertices don't pick the same min edge between them
  // so, now we add the reverse edge to make it undirected
  if (symmetrize_output) {
    detail::add_reverse_edge<<<nblocks, nthreads, 0, stream>>>(
      new_mst_edge_ptr, indices, weights, temp_src_ptr, temp_dst_ptr,
      temp_weights_ptr, v, symmetrize_output);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::check_termination() {
  vertex_t nthreads = std::min(2 * v, (vertex_t)max_threads);
  vertex_t nblocks =
    std::min((2 * v + nthreads - 1) / nthreads, (vertex_t)max_blocks);

  // count number of new mst edges
  edge_t* mst_edge_count_ptr = mst_edge_count.data().get();
  vertex_t* temp_src_ptr = temp_src.data().get();

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
  thrust::copy_if(policy, temp_src_dst_zip_begin, temp_src_dst_zip_end,
                  src_dst_zip_end, new_edges_functor<vertex_t, weight_t>());
}

}  // namespace mst
}  // namespace raft
