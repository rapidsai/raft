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
#include "mst_kernels.cuh"
#include "utils.cuh"

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
// #include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <iostream>

namespace raft {
namespace mst {

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
  const raft::handle_t& handle_, vertex_t const* offsets_,
  vertex_t const* indices_, weight_t const* weights_, vertex_t const v_,
  vertex_t const e_)
  : handle(handle_),
    offsets(offsets_),
    indices(indices_),
    weights(weights_),
    alterated_weights(e_),
    v(v_),
    e(e_),
    color(v_),
    next_color(v_),
    active_color(v_),
    successor(v_, std::numeric_limits<vertex_t>::max()),
    mst_edge(e_, false),
    prev_mst_edge(e_, false),
    min_edge_color(v_),
    new_mst_edge(v_),
    msf_done(1, true) {
  max_blocks = handle_.get_device_properties().maxGridSize[0];
  max_threads = handle_.get_device_properties().maxThreadsPerBlock;
  sm_count = handle_.get_device_properties().multiProcessorCount;

  //Initially, color holds the vertex id as color
  thrust::sequence(color.begin(), color.end());
  //Initially, each next_color redirects to its own color
  thrust::sequence(next_color.begin(), next_color.end());
  //Initially, each edge is not in the mst
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::solve() {
  RAFT_EXPECTS(v > 0, "0 vertices");
  RAFT_EXPECTS(e > 0, "0 edges");
  RAFT_EXPECTS(offsets != nullptr, "Null offsets.");
  RAFT_EXPECTS(indices != nullptr, "Null indices.");
  RAFT_EXPECTS(weights != nullptr, "Null weights.");

  // Alterating the weights
  // this is done by identifying the lowest cost edge weight gap that is not 0, call this theta.
  // For each edge, add noise that is less than theta. That is, generate a random number in the range [0.0, theta) and add it to each edge weight.
  alteration();
  detail::printv(alterated_weights);

  // Boruvka original formulation says "while more than 1 supervertex remains"
  // Here we adjust it to support disconnected components (spanning forest)
  // track completion with mst_edge_found status.
  // should have max_iter ensure it always exits.
  for (auto i = 0; i < 3; i++) {
    std::cout << "Iteration: " << i << std::endl;
    // Finds the minimum outgoing edge from each supervertex to the lowest outgoing color
    // by working at each vertex of the supervertex
    min_edge_per_vertex();
    std::cout << "Successor Before: " << std::endl;
    detail::printv(successor);
    std::cout << "New MST Edge: " << std::endl;
    detail::printv(new_mst_edge);

    min_edge_per_supervertex();
    std::cout << "Successor After: " << std::endl;
    detail::printv(successor);

    // check if msf/mst done
    std::cout << "MST Edge: " << std::endl;
    detail::printv(mst_edge);
    std::cout << "Prev MST Edge: " << std::endl;
    detail::printv(prev_mst_edge);
    check_termination();
    std::cout << "MSF/MST done: " << msf_done[0] << std::endl;
    if (msf_done[0] == true) {
      break;
    }

    // updates colors of supervertices by propagating the lower color to the higher
    label_prop();
    std::cout << "Color: " << std::endl;
    detail::printv(color);
    std::cout << "Next Color: " << std::endl;
    detail::printv(next_color);

    // copy this iteration's results and store
    prev_mst_edge = mst_edge;
  }
}

//|b|-|a|
template <typename weight_t>
struct alteration_functor {
  __host__ __device__ weight_t
  operator()(const thrust::tuple<weight_t, weight_t>& t) {
    auto x = thrust::get<0>(t);
    auto y = thrust::get<1>(t);
    x < weight_t(0) ? -x : x;
    y < weight_t(0) ? -y : y;
    return y - x;
  }
};

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t MST_solver<vertex_t, edge_t, weight_t>::alteration_max() {
  auto stream = handle.get_stream();
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
  auto stream = handle.get_stream();
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
    thrust::raw_pointer_cast(alterated_weights.data()));
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::label_prop() {
  // update the colors of both ends its until there is no change in colors
  auto nthreads = std::min(v, max_threads);
  auto nblocks = std::min((v + nthreads - 1) / nthreads, max_blocks);
  auto stream = handle.get_stream();

  rmm::device_vector<bool> done(1, false);
  vertex_t* color_ptr = thrust::raw_pointer_cast(color.data());
  vertex_t* next_color_ptr = thrust::raw_pointer_cast(next_color.data());
  vertex_t* successor_ptr = thrust::raw_pointer_cast(successor.data());

  bool* done_ptr = thrust::raw_pointer_cast(done.data());

  auto i = 0;
  //std::cout << "==================" << std::endl;
  //detail::printv(color);
  while (!done[0]) {
    done[0] = true;
    detail::min_pair_colors<<<nblocks, nthreads, 0, stream>>>(
      v, successor_ptr, color_ptr, next_color_ptr);
    //detail::printv(next_color);
    detail::check_color_change<<<nblocks, nthreads, 0, stream>>>(
      v, color_ptr, next_color_ptr, done_ptr);
    //detail::printv(color);
    i++;
  }
  // std::cout << "Label prop iterations : " << i << std::endl;
  //std::cout << "==================" << std::endl;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::min_edge_per_vertex() {
  thrust::fill(new_mst_edge.begin(), new_mst_edge.end(), std::numeric_limits<edge_t>::max());
  thrust::fill(min_edge_color.begin(), min_edge_color.end(), std::numeric_limits<weight_t>::max());

  auto stream = handle.get_stream();
  int n_threads = 32;

  vertex_t* color_ptr = thrust::raw_pointer_cast(color.data());
  vertex_t* successor_ptr = thrust::raw_pointer_cast(successor.data());
  bool* mst_edge_ptr = thrust::raw_pointer_cast(mst_edge.data());
  edge_t *new_mst_edge_ptr = thrust::raw_pointer_cast(new_mst_edge.data());
  weight_t *min_edge_color_ptr = thrust::raw_pointer_cast(min_edge_color.data());

  detail::kernel_min_edge_per_vertex<<<v, n_threads, 0, stream>>>(
    offsets, indices, weights, color_ptr, successor_ptr, mst_edge_ptr, new_mst_edge_ptr, min_edge_color_ptr, v);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::min_edge_per_supervertex() {
  int nthreads = std::min(v, max_threads);
  int nblocks = std::min((v + nthreads - 1) / nthreads, max_blocks);
  auto stream = handle.get_stream();

  vertex_t* color_ptr = thrust::raw_pointer_cast(color.data());
  edge_t* new_mst_edge_ptr = thrust::raw_pointer_cast(new_mst_edge.data());
  vertex_t* successor_ptr = thrust::raw_pointer_cast(successor.data());
  bool* mst_edge_ptr = thrust::raw_pointer_cast(mst_edge.data());
  weight_t *min_edge_color_ptr = thrust::raw_pointer_cast(min_edge_color.data());
  
  // temp buffer to hold final min color weight
  // rmm::device_vector<weight_t> out_edge(v, std::numeric_limits<weight_t>::max());
  // weight_t *out_edge_ptr = thrust::raw_pointer_cast(out_edge.data());

  detail::min_edge_per_supervertex<<<nblocks, nthreads, 0, stream>>>(color_ptr, new_mst_edge_ptr, weights, successor_ptr, mst_edge_ptr, min_edge_color_ptr, v);

  // detail::printv(mst_edge);

  // auto pred_op = [] __device__ (const auto &x1, const auto &x2) {
  //   return x1 == x2;
  // }

  // auto min_op = [&weights] __device__ (const auto &x1, const auto &x2) {
  //   if (weights[x1] < weights[x2]) {
  //     return x1;
  //   }
  //   else {
  //     return x2;
  //   }
  // };

  // rmm::device_vector<vertex_t> out_color(v);
  // rmm::device_vector<edge_t> out_edge(v);

  // auto new_end = thrust::reduce_by_key(thrust::device, color.begin(), color.end(), new_mst_edge.begin(), out_color.begin(), out_edge.begin(), pred_op, min_op);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::check_termination() {
  int nthreads = std::min(e, max_threads);
  int nblocks = std::min((e + nthreads - 1) / nthreads, max_blocks);
  auto stream = handle.get_stream();

  bool* mst_edge_ptr = thrust::raw_pointer_cast(mst_edge.data());
  bool* prev_mst_edge_ptr = thrust::raw_pointer_cast(prev_mst_edge.data());

  bool *msf_done_ptr = thrust::raw_pointer_cast(msf_done.data());

  detail::kernel_check_termination<<<nblocks, nthreads, 0, stream>>>(e, mst_edge_ptr, prev_mst_edge_ptr, msf_done_ptr);
}

}  // namespace mst
}  // namespace raft
