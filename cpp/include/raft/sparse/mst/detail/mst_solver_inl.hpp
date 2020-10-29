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

#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

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
    v(v_),
    e(e_),
    color(v_),
    next_color(v_),
    active_color(v_),
    successor(v_),
    mst_edge(e_, false) {
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

  // Alternating the weights
  // this is done by identifying the lowest cost edge weight gap that is not 0, call this theta.
  // For each edge, add noise that is less than theta. That is, generate a random number in the range [0.0, theta) and add it to each edge weight.
  alteration();
  detail::printv(alterated_weights);

  // Boruvka original formulation says "while more than 1 supervertex remains"
  // Here we adjust it to support disconnected components (spanning forest)
  // track completion with mst_edge_found status.
  // should have max_iter ensure it always exits.
  for (auto i = 0; i < 2; i++) {
    // Finds the minimum outgoing edge from each supervertex to the lowest outgoing color
    // by working at each vertex of the supervertex
    min_edge_per_vertex();
    detail::printv(successor);
    // updates colors of supervertices by propagating the lower color to the higher
    label_prop();
    detail::printv(color);
  }
}

//|a|-|b|
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
  auto policy = rmm::exec_policy(stream)->on(stream);
  rmm::device_vector<weight_t> tmp(e);
  thrust::device_ptr<const weight_t> weights_ptr(weights);
  thrust::copy(policy, weights_ptr, weights_ptr + e, tmp.begin());
  detail::printv(tmp);
  //sort tmp weights
  thrust::sort(policy, tmp.begin(), tmp.begin());
  detail::printv(tmp);

  //remove duplicates
  auto new_end = thrust::unique(policy, tmp.begin(), tmp.end());
  detail::printv(tmp);

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

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::alteration() {
  auto stream = handle.get_stream();
  auto policy = rmm::exec_policy(handle.get_stream())->on(handle.get_stream());
  auto nthreads = std::min(v, max_threads);
  auto nblocks = std::min((v + nthreads - 1) / nthreads, max_blocks);

  // maximum alteration that does not change realtive weights order
  weight_t max = alteration_max();

  // pool of rand values
  rmm::device_vector<weight_t> rand_values(e / 2);

  // Random number generator
  curandGenerator_t randGen;
  curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(randGen, 1234567);

  // Initialize rand values
  auto curand_status = curand_generate_uniformX(
    randGen, thrust::raw_pointer_cast(rand_values.data()), e / 2);
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
  std::cout << "==================" << std::endl;
  detail::printv(color);
  while (!done[0]) {
    done[0] = true;
    detail::min_pair_colors<<<nblocks, nthreads, 0, stream>>>(
      v, successor_ptr, color_ptr, next_color_ptr);
    detail::printv(next_color);
    detail::check_color_change<<<nblocks, nthreads, 0, stream>>>(
      v, color_ptr, next_color_ptr, done_ptr);
    detail::printv(color);
    i++;
  }
  std::cout << "Label prop iterations : " << i << std::endl;
  std::cout << "==================" << std::endl;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::min_edge_per_vertex() {
  auto stream = handle.get_stream();
  int n_threads = 32;

  vertex_t* color_ptr = thrust::raw_pointer_cast(color.data());
  vertex_t* successor_ptr = thrust::raw_pointer_cast(successor.data());
  bool* mst_edge_ptr = thrust::raw_pointer_cast(mst_edge.data());
  weight_t* alterated_weights_ptr =
    thrust::raw_pointer_cast(alterated_weights.data());
  detail::kernel_min_edge_per_vertex<<<v, n_threads, 0, stream>>>(
    offsets, indices, alterated_weights_ptr, color_ptr, successor_ptr,
    mst_edge_ptr, v);
}

}  // namespace mst
}  // namespace raft
