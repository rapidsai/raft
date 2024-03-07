/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "utils.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <raft/util/bitonic_sort.cuh>
#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_fp16.h>

#include <float.h>
#include <omp.h>
#include <sys/time.h>

#include <cassert>
#include <climits>
#include <iostream>
#include <memory>
#include <random>

namespace raft::neighbors::cagra::detail {
namespace graph {

// unnamed namespace to avoid multiple definition error
namespace {
inline double cur_time(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return ((double)tv.tv_sec + (double)tv.tv_usec * 1e-6);
}

template <typename T>
__device__ inline void swap(T& val1, T& val2)
{
  T val0 = val1;
  val1   = val2;
  val2   = val0;
}

template <typename K, typename V>
__device__ inline bool swap_if_needed(K& key1, K& key2, V& val1, V& val2, bool ascending)
{
  if (key1 == key2) { return false; }
  if ((key1 > key2) == ascending) {
    swap<K>(key1, key2);
    swap<V>(val1, val2);
    return true;
  }
  return false;
}

template <class DATA_T, class IdxT, int numElementsPerThread>
RAFT_KERNEL kern_sort(const DATA_T* const dataset,  // [dataset_chunk_size, dataset_dim]
                      const IdxT dataset_size,
                      const uint32_t dataset_dim,
                      IdxT* const knn_graph,  // [graph_chunk_size, graph_degree]
                      const uint32_t graph_size,
                      const uint32_t graph_degree)
{
  const IdxT srcNode = (blockDim.x * blockIdx.x + threadIdx.x) / raft::WarpSize;
  if (srcNode >= graph_size) { return; }

  const uint32_t lane_id = threadIdx.x % raft::WarpSize;

  float my_keys[numElementsPerThread];
  IdxT my_vals[numElementsPerThread];

  // Compute distance from a src node to its neighbors
  for (int k = 0; k < graph_degree; k++) {
    const IdxT dstNode = knn_graph[k + static_cast<uint64_t>(graph_degree) * srcNode];
    float dist         = 0.0;
    for (int d = lane_id; d < dataset_dim; d += raft::WarpSize) {
      float diff = spatial::knn::detail::utils::mapping<float>{}(
                     dataset[d + static_cast<uint64_t>(dataset_dim) * srcNode]) -
                   spatial::knn::detail::utils::mapping<float>{}(
                     dataset[d + static_cast<uint64_t>(dataset_dim) * dstNode]);
      dist += diff * diff;
    }
    dist += __shfl_xor_sync(0xffffffff, dist, 1);
    dist += __shfl_xor_sync(0xffffffff, dist, 2);
    dist += __shfl_xor_sync(0xffffffff, dist, 4);
    dist += __shfl_xor_sync(0xffffffff, dist, 8);
    dist += __shfl_xor_sync(0xffffffff, dist, 16);
    if (lane_id == (k % raft::WarpSize)) {
      my_keys[k / raft::WarpSize] = dist;
      my_vals[k / raft::WarpSize] = dstNode;
    }
  }
  for (int k = graph_degree; k < raft::WarpSize * numElementsPerThread; k++) {
    if (lane_id == k % raft::WarpSize) {
      my_keys[k / raft::WarpSize] = utils::get_max_value<float>();
      my_vals[k / raft::WarpSize] = utils::get_max_value<IdxT>();
    }
  }

  // Sort by RAFT bitonic sort
  raft::util::bitonic<numElementsPerThread>(true).sort(my_keys, my_vals);

  // Update knn_graph
  for (int i = 0; i < numElementsPerThread; i++) {
    const int k = i * raft::WarpSize + lane_id;
    if (k < graph_degree) {
      knn_graph[k + (static_cast<uint64_t>(graph_degree) * srcNode)] = my_vals[i];
    }
  }
}

template <int MAX_DEGREE, class IdxT>
RAFT_KERNEL kern_prune(const IdxT* const knn_graph,  // [graph_chunk_size, graph_degree]
                       const uint32_t graph_size,
                       const uint32_t graph_degree,
                       const uint32_t degree,
                       const uint32_t batch_size,
                       const uint32_t batch_id,
                       uint8_t* const detour_count,          // [graph_chunk_size, graph_degree]
                       uint32_t* const num_no_detour_edges,  // [graph_size]
                       uint64_t* const stats)
{
  __shared__ uint32_t smem_num_detour[MAX_DEGREE];
  uint64_t* const num_retain = stats;
  uint64_t* const num_full   = stats + 1;

  const uint64_t nid = blockIdx.x + (batch_size * batch_id);
  if (nid >= graph_size) { return; }
  for (uint32_t k = threadIdx.x; k < graph_degree; k += blockDim.x) {
    smem_num_detour[k] = 0;
  }
  __syncthreads();

  const uint64_t iA = nid;
  if (iA >= graph_size) { return; }

  // Count number of detours for the edge A->B
  for (uint32_t kAD = 0; kAD < graph_degree - 1; kAD++) {
    const uint64_t iD = knn_graph[kAD + ((uint64_t)graph_degree * iA)];
    for (uint32_t kDB = threadIdx.x; kDB < graph_degree; kDB += blockDim.x) {
      const uint64_t iB_candidate = knn_graph[kDB + ((uint64_t)graph_degree * iD)];
      if (iB_candidate >= graph_size) {
        continue;  // Skip if ID of node-B candidate is invalid
      }
      for (uint32_t kAB = kAD + 1; kAB < graph_degree; kAB++) {
        const uint64_t iB = knn_graph[kAB + ((uint64_t)graph_degree * iA)];
        if (iB >= graph_size) {
          // If ID of node-B is invalid, always increase # detours of edge A->B.
          atomicAdd(smem_num_detour + kAB, 1);
          continue;
        }
        // if (kDB < kAB)
        {
          if (iB == iB_candidate) {
            atomicAdd(smem_num_detour + kAB, 1);
            break;
          }
        }
      }
    }
    __syncthreads();
  }

  uint32_t num_edges_no_detour = 0;
  for (uint32_t k = threadIdx.x; k < graph_degree; k += blockDim.x) {
    detour_count[k + (graph_degree * iA)] = min(smem_num_detour[k], (uint32_t)255);
    if (smem_num_detour[k] == 0) { num_edges_no_detour++; }
  }
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 1);
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 2);
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 4);
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 8);
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 16);
  num_edges_no_detour = min(num_edges_no_detour, degree);

  if (threadIdx.x == 0) {
    num_no_detour_edges[iA] = num_edges_no_detour;
    atomicAdd((unsigned long long int*)num_retain, (unsigned long long int)num_edges_no_detour);
    if (num_edges_no_detour >= degree) { atomicAdd((unsigned long long int*)num_full, 1); }
  }
}

template <class IdxT>
RAFT_KERNEL kern_make_rev_graph(const IdxT* const dest_nodes,     // [graph_size]
                                IdxT* const rev_graph,            // [graph_size, degree]
                                uint32_t* const rev_graph_count,  // [graph_size]
                                const uint32_t graph_size,
                                const uint32_t degree,
                                uint32_t reverse = 0)
{
  const uint32_t tid  = threadIdx.x + (blockDim.x * blockIdx.x);
  const uint32_t tnum = blockDim.x * gridDim.x;

  for (uint32_t i = tid; i < graph_size; i += tnum) {
    uint32_t src_id = i;
    if (reverse) {
      src_id = graph_size - 1 - i;
    }
    const IdxT dest_id = dest_nodes[src_id];
    if (dest_id >= graph_size) continue;
    if (rev_graph_count[dest_id] >= degree) continue;

    // Check if the same node already exists.
    bool flag_match = false;
    for (uint32_t k = 0; k < min(rev_graph_count[dest_id], degree); k++) {
      if (rev_graph[k + ((uint64_t)degree * dest_id)] == src_id) {
        flag_match = true;
        break;
      }
    }
    if (flag_match) continue;
    
    const uint32_t pos = atomicAdd(rev_graph_count + dest_id, 1);
    if (pos < degree) { rev_graph[pos + ((uint64_t)degree * dest_id)] = src_id; }
  }
}

template <class IdxT, class LabelT>
__device__ __host__ LabelT get_root_label(IdxT i, const LabelT *label)
{
  LabelT l = label[i];
  while (l != label[l]) {
    l = label[l];
  }
  return l;
}

template <class IdxT>
RAFT_KERNEL kern_mst_opt_update_graph(
    IdxT *mst_graph,                 // [graph_size, graph_degree]
    const IdxT *candidate_edges,     // [graph_size]
    IdxT *outgoing_num_edges,        // [graph_size]
    IdxT *incoming_num_edges,        // [graph_size]
    const IdxT *outgoing_max_edges,  // [graph_size]
    const IdxT *incoming_max_edges,  // [graph_size]
    const IdxT *label,               // [graph_size]
    const uint32_t graph_size,
    const uint32_t graph_degree,
    uint64_t *stats)
{
  const uint64_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= graph_size) return;

  int ret = 0;  // 0: No edge, 1: Direct edge, 2: Alternate edge, 3: Failure
    
  if ( outgoing_num_edges[i] >= outgoing_max_edges[i] ) return;
  uint64_t j = candidate_edges[i];
  if ( j >= graph_size ) return;
  const uint32_t ri = get_root_label(i, label);
  const uint32_t rj = get_root_label(j, label);
  if ( ri == rj ) return;

  // Try to add a direct edge to destination node with different label.
  if ( incoming_num_edges[j] < incoming_max_edges[j] ) {
    ret = 1;
    // Check to avoid duplication
    for (uint64_t kj = 0; kj < graph_degree; kj++) {
      uint64_t l = mst_graph[(graph_degree * j) + kj];
      if ( l >= graph_size ) continue;
      const uint32_t rl = get_root_label(l, label);
      if ( ri == rl ) {
        ret = 0;
        break;
      }
    }
    if ( ret == 0 ) return;

    ret = 0;
    auto kj = atomicAdd( incoming_num_edges + j, 1 );
    if ( kj < incoming_max_edges[j] ) {
      auto ki = outgoing_num_edges[i]++;
      mst_graph[(graph_degree * (i  ))   + ki] = j;  // outgoing
      mst_graph[(graph_degree * (j+1))-1 - kj] = i;  // incoming
      ret = 1;
    }
  }
  if ( ret > 0 ) {
    atomicAdd( (unsigned long long int*)stats + ret, 1 );
    return;
  }

  // Try to add an edge to an alternate node instead
  ret = 3;
  for (uint64_t kj = 0; kj < graph_degree; kj++) {
    uint64_t l = mst_graph[(graph_degree * (j+1))-1 - kj];
    if ( l >= graph_size ) continue;
    uint32_t rl = get_root_label(l, label);
    if ( ri == rl ) {
      ret = 0;
      break;
    }
    if ( incoming_num_edges[l] >= incoming_max_edges[l] ) continue;

    // Check to avoid duplication
    for (uint64_t kl = 0; kl < graph_degree; kl++) {
      uint64_t m = mst_graph[(graph_degree * l) + kl];
      if ( m > graph_size ) continue;
      uint32_t rm = get_root_label(m, label);
      if ( ri == rm ) {
        ret = 0;
        break;
      }
    }
    if ( ret == 0 ) {
      break;
    }

    auto kl = atomicAdd( incoming_num_edges + l, 1 );
    if ( kl < incoming_max_edges[l] ) {
      auto ki = outgoing_num_edges[i]++;
      mst_graph[(graph_degree * (i  ))   + ki] = l;  // outgoing
      mst_graph[(graph_degree * (l+1))-1 - kl] = i;  // incoming
      ret = 2;
      break;
    }
  }
  if ( ret > 0 ) {
    atomicAdd( (unsigned long long int*)stats + ret, 1 );
  }
}

template <class IdxT>
RAFT_KERNEL kern_mst_opt_labeling(
    IdxT *label,            // [graph_size]
    const IdxT *mst_graph,  // [graph_size, graph_degree]
    const uint32_t graph_size,
    const uint32_t graph_degree,
    uint64_t *stats)
{
  const uint64_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= graph_size) return;

  __shared__ uint32_t smem_updated[1];
  if (threadIdx.x == 0) {
    smem_updated[0] = 0;
  }
  __syncthreads();

  for (uint64_t ki = 0; ki < graph_degree; ki++ ) {
    uint64_t j = mst_graph[(graph_degree * i) + ki];
    if ( j >= graph_size ) continue;

    IdxT li = label[i];
    IdxT ri = get_root_label(i, label);
    if (ri < li) {
      atomicMin( label + i, ri );
    }
    IdxT lj = label[j];
    IdxT rj = get_root_label(j, label);
    if (rj < lj) {
      atomicMin( label + j, rj );
    }
    if (ri == rj) continue;

    if (ri > rj) {
      atomicCAS( label + i, ri, rj );
    } else if (rj > ri) {
      atomicCAS( label + j, rj, ri );
    }
    smem_updated[0] = 1;
  }

  __syncthreads();
  if ((threadIdx.x == 0) && (smem_updated[0] > 0)) {
    stats[0] = 1;
  }
}

template <class IdxT>
RAFT_KERNEL kern_mst_opt_cluster_size(
    IdxT *cluster_size,     // [graph_size]
    const IdxT *label,      // [graph_size]
    const uint32_t graph_size,
    uint64_t *stats)
{
  const uint64_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= graph_size) return;

  __shared__ uint64_t smem_num_clusters[1];
  if (threadIdx.x == 0) {
    smem_num_clusters[0] = 0;
  }
  __syncthreads();

  IdxT ri = get_root_label(i, label);
  if (ri == i) {
    atomicAdd( smem_num_clusters, (uint64_t) 1 );
  } else {
    atomicAdd( cluster_size + ri, cluster_size[i] );
    cluster_size[i] = 0;
  }

  __syncthreads();
  if ((threadIdx.x == 0) && (smem_num_clusters[0] > 0)) {
    atomicAdd( stats, smem_num_clusters[0] );
  }
}

template <class IdxT>
RAFT_KERNEL kern_mst_opt_postprocessing(
    IdxT *outgoing_num_edges,  // [graph_size]
    IdxT *incoming_num_edges,  // [graph_size]
    IdxT *outgoing_max_edges,  // [graph_size]
    IdxT *incoming_max_edges,  // [graph_size]
    const IdxT *cluster_size,  // [graph_size]
    const uint32_t graph_size,
    const uint32_t graph_degree,
    uint64_t *stats)
{
  const uint64_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= graph_size) return;

  __shared__ uint64_t smem_cluster_size_min[1];
  __shared__ uint64_t smem_cluster_size_max[1];
  __shared__ uint64_t smem_total_outgoing_edges[1];
  __shared__ uint64_t smem_total_incoming_edges[1];
  if (threadIdx.x == 0) {
    smem_cluster_size_min[0] = stats[0];
    smem_cluster_size_max[0] = stats[1];
    smem_total_outgoing_edges[0] = 0;
    smem_total_incoming_edges[0] = 0;
  }
  __syncthreads();
  
  // Adjust incoming_num_edges
  if (incoming_num_edges[i] > incoming_max_edges[i]) {
    incoming_num_edges[i] = incoming_max_edges[i];
  }
  
  // Calculate min/max of cluster_size
  if (cluster_size[i] > 0) {
    if (smem_cluster_size_min[0] > cluster_size[i]) {
      atomicMin( smem_cluster_size_min, (uint64_t)(cluster_size[i]) );
    }
    if (smem_cluster_size_max[0] < cluster_size[i]) {
      atomicMax( smem_cluster_size_max, (uint64_t)(cluster_size[i]) );
    }
  }
  
  // Calculate total number of outgoing/incoming edges
  atomicAdd( smem_total_outgoing_edges, (uint64_t)(outgoing_num_edges[i]) );
  atomicAdd( smem_total_incoming_edges, (uint64_t)(incoming_num_edges[i]) );
  
  // Adjust incoming/outgoing_max_edges
  if (outgoing_num_edges[i] == outgoing_max_edges[i]) {
    if (outgoing_num_edges[i] + incoming_num_edges[i] < graph_degree) {
      outgoing_max_edges[i] += 1;
      incoming_max_edges[i] -= 1;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMin( stats + 0, smem_cluster_size_min[0] );
    atomicMax( stats + 1, smem_cluster_size_max[0] );
    atomicAdd( stats + 2, smem_total_outgoing_edges[0] );
    atomicAdd( stats + 3, smem_total_incoming_edges[0] );
  }
}

template <class T>
uint64_t pos_in_array(T val, const T* array, uint64_t num)
{
  for (uint64_t i = 0; i < num; i++) {
    if (val == array[i]) { return i; }
  }
  return num;
}

template <class T>
void shift_array(T* array, uint64_t num)
{
  for (uint64_t i = num; i > 0; i--) {
    array[i] = array[i - 1];
  }
}
}  // namespace

template <typename DataT,
          typename IdxT = uint32_t,
          typename d_accessor =
            host_device_accessor<std::experimental::default_accessor<DataT>, memory_type::device>,
          typename g_accessor =
            host_device_accessor<std::experimental::default_accessor<IdxT>, memory_type::host>>
void sort_knn_graph(raft::resources const& res,
                    mdspan<const DataT, matrix_extent<int64_t>, row_major, d_accessor> dataset,
                    mdspan<IdxT, matrix_extent<int64_t>, row_major, g_accessor> knn_graph)
{
  RAFT_EXPECTS(dataset.extent(0) == knn_graph.extent(0),
               "dataset size is expected to have the same number of graph index size");
  const uint32_t dataset_size = dataset.extent(0);
  const uint32_t dataset_dim  = dataset.extent(1);
  const DataT* dataset_ptr    = dataset.data_handle();

  const IdxT graph_size             = dataset_size;
  const uint32_t input_graph_degree = knn_graph.extent(1);
  IdxT* const input_graph_ptr       = knn_graph.data_handle();

  auto d_input_graph = raft::make_device_matrix<IdxT, int64_t>(res, graph_size, input_graph_degree);

  //
  // Sorting kNN graph
  //
  const double time_sort_start = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN Graph on GPUs ");

  auto d_dataset = raft::make_device_matrix<DataT, int64_t>(res, dataset_size, dataset_dim);
  raft::copy(d_dataset.data_handle(),
             dataset_ptr,
             dataset_size * dataset_dim,
             resource::get_cuda_stream(res));

  raft::copy(d_input_graph.data_handle(),
             input_graph_ptr,
             graph_size * input_graph_degree,
             resource::get_cuda_stream(res));

  void (*kernel_sort)(
    const DataT* const, const IdxT, const uint32_t, IdxT* const, const uint32_t, const uint32_t);
  if (input_graph_degree <= 32) {
    constexpr int numElementsPerThread = 1;
    kernel_sort                        = kern_sort<DataT, IdxT, numElementsPerThread>;
  } else if (input_graph_degree <= 64) {
    constexpr int numElementsPerThread = 2;
    kernel_sort                        = kern_sort<DataT, IdxT, numElementsPerThread>;
  } else if (input_graph_degree <= 128) {
    constexpr int numElementsPerThread = 4;
    kernel_sort                        = kern_sort<DataT, IdxT, numElementsPerThread>;
  } else if (input_graph_degree <= 256) {
    constexpr int numElementsPerThread = 8;
    kernel_sort                        = kern_sort<DataT, IdxT, numElementsPerThread>;
  } else if (input_graph_degree <= 512) {
    constexpr int numElementsPerThread = 16;
    kernel_sort                        = kern_sort<DataT, IdxT, numElementsPerThread>;
  } else if (input_graph_degree <= 1024) {
    constexpr int numElementsPerThread = 32;
    kernel_sort                        = kern_sort<DataT, IdxT, numElementsPerThread>;
  } else {
    RAFT_FAIL(
      "The degree of input knn graph is too large (%u). "
      "It must be equal to or smaller than %d.",
      input_graph_degree,
      1024);
  }
  const auto block_size          = 256;
  const auto num_warps_per_block = block_size / raft::WarpSize;
  const auto grid_size           = (graph_size + num_warps_per_block - 1) / num_warps_per_block;

  RAFT_LOG_DEBUG(".");
  kernel_sort<<<grid_size, block_size, 0, resource::get_cuda_stream(res)>>>(
    d_dataset.data_handle(),
    dataset_size,
    dataset_dim,
    d_input_graph.data_handle(),
    graph_size,
    input_graph_degree);
  resource::sync_stream(res);
  RAFT_LOG_DEBUG(".");
  raft::copy(input_graph_ptr,
             d_input_graph.data_handle(),
             graph_size * input_graph_degree,
             resource::get_cuda_stream(res));
  RAFT_LOG_DEBUG("\n");

  const double time_sort_end = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN graph time: %.1lf sec\n", time_sort_end - time_sort_start);
}

template <typename IdxT = uint32_t>
void mst_optimization(raft::resources const& res,
                      raft::host_matrix_view<IdxT, int64_t, row_major> input_graph,
                      raft::host_matrix_view<IdxT, int64_t, row_major> output_graph,
                      raft::host_vector_view<uint32_t, int64_t> mst_graph_num_edges,
                      bool use_gpu = true)
{
  if (use_gpu) {
    RAFT_LOG_DEBUG("# MST optimization on GPU\n");
  } else {
    RAFT_LOG_DEBUG("# MST optimization on CPU\n");
  }
  const double time_mst_opt_start = cur_time();

  const IdxT graph_size = input_graph.extent(0);
  const uint32_t input_graph_degree = input_graph.extent(1);
  const uint32_t output_graph_degree = output_graph.extent(1);
  auto input_graph_ptr = input_graph.data_handle();
  auto output_graph_ptr = output_graph.data_handle();
  auto mst_graph_num_edges_ptr = mst_graph_num_edges.data_handle();

  // Allocate temporal arrays
  const uint32_t mst_graph_degree = output_graph_degree;
  auto mst_graph = raft::make_host_matrix<IdxT, int64_t>(graph_size, mst_graph_degree);
  auto outgoing_max_edges = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto incoming_max_edges = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto outgoing_num_edges = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto incoming_num_edges = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto label = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto cluster_size = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto candidate_edges = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto mst_graph_ptr = mst_graph.data_handle();
  auto outgoing_max_edges_ptr = outgoing_max_edges.data_handle();
  auto incoming_max_edges_ptr = incoming_max_edges.data_handle();
  auto outgoing_num_edges_ptr = outgoing_num_edges.data_handle();
  auto incoming_num_edges_ptr = incoming_num_edges.data_handle();
  auto label_ptr = label.data_handle();
  auto cluster_size_ptr = cluster_size.data_handle();
  auto candidate_edges_ptr = candidate_edges.data_handle();

  // Initialize arrays
#pragma omp parallel for
  for (uint64_t i = 0; i < graph_size; i++) {
    for (uint64_t k = 0; k < mst_graph_degree; k++) {
      mst_graph_ptr[(mst_graph_degree * i) + k] = graph_size;
    }
    outgoing_max_edges_ptr[i] = 2;
    incoming_max_edges_ptr[i] = mst_graph_degree - outgoing_max_edges_ptr[i];
    outgoing_num_edges_ptr[i] = 0;
    incoming_num_edges_ptr[i] = 0;
    label_ptr[i] = i;
    cluster_size_ptr[i] = 1;
  }

  // Allocate arrays on GPU
  uint32_t d_graph_size = graph_size;
  if (!use_gpu) {
    // (*) If GPU is not used, arrays of size 0 are created.
    d_graph_size = 0;
  }
  auto d_mst_graph_num_edges = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_mst_graph = raft::make_device_matrix<IdxT, int64_t>(res, d_graph_size, mst_graph_degree);
  auto d_outgoing_max_edges = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_incoming_max_edges = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_outgoing_num_edges = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_incoming_num_edges = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_label = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_cluster_size = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_candidate_edges = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_mst_graph_num_edges_ptr = d_mst_graph_num_edges.data_handle();
  auto d_mst_graph_ptr = d_mst_graph.data_handle();
  auto d_outgoing_max_edges_ptr = d_outgoing_max_edges.data_handle();
  auto d_incoming_max_edges_ptr = d_incoming_max_edges.data_handle();
  auto d_outgoing_num_edges_ptr = d_outgoing_num_edges.data_handle();
  auto d_incoming_num_edges_ptr = d_incoming_num_edges.data_handle();
  auto d_label_ptr = d_label.data_handle();
  auto d_cluster_size_ptr = d_cluster_size.data_handle();
  auto d_candidate_edges_ptr = d_candidate_edges.data_handle();

  constexpr int stats_size = 4;
  auto stats = raft::make_host_vector<uint64_t, int64_t>(stats_size);
  auto d_stats = raft::make_device_vector<uint64_t, int64_t>(res, stats_size);
  auto stats_ptr = stats.data_handle();
  auto d_stats_ptr = d_stats.data_handle();
  
  if (use_gpu) {
    raft::copy(d_mst_graph_ptr, mst_graph_ptr,
               (size_t) graph_size * mst_graph_degree, resource::get_cuda_stream(res));
    raft::copy(d_outgoing_num_edges_ptr, outgoing_num_edges_ptr,
               (size_t) graph_size, resource::get_cuda_stream(res));
    raft::copy(d_incoming_num_edges_ptr, incoming_num_edges_ptr,
               (size_t) graph_size, resource::get_cuda_stream(res));
    raft::copy(d_outgoing_max_edges_ptr, outgoing_max_edges_ptr,
               (size_t) graph_size, resource::get_cuda_stream(res));
    raft::copy(d_incoming_max_edges_ptr, incoming_max_edges_ptr,
               (size_t) graph_size, resource::get_cuda_stream(res));
    raft::copy(d_label_ptr, label_ptr,
               (size_t) graph_size, resource::get_cuda_stream(res));
    raft::copy(d_cluster_size_ptr, cluster_size_ptr,
               (size_t) graph_size, resource::get_cuda_stream(res));
  }

  uint32_t num_clusters = 0;
  uint32_t num_clusters_pre = graph_size;
  uint32_t cluster_size_min = graph_size;
  uint32_t cluster_size_max = 0;
  for (uint64_t k = 0; k <= input_graph_degree; k++) {
    int num_direct = 0;
    int num_alternate = 0;
    int num_failure = 0;

    // 1. Prepare candidate edges
    if (k == input_graph_degree) {
      // If the number of clusters does not converge to 1, then edges are
      // made from all nodes not belonging to the main cluster to any node
      // in the main cluster.
      uint32_t main_cluster_label = graph_size;
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        if ((cluster_size_ptr[i] == cluster_size_max) && (main_cluster_label > i)) {
          main_cluster_label = i;
        }
      }
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        candidate_edges_ptr[i] = graph_size;
        if (label_ptr[i] == main_cluster_label) continue;
        uint64_t j = i;
        while (label_ptr[j] != main_cluster_label) {
          constexpr uint32_t ofst = 97;
          j = (j + ofst) % graph_size;
        }
        candidate_edges_ptr[i] = j;
      }
    } else {
      // Copy rank-k edges from the input knn graph to 'candidate_edges'
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        candidate_edges_ptr[i] = input_graph_ptr[k + (input_graph_degree * i)];
      }
    }

    // 2. Update MST graph
    //  * Try to add candidate edges to MST graph
    if (use_gpu) {
      raft::copy(d_candidate_edges_ptr, candidate_edges_ptr,
                 graph_size, resource::get_cuda_stream(res));
      stats_ptr[0] = 0;
      stats_ptr[1] = num_direct;
      stats_ptr[2] = num_alternate;
      stats_ptr[3] = num_failure;
      raft::copy(d_stats_ptr, stats_ptr, 4, resource::get_cuda_stream(res));

      constexpr uint64_t n_threads = 256;
      const dim3 threads(n_threads, 1, 1);
      const dim3 blocks((graph_size + n_threads - 1) / n_threads, 1, 1);
      kern_mst_opt_update_graph<<<blocks, threads, 0, resource::get_cuda_stream(res)>>>
        (d_mst_graph_ptr,
         d_candidate_edges_ptr,
         d_outgoing_num_edges_ptr,
         d_incoming_num_edges_ptr,
         d_outgoing_max_edges_ptr,
         d_incoming_max_edges_ptr,
         d_label_ptr,
         graph_size, mst_graph_degree,
         d_stats_ptr);
      
      raft::copy(stats_ptr, d_stats_ptr, 4, resource::get_cuda_stream(res));
      resource::sync_stream(res);
      num_direct = stats_ptr[1];
      num_alternate = stats_ptr[2];
      num_failure = stats_ptr[3];
    } else {
#pragma omp parallel for reduction(+:num_direct,num_alternate,num_failure)
      for (uint64_t ii = 0; ii < graph_size; ii++) {
        uint64_t i = ii;
        if ( k % 2 == 0 ) {
          i = graph_size - (ii + 1);
        }
        int ret = 0;  // 0: No edge, 1: Direct edge, 2: Alternate edge, 3: Failure

        if ( outgoing_num_edges_ptr[i] >= outgoing_max_edges_ptr[i] ) continue;
        uint64_t j = candidate_edges_ptr[i];
        if ( j >= graph_size ) continue;
        if ( label_ptr[i] == label_ptr[j] ) continue;

        // Try to add a direct edge to destination node with different label.
        if ( incoming_num_edges_ptr[j] < incoming_max_edges_ptr[j] ) {
          ret = 1;
          // Check to avoid duplication
          for (uint64_t kj = 0; kj < mst_graph_degree; kj++) {
            uint64_t l = mst_graph_ptr[(mst_graph_degree * j) + kj];
            if ( l >= graph_size ) continue;
            if ( label_ptr[i] == label_ptr[l] ) {
              ret = 0;
              break;
            }
          }
          if ( ret == 0 ) continue;
          
          // Use atomic to avoid conflicts, since 'incoming_num_edges_ptr[j]'
          // can be updated by other threads.
          ret = 0;
          uint32_t kj;
#pragma omp atomic capture
          kj = incoming_num_edges_ptr[j]++;
          if ( kj < incoming_max_edges_ptr[j] ) {
            auto ki = outgoing_num_edges_ptr[i]++;
            mst_graph_ptr[(mst_graph_degree * (i  ))   + ki] = j;  // OUT
            mst_graph_ptr[(mst_graph_degree * (j+1))-1 - kj] = i;  // IN
            ret = 1;
          }
        }
        if ( ret == 1 ) {
          num_direct += 1;
          continue;
        }

        // Try to add an edge to an alternate node instead
        ret = 3;
        for (uint64_t kj = 0; kj < mst_graph_degree; kj++) {
          uint64_t l = mst_graph_ptr[(mst_graph_degree * (j+1))-1 - kj];
          if ( l >= graph_size ) continue;
          if ( label_ptr[i] == label_ptr[l] ) {
            ret = 0;
            break;
          }
          if ( incoming_num_edges_ptr[l] >= incoming_max_edges_ptr[l] ) continue;

          // Check to avoid duplication
          for (uint64_t kl = 0; kl < mst_graph_degree; kl++) {
            uint64_t m = mst_graph_ptr[(mst_graph_degree * l) + kl];
            if ( m > graph_size ) continue;
            if ( label_ptr[i] == label_ptr[m] ) {
              ret = 0;
              break;
            }
          }
          if ( ret == 0 ) {
            break;
          }

          // Use atomic to avoid conflicts, since 'incoming_num_edges_ptr[l]'
          // can be updated by other threads.
          uint32_t kl;
#pragma omp atomic capture
          kl = incoming_num_edges_ptr[l]++;
          if ( kl < incoming_max_edges_ptr[l] ) {
            auto ki = outgoing_num_edges_ptr[i]++;
            mst_graph_ptr[(mst_graph_degree * (i  ))   + ki] = l;  // OUT
            mst_graph_ptr[(mst_graph_degree * (l+1))-1 - kl] = i;  // IN
            ret = 2;
            break;
          }
        }
        if ( ret == 2 ) {
          num_alternate += 1;
        } else if ( ret == 3 ) {
          num_failure += 1;
        }
      }
    }

    // 3. Labeling
    uint32_t flag_update = 1;
    while (flag_update) {
      flag_update = 0;
      if (use_gpu) {
        stats_ptr[0] = flag_update;
        raft::copy(d_stats_ptr, stats_ptr, 1, resource::get_cuda_stream(res));

        constexpr uint64_t n_threads = 256;
        const dim3 threads(n_threads, 1, 1);
        const dim3 blocks((graph_size + n_threads - 1) / n_threads, 1, 1);
        kern_mst_opt_labeling<<<blocks, threads, 0, resource::get_cuda_stream(res)>>>(
          d_label_ptr, d_mst_graph_ptr, graph_size, mst_graph_degree, d_stats_ptr);

        raft::copy(stats_ptr, d_stats_ptr, 1, resource::get_cuda_stream(res));
        resource::sync_stream(res);
        flag_update = stats_ptr[0];
      } else {
#pragma omp parallel for reduction(+:flag_update)
        for (uint64_t i = 0; i < graph_size; i++) {
          for (uint64_t ki = 0; ki < mst_graph_degree; ki++ ) {
            uint64_t j = mst_graph_ptr[(mst_graph_degree * i) + ki];
            if ( j >= graph_size ) continue;
            if ( label_ptr[i] > label_ptr[j] ) {
              flag_update += 1;
              label_ptr[i] = label_ptr[j];
            }
          }
        }
      }
    }

    // 4. Calculate the number of clusters and the size of each cluster
    num_clusters = 0;
    if (use_gpu) {
      stats_ptr[0] = num_clusters;
      raft::copy(d_stats_ptr, stats_ptr, 1, resource::get_cuda_stream(res));

      constexpr uint64_t n_threads = 256;
      const dim3 threads(n_threads, 1, 1);
      const dim3 blocks((graph_size + n_threads - 1) / n_threads, 1, 1);
      kern_mst_opt_cluster_size<<<blocks, threads, 0, resource::get_cuda_stream(res)>>>(
        d_cluster_size_ptr, d_label_ptr, graph_size, d_stats_ptr);
      
      raft::copy(stats_ptr, d_stats_ptr, 1, resource::get_cuda_stream(res));
      resource::sync_stream(res);
      num_clusters = stats_ptr[0];
    } else {
#pragma omp parallel for reduction(+:num_clusters)
      for (uint64_t i = 0; i < graph_size; i++) {
        uint64_t ri = get_root_label(i, label_ptr);
        if (ri == i) {
          num_clusters += 1;
        } else {
#pragma omp atomic update
          cluster_size_ptr[ri] += cluster_size_ptr[i];
          cluster_size_ptr[i] = 0;
        }
      }
    }

    // 5. Postprocessings
    //  * Adjust incoming_num_edges
    //  * Calculate the min/max size of clusters.
    //  * Calculate the total number of outgoing/incoming edges
    //  * Increase the limit of outgoing edges as needed
    cluster_size_min = graph_size;
    cluster_size_max = 0;
    uint64_t total_outgoing_edges = 0;
    uint64_t total_incoming_edges = 0;
    if (use_gpu) {
      stats_ptr[0] = cluster_size_min;
      stats_ptr[1] = cluster_size_max;
      stats_ptr[2] = total_outgoing_edges;
      stats_ptr[3] = total_incoming_edges;
      raft::copy(d_stats_ptr, stats_ptr, 4, resource::get_cuda_stream(res));

      constexpr uint64_t n_threads = 256;
      const dim3 threads(n_threads, 1, 1);
      const dim3 blocks((graph_size + n_threads - 1) / n_threads, 1, 1);
      kern_mst_opt_postprocessing<<<blocks, threads, 0, resource::get_cuda_stream(res)>>>(
        d_outgoing_num_edges_ptr, d_incoming_num_edges_ptr,
        d_outgoing_max_edges_ptr, d_incoming_max_edges_ptr,
        d_cluster_size_ptr,
        graph_size, mst_graph_degree, d_stats_ptr);
      
      raft::copy(stats_ptr, d_stats_ptr, 4, resource::get_cuda_stream(res));
      resource::sync_stream(res);
      cluster_size_min = stats_ptr[0];
      cluster_size_max = stats_ptr[1];
      total_outgoing_edges = stats_ptr[2];
      total_incoming_edges = stats_ptr[3];
    } else {
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        if (incoming_num_edges_ptr[i] > incoming_max_edges_ptr[i]) {
          incoming_num_edges_ptr[i] = incoming_max_edges_ptr[i];
        }
      }

#pragma omp parallel for reduction(max:cluster_size_max) reduction(min:cluster_size_min)
      for (uint64_t i = 0; i < graph_size; i++) {
        if (cluster_size_ptr[i] == 0) continue;
        cluster_size_min = min(cluster_size_min, cluster_size_ptr[i]);
        cluster_size_max = max(cluster_size_max, cluster_size_ptr[i]);
      }

#pragma omp parallel for reduction(+:total_outgoing_edges, total_incoming_edges)
      for (uint64_t i = 0; i < graph_size; i++) {
        total_outgoing_edges += outgoing_num_edges_ptr[i];
        total_incoming_edges += incoming_num_edges_ptr[i];
      }

#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        if (outgoing_num_edges_ptr[i] < outgoing_max_edges_ptr[i]) continue;
        if (outgoing_num_edges_ptr[i] + incoming_num_edges_ptr[i] == mst_graph_degree) continue;
        assert(outgoing_num_edges_ptr[i] + incoming_num_edges_ptr[i] < mst_graph_degree);
        outgoing_max_edges_ptr[i] += 1;
        incoming_max_edges_ptr[i] = mst_graph_degree - outgoing_max_edges_ptr[i];
      }
    }

    // 6. Show stats
    if ( num_clusters != num_clusters_pre ) {
      std::string msg = "# k: " + std::to_string(k);
      msg += ", num_clusters: " + std::to_string(num_clusters);
      msg += ", cluster_size: " + std::to_string(cluster_size_min) + " to " + std::to_string(cluster_size_max);
      msg += ", total_num_edges: " + std::to_string(total_outgoing_edges) + ", " + std::to_string(total_incoming_edges);
      if ( num_alternate + num_failure > 0 ) {
        msg += ", altenate: " + std::to_string(num_alternate);
        if ( num_failure > 0 ) {
          msg += ", failure: " + std::to_string(num_failure);
        }
      }
      RAFT_LOG_DEBUG( "%s\n", msg.c_str() );
    }
    assert( num_clusters > 0 );
    assert( total_outgoing_edges == total_incoming_edges );
    if ( num_clusters == 1 ) {
      break;
    }
    num_clusters_pre = num_clusters;
  }

  // The edges that make up the MST are stored as edges in the output graph.
  if (use_gpu) {
    raft::copy(mst_graph_ptr, d_mst_graph_ptr,
               (size_t) graph_size * mst_graph_degree, resource::get_cuda_stream(res));
    resource::sync_stream(res);
  }
#pragma omp parallel for
  for (uint64_t i = 0; i < graph_size; i++) {
    uint64_t k = 0;
    for (uint64_t kj = 0; kj < mst_graph_degree; kj++) {
      uint64_t j = mst_graph_ptr[(mst_graph_degree * i) + kj];
      if (j >= graph_size) continue;

      // Check to avoid duplication
      auto flag_match = false;
      for (uint64_t ki = 0; ki < k; ki++) {
        if (j == output_graph_ptr[(output_graph_degree * i) + ki]) {
          flag_match = true;
          break;
        }
      }
      if (flag_match) continue;
        
      output_graph_ptr[(output_graph_degree * i) + k] = j;
      k += 1;
    }
    mst_graph_num_edges_ptr[i] = k;
  }

  const double time_mst_opt_end = cur_time();
  RAFT_LOG_DEBUG( "# MST optimization time: %.1lf sec\n", time_mst_opt_end - time_mst_opt_start );
}

template <typename IdxT = uint32_t,
          typename g_accessor =
            host_device_accessor<std::experimental::default_accessor<IdxT>, memory_type::host>>
void optimize(raft::resources const& res,
              mdspan<IdxT, matrix_extent<int64_t>, row_major, g_accessor> knn_graph,
              raft::host_matrix_view<IdxT, int64_t, row_major> new_graph,
              const bool use_mst = false)
{
  RAFT_LOG_DEBUG(
    "# Pruning kNN graph (size=%lu, degree=%lu)\n", knn_graph.extent(0), knn_graph.extent(1));

  RAFT_EXPECTS(knn_graph.extent(0) == new_graph.extent(0),
               "Each input array is expected to have the same number of rows");
  RAFT_EXPECTS(new_graph.extent(1) <= knn_graph.extent(1),
               "output graph cannot have more columns than input graph");
  const uint32_t input_graph_degree  = knn_graph.extent(1);
  const uint32_t output_graph_degree = new_graph.extent(1);
  auto input_graph_ptr               = knn_graph.data_handle();
  auto output_graph_ptr              = new_graph.data_handle();
  const IdxT graph_size              = new_graph.extent(0);

  // MST optimization
  auto mst_graph_num_edges = raft::make_host_vector<uint32_t, int64_t>(graph_size);
  auto mst_graph_num_edges_ptr = mst_graph_num_edges.data_handle();
#pragma omp parallel for
  for (uint64_t i = 0; i < graph_size; i++) {
    mst_graph_num_edges_ptr[i] = 0;
  }
  if (use_mst) {
    constexpr bool use_gpu = true;
    mst_optimization(
      res,
      raft::make_host_matrix_view<IdxT, int64_t>(input_graph_ptr, graph_size, input_graph_degree),
      raft::make_host_matrix_view<IdxT, int64_t>(output_graph_ptr, graph_size, output_graph_degree),
      mst_graph_num_edges.view(), use_gpu );

    for (uint64_t i = 0; i < graph_size; i++) {
      if (i < 8 || i >= graph_size - 8) {
        RAFT_LOG_DEBUG( "# mst_graph_num_edges_ptr[%lu]: %u\n", i, mst_graph_num_edges_ptr[i] );
      }
    }
  }
  
  auto pruned_graph = raft::make_host_matrix<uint32_t, int64_t>(graph_size, output_graph_degree);
  auto pruned_graph_ptr = pruned_graph.data_handle();
  {
    //
    // Prune kNN graph
    //
    auto d_detour_count =
      raft::make_device_matrix<uint8_t, int64_t>(res, graph_size, input_graph_degree);
    auto d_detour_count_ptr = d_detour_count.data_handle();
    RAFT_CUDA_TRY(cudaMemsetAsync(d_detour_count_ptr, 0xff,
                                  graph_size * input_graph_degree * sizeof(uint8_t),
                                  resource::get_cuda_stream(res)));

    auto d_num_no_detour_edges = raft::make_device_vector<uint32_t, int64_t>(res, graph_size);
    auto d_num_no_detour_edges_ptr = d_num_no_detour_edges.data_handle();
    RAFT_CUDA_TRY(cudaMemsetAsync(d_num_no_detour_edges_ptr, 0x00,
                                  graph_size * sizeof(uint32_t),
                                  resource::get_cuda_stream(res)));

    auto dev_stats  = raft::make_device_vector<uint64_t>(res, 2);
    auto host_stats = raft::make_host_vector<uint64_t>(2);

    //
    // Prune unimportant edges.
    //
    // The edge to be retained is determined without explicitly considering
    // distance or angle. Suppose the edge is the k-th edge of some node-A to
    // node-B (A->B). Among the edges originating at node-A, there are k-1 edges
    // shorter than the edge A->B. Each of these k-1 edges are connected to a
    // different k-1 nodes. Among these k-1 nodes, count the number of nodes with
    // edges to node-B, which is the number of 2-hop detours for the edge A->B.
    // Once the number of 2-hop detours has been counted for all edges, the
    // specified number of edges are picked up for each node, starting with the
    // edge with the lowest number of 2-hop detours.
    //
    const double time_prune_start = cur_time();
    RAFT_LOG_DEBUG("# Pruning kNN Graph on GPUs\r");

    // Copy input_graph_ptr over to device if necessary
    device_matrix_view_from_host d_input_graph(
      res,
      raft::make_host_matrix_view<IdxT, int64_t>(input_graph_ptr, graph_size, input_graph_degree));

    constexpr int MAX_DEGREE = 1024;
    if (input_graph_degree > MAX_DEGREE) {
      RAFT_FAIL(
        "The degree of input knn graph is too large (%u). "
        "It must be equal to or smaller than %d.",
        input_graph_degree,
        1024);
    }
    const uint32_t batch_size =
      std::min(static_cast<uint32_t>(graph_size), static_cast<uint32_t>(256 * 1024));
    const uint32_t num_batch = (graph_size + batch_size - 1) / batch_size;
    const dim3 threads_prune(32, 1, 1);
    const dim3 blocks_prune(batch_size, 1, 1);

    RAFT_CUDA_TRY(cudaMemsetAsync(
      dev_stats.data_handle(), 0, sizeof(uint64_t) * 2, resource::get_cuda_stream(res)));

    for (uint32_t i_batch = 0; i_batch < num_batch; i_batch++) {
      kern_prune<MAX_DEGREE, IdxT>
        <<<blocks_prune, threads_prune, 0, resource::get_cuda_stream(res)>>>(
          d_input_graph.data_handle(),
          graph_size,
          input_graph_degree,
          output_graph_degree,
          batch_size,
          i_batch,
          d_detour_count_ptr,
          d_num_no_detour_edges_ptr,
          dev_stats.data_handle());
      resource::sync_stream(res);
      RAFT_LOG_DEBUG(
        "# Pruning kNN Graph on GPUs (%.1lf %%)\r",
        (double)std::min<IdxT>((i_batch + 1) * batch_size, graph_size) / graph_size * 100);
    }
    resource::sync_stream(res);
    RAFT_LOG_DEBUG("\n");

    host_matrix_view_from_device<uint8_t, int64_t> detour_count(res, d_detour_count.view());
    auto detour_count_ptr = detour_count.data_handle();

    raft::copy(
      host_stats.data_handle(), dev_stats.data_handle(), 2, resource::get_cuda_stream(res));
    const auto num_keep = host_stats.data_handle()[0];
    const auto num_full = host_stats.data_handle()[1];

    // Create pruned kNN graph
    uint32_t max_detour = 0;
#pragma omp parallel for reduction(max : max_detour)
    for (uint64_t i = 0; i < graph_size; i++) {
      uint64_t pk = 0;
      for (uint32_t num_detour = 0; num_detour < output_graph_degree; num_detour++) {
        if (max_detour < num_detour) { max_detour = num_detour; /* stats */ }
        for (uint64_t k = 0; k < input_graph_degree; k++) {
          if (detour_count_ptr[k + (input_graph_degree * i)] != num_detour) { continue; }
          pruned_graph_ptr[pk + (output_graph_degree * i)] =
            input_graph_ptr[k + (input_graph_degree * i)];
          pk += 1;
          if (pk >= output_graph_degree) break;
        }
        if (pk >= output_graph_degree) break;
      }
      assert(pk == output_graph_degree);
    }
    // RAFT_LOG_DEBUG("# max_detour: %u\n", max_detour);

    const double time_prune_end = cur_time();
    RAFT_LOG_DEBUG(
      "# Pruning time: %.1lf sec, "
      "avg_no_detour_edges_per_node: %.2lf/%u, "
      "nodes_with_no_detour_at_all_edges: %.1lf%%\n",
      time_prune_end - time_prune_start,
      (double)num_keep / graph_size,
      output_graph_degree,
      (double)num_full / graph_size * 100);
  }

  auto rev_graph_count = raft::make_host_vector<uint32_t, int64_t>(graph_size);
  auto rev_graph_count_ptr = rev_graph_count.data_handle();
#pragma omp parallel for
  for (uint64_t i = 0; i < graph_size; i++) {
    rev_graph_count_ptr[i] = 0;
  }
  
  auto rev_graph = raft::make_host_matrix<IdxT, int64_t>(graph_size, output_graph_degree);
  auto rev_graph_ptr = rev_graph.data_handle();
#pragma omp parallel for
  for (uint64_t i = 0; i < (uint64_t)graph_size * output_graph_degree; i++) {
    rev_graph_ptr[i] = graph_size;
  }

  {
    //
    // Make reverse graph
    //
    const double time_make_start = cur_time();

    device_matrix_view_from_host<IdxT, int64_t> d_rev_graph(res, rev_graph.view());
    auto d_rev_graph_ptr = d_rev_graph.data_handle();
    raft::copy(d_rev_graph_ptr, rev_graph_ptr,
               (size_t) graph_size * output_graph_degree,
               resource::get_cuda_stream(res));

    auto d_rev_graph_count = raft::make_device_vector<uint32_t, int64_t>(res, graph_size);
    auto d_rev_graph_count_ptr = d_rev_graph_count.data_handle();
    raft::copy(d_rev_graph_count_ptr, rev_graph_count_ptr,
               graph_size, resource::get_cuda_stream(res));

    auto dest_nodes = raft::make_host_vector<IdxT, int64_t>(graph_size);
    auto dest_nodes_ptr = dest_nodes.data_handle();
    auto d_dest_nodes = raft::make_device_vector<IdxT, int64_t>(res, graph_size);
    auto d_dest_nodes_ptr = d_dest_nodes.data_handle();

    for (uint64_t k = 0; k < output_graph_degree; k++) {
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        dest_nodes_ptr[i] = pruned_graph_ptr[k + (output_graph_degree * i)];
      }
      resource::sync_stream(res);

      raft::copy(d_dest_nodes_ptr, dest_nodes_ptr,
                 graph_size, resource::get_cuda_stream(res));

      dim3 threads(256, 1, 1);
      dim3 blocks(1024, 1, 1);
      kern_make_rev_graph<<<blocks, threads, 0, resource::get_cuda_stream(res)>>>(
        d_dest_nodes_ptr,
        d_rev_graph_ptr,
        d_rev_graph_count_ptr,
        graph_size,
        output_graph_degree, (k % 2));
      RAFT_LOG_DEBUG("# Making reverse graph on GPUs: %lu / %u    \r", k, output_graph_degree);
    }

    resource::sync_stream(res);
    RAFT_LOG_DEBUG("\n");

    if (d_rev_graph.allocated_memory()) {
      raft::copy(rev_graph_ptr, d_rev_graph_ptr,
                 (size_t)graph_size * output_graph_degree,
                 resource::get_cuda_stream(res));
    }
    raft::copy(rev_graph_count_ptr, d_rev_graph_count_ptr,
               graph_size, resource::get_cuda_stream(res));

    resource::sync_stream(res);
    const double time_make_end = cur_time();
    RAFT_LOG_DEBUG("# Making reverse graph time: %.1lf sec", time_make_end - time_make_start);
  }

  {
    //
    // Merge forward and reverse edges
    //
    const double time_replace_start = cur_time();
#pragma omp parallel for
    for (uint64_t i = 0; i < graph_size; i++) {
      auto my_fwd_graph = pruned_graph_ptr + (output_graph_degree * i);
      auto my_rev_graph = rev_graph_ptr + (output_graph_degree * i);
      auto my_out_graph = output_graph_ptr + (output_graph_degree * i);
      uint32_t kf = 0;
      uint32_t kr = 0;
      uint32_t k = mst_graph_num_edges_ptr[i];
      while (kf < output_graph_degree || kr < output_graph_degree) {
	if (kf < output_graph_degree) {
	  if (my_fwd_graph[kf] < graph_size) {
	    auto flag_match = false;
	    for (uint32_t kk = 0; kk < k; kk++) {
	      if (my_out_graph[kk] == my_fwd_graph[kf]) {
		flag_match = true;
		break;
	      }
	    }
	    if (!flag_match) {
	      my_out_graph[k] = my_fwd_graph[kf];
	      k += 1;
	    }
	  }
	  kf += 1;
	  if (k >= output_graph_degree) break;
	}
	if (kr < output_graph_degree) {
	  if (my_rev_graph[kr] < graph_size) {
	    auto flag_match = false;
	    for (uint32_t kk = 0; kk < k; kk++) {
	      if (my_out_graph[kk] == my_rev_graph[kr]) {
		flag_match = true;
		break;
	      }
	    }
	    if (!flag_match) {
	      my_out_graph[k] = my_rev_graph[kr];
	      k += 1;
	    }
	  }
	  kr += 1;
	  if (k >= output_graph_degree) break;
	}
      }
      assert(k == output_graph_degree);
      assert(kf <= output_graph_degree);
      assert(kr <= output_graph_degree);
    }

    const double time_replace_end = cur_time();
    RAFT_LOG_DEBUG("# Replacing edges time: %.1lf sec", time_replace_end - time_replace_start);

    /* stats */
    uint64_t num_replaced_edges = 0;
#pragma omp parallel for reduction(+ : num_replaced_edges)
    for (uint64_t i = 0; i < graph_size; i++) {
      for (uint64_t k = 0; k < output_graph_degree; k++) {
        const uint64_t j = pruned_graph_ptr[k + (output_graph_degree * i)];
        const uint64_t pos =
          pos_in_array<IdxT>(j, output_graph_ptr + (output_graph_degree * i), output_graph_degree);
        if (pos == output_graph_degree) { num_replaced_edges += 1; }
      }
    }
    RAFT_LOG_DEBUG("# Average number of replaced edges per node: %.2f",
                   (double)num_replaced_edges / graph_size);
  }

  // Check number of incoming edges
  {
    auto in_edge_count = raft::make_host_vector<uint32_t, int64_t>(graph_size);
    auto in_edge_count_ptr = in_edge_count.data_handle();
#pragma omp parallel for
    for (uint64_t i = 0; i < graph_size; i++) {
      in_edge_count_ptr[i] = 0;
    }
#pragma omp parallel for
    for (uint64_t i = 0; i < graph_size; i++) {
      for (uint64_t k = 0; k < output_graph_degree; k++) {
        const uint64_t j = output_graph_ptr[k + (output_graph_degree * i)];
        if (j >= graph_size) continue;
#pragma omp atomic
        in_edge_count_ptr[j] += 1;
      }
    }
    auto hist = raft::make_host_vector<uint32_t, int64_t>(output_graph_degree);
    auto hist_ptr = hist.data_handle();
    for (uint64_t k = 0; k < output_graph_degree; k++) {
      hist_ptr[k] = 0;
    }
#pragma omp parallel for
    for (uint64_t i = 0; i < graph_size; i++) {
      uint32_t count = in_edge_count_ptr[i];
      if (count >= output_graph_degree) continue;
#pragma omp atomic
      hist_ptr[count] += 1;
    }
    RAFT_LOG_DEBUG( "# Histogram for number of incoming edges\n" );
    uint32_t sum_hist = 0;
    for ( uint64_t k = 0; k < output_graph_degree; k++ ) {
      sum_hist += hist_ptr[k];
      RAFT_LOG_DEBUG( "# %3lu, %8u, %lf, (%8u, %lf)\n", k,
                     hist_ptr[k],(double) hist_ptr[k] / graph_size,
                     sum_hist, (double) sum_hist / graph_size );
    }
  }
}

}  // namespace graph
}  // namespace raft::neighbors::cagra::detail
