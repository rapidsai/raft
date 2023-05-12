/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cassert>
#include <climits>
#include <cuda_fp16.h>
#include <float.h>
#include <iostream>
#include <memory>
#include <omp.h>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <random>
#include <sys/time.h>

#include <raft/util/cuda_rt_essentials.hpp>

namespace raft::neighbors::experimental::cagra::detail {
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

template <class DATA_T, class IdxT, int blockDim_x, int numElementsPerThread>
__global__ void kern_sort(const DATA_T* const dataset,  // [dataset_chunk_size, dataset_dim]
                          const IdxT dataset_size,
                          const uint32_t dataset_dim,
                          IdxT* const knn_graph,  // [graph_chunk_size, graph_degree]
                          const uint32_t graph_size,
                          const uint32_t graph_degree)
{
  __shared__ float smem_keys[blockDim_x * numElementsPerThread];
  __shared__ IdxT smem_vals[blockDim_x * numElementsPerThread];

  const IdxT srcNode = blockIdx.x;
  if (srcNode >= graph_size) { return; }

  const uint32_t num_warps = blockDim_x / 32;
  const uint32_t warp_id   = threadIdx.x / 32;
  const uint32_t lane_id   = threadIdx.x % 32;

  // Compute distance from a src node to its neighbors
  for (int k = warp_id; k < graph_degree; k += num_warps) {
    const IdxT dstNode = knn_graph[k + ((uint64_t)graph_degree * srcNode)];
    float dist         = 0.0;
    for (int d = lane_id; d < dataset_dim; d += 32) {
      float diff = spatial::knn::detail::utils::mapping<float>{}(
                     dataset[d + ((uint64_t)dataset_dim * srcNode)]) -
                   spatial::knn::detail::utils::mapping<float>{}(
                     dataset[d + ((uint64_t)dataset_dim * dstNode)]);
      dist += diff * diff;
    }
    dist += __shfl_xor_sync(0xffffffff, dist, 1);
    dist += __shfl_xor_sync(0xffffffff, dist, 2);
    dist += __shfl_xor_sync(0xffffffff, dist, 4);
    dist += __shfl_xor_sync(0xffffffff, dist, 8);
    dist += __shfl_xor_sync(0xffffffff, dist, 16);
    if (lane_id == 0) {
      smem_keys[k] = dist;
      smem_vals[k] = dstNode;
    }
  }
  __syncthreads();

  float my_keys[numElementsPerThread];
  IdxT my_vals[numElementsPerThread];
  for (int i = 0; i < numElementsPerThread; i++) {
    const int k = i + (numElementsPerThread * threadIdx.x);
    if (k < graph_degree) {
      my_keys[i] = smem_keys[k];
      my_vals[i] = smem_vals[k];
    } else {
      my_keys[i] = FLT_MAX;
      my_vals[i] = ~static_cast<IdxT>(0);
    }
  }
  __syncthreads();

  // Sorting by thread
  uint32_t mask        = 1;
  const bool ascending = ((threadIdx.x & mask) == 0);
  for (int j = 0; j < numElementsPerThread; j += 2) {
#pragma unroll
    for (int i = 0; i < numElementsPerThread; i += 2) {
      swap_if_needed<float, IdxT>(
        my_keys[i], my_keys[i + 1], my_vals[i], my_vals[i + 1], ascending);
    }
#pragma unroll
    for (int i = 1; i < numElementsPerThread - 1; i += 2) {
      swap_if_needed<float, IdxT>(
        my_keys[i], my_keys[i + 1], my_vals[i], my_vals[i + 1], ascending);
    }
  }

  // Bitonic Sorting
  while (mask < blockDim_x) {
    const uint32_t next_mask = mask << 1;

    for (uint32_t curr_mask = mask; curr_mask > 0; curr_mask >>= 1) {
      const bool ascending = ((threadIdx.x & curr_mask) == 0) == ((threadIdx.x & next_mask) == 0);
      if (mask >= 32) {
        // inter warp
        __syncthreads();
#pragma unroll
        for (int i = 0; i < numElementsPerThread; i++) {
          smem_keys[threadIdx.x + (blockDim_x * i)] = my_keys[i];
          smem_vals[threadIdx.x + (blockDim_x * i)] = my_vals[i];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < numElementsPerThread; i++) {
          float opp_key = smem_keys[(threadIdx.x ^ curr_mask) + (blockDim_x * i)];
          IdxT opp_val  = smem_vals[(threadIdx.x ^ curr_mask) + (blockDim_x * i)];
          swap_if_needed<float, IdxT>(my_keys[i], opp_key, my_vals[i], opp_val, ascending);
        }
      } else {
// intra warp
#pragma unroll
        for (int i = 0; i < numElementsPerThread; i++) {
          float opp_key = __shfl_xor_sync(0xffffffff, my_keys[i], curr_mask);
          IdxT opp_val  = __shfl_xor_sync(0xffffffff, my_vals[i], curr_mask);
          swap_if_needed<float, IdxT>(my_keys[i], opp_key, my_vals[i], opp_val, ascending);
        }
      }
    }

    const bool ascending = ((threadIdx.x & next_mask) == 0);
#pragma unroll
    for (uint32_t curr_mask = numElementsPerThread / 2; curr_mask > 0; curr_mask >>= 1) {
#pragma unroll
      for (int i = 0; i < numElementsPerThread; i++) {
        int j = i ^ curr_mask;
        if (i > j) continue;
        swap_if_needed<float, IdxT>(my_keys[i], my_keys[j], my_vals[i], my_vals[j], ascending);
      }
    }
    mask = next_mask;
  }

  // Update knn_graph
  for (int i = 0; i < numElementsPerThread; i++) {
    const int k = i + (numElementsPerThread * threadIdx.x);
    if (k < graph_degree) {
      knn_graph[k + (static_cast<uint64_t>(graph_degree) * srcNode)] = my_vals[i];
    }
  }
}

template <int MAX_DEGREE, class IdxT>
__global__ void kern_prune(const IdxT* const knn_graph,  // [graph_chunk_size, graph_degree]
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

  // count number of detours (A->D->B)
  for (uint32_t kAD = 0; kAD < graph_degree - 1; kAD++) {
    const uint64_t iD = knn_graph[kAD + (graph_degree * iA)];
    for (uint32_t kDB = threadIdx.x; kDB < graph_degree; kDB += blockDim.x) {
      const uint64_t iB_candidate = knn_graph[kDB + ((uint64_t)graph_degree * iD)];
      for (uint32_t kAB = kAD + 1; kAB < graph_degree; kAB++) {
        // if ( kDB < kAB )
        {
          const uint64_t iB = knn_graph[kAB + (graph_degree * iA)];
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
__global__ void kern_make_rev_graph(const IdxT* const dest_nodes,     // [graph_size]
                                    IdxT* const rev_graph,            // [size, degree]
                                    uint32_t* const rev_graph_count,  // [graph_size]
                                    const uint32_t graph_size,
                                    const uint32_t degree)
{
  const uint32_t tid  = threadIdx.x + (blockDim.x * blockIdx.x);
  const uint32_t tnum = blockDim.x * gridDim.x;

  for (uint32_t src_id = tid; src_id < graph_size; src_id += tnum) {
    const IdxT dest_id = dest_nodes[src_id];
    if (dest_id >= graph_size) continue;

    const uint32_t pos = atomicAdd(rev_graph_count + dest_id, 1);
    if (pos < degree) { rev_graph[pos + ((uint64_t)degree * dest_id)] = src_id; }
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
void sort_knn_graph(raft::device_resources const& res,
                    mdspan<const DataT, matrix_extent<IdxT>, row_major, d_accessor> dataset,
                    mdspan<IdxT, matrix_extent<IdxT>, row_major, g_accessor> knn_graph)
{
  RAFT_EXPECTS(dataset.extent(0) == knn_graph.extent(0),
               "dataset size is expected to have the same number of graph index size");
  const uint32_t dataset_size = dataset.extent(0);
  const uint32_t dataset_dim  = dataset.extent(1);
  const DataT* dataset_ptr    = dataset.data_handle();

  const IdxT graph_size             = dataset_size;
  const uint32_t input_graph_degree = knn_graph.extent(1);
  IdxT* const input_graph_ptr       = knn_graph.data_handle();

  auto d_input_graph = raft::make_device_matrix<IdxT, IdxT>(res, graph_size, input_graph_degree);

  //
  // Sorting kNN graph
  //
  const double time_sort_start = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN Graph on GPUs ");

  auto d_dataset = raft::make_device_matrix<DataT, IdxT>(res, dataset_size, dataset_dim);
  raft::copy(d_dataset.data_handle(), dataset_ptr, dataset_size * dataset_dim, res.get_stream());

  raft::copy(d_input_graph.data_handle(),
             input_graph_ptr,
             graph_size * input_graph_degree,
             res.get_stream());

  void (*kernel_sort)(
    const DataT* const, const IdxT, const uint32_t, IdxT* const, const uint32_t, const uint32_t);
  constexpr int numElementsPerThread = 4;
  dim3 threads_sort(1, 1, 1);
  if (input_graph_degree <= numElementsPerThread * 32) {
    constexpr int blockDim_x = 32;
    kernel_sort              = kern_sort<DataT, IdxT, blockDim_x, numElementsPerThread>;
    threads_sort.x           = blockDim_x;
  } else if (input_graph_degree <= numElementsPerThread * 64) {
    constexpr int blockDim_x = 64;
    kernel_sort              = kern_sort<DataT, IdxT, blockDim_x, numElementsPerThread>;
    threads_sort.x           = blockDim_x;
  } else if (input_graph_degree <= numElementsPerThread * 128) {
    constexpr int blockDim_x = 128;
    kernel_sort              = kern_sort<DataT, IdxT, blockDim_x, numElementsPerThread>;
    threads_sort.x           = blockDim_x;
  } else if (input_graph_degree <= numElementsPerThread * 256) {
    constexpr int blockDim_x = 256;
    kernel_sort              = kern_sort<DataT, IdxT, blockDim_x, numElementsPerThread>;
    threads_sort.x           = blockDim_x;
  } else {
    RAFT_LOG_ERROR(
      "[ERROR] The degree of input knn graph is too large (%u). "
      "It must be equal to or small than %d.\n",
      input_graph_degree,
      numElementsPerThread * 256);
    exit(-1);
  }
  dim3 blocks_sort(graph_size, 1, 1);
  RAFT_LOG_DEBUG(".");
  kernel_sort<<<blocks_sort, threads_sort, 0, res.get_stream()>>>(d_dataset.data_handle(),
                                                                  dataset_size,
                                                                  dataset_dim,
                                                                  d_input_graph.data_handle(),
                                                                  graph_size,
                                                                  input_graph_degree);
  res.sync_stream();
  RAFT_LOG_DEBUG(".");
  raft::copy(input_graph_ptr,
             d_input_graph.data_handle(),
             graph_size * input_graph_degree,
             res.get_stream());
  RAFT_LOG_DEBUG("\n");

  const double time_sort_end = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN graph time: %.1lf sec\n", time_sort_end - time_sort_start);
}

template <typename IdxT = uint32_t,
          typename g_accessor =
            host_device_accessor<std::experimental::default_accessor<IdxT>, memory_type::host>>
void prune(raft::device_resources const& res,
           mdspan<IdxT, matrix_extent<IdxT>, row_major, g_accessor> knn_graph,
           raft::host_matrix_view<IdxT, IdxT, row_major> new_graph)
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

  auto pruned_graph = raft::make_host_matrix<IdxT, IdxT>(graph_size, output_graph_degree);

  {
    //
    // Prune kNN graph
    //
    auto d_input_graph = raft::make_device_matrix<IdxT, IdxT>(res, graph_size, input_graph_degree);

    auto detour_count = raft::make_host_matrix<uint8_t, IdxT>(graph_size, input_graph_degree);
    auto d_detour_count =
      raft::make_device_matrix<uint8_t, IdxT>(res, graph_size, input_graph_degree);
    RAFT_CUDA_TRY(cudaMemsetAsync(d_detour_count.data_handle(),
                                  0xff,
                                  graph_size * input_graph_degree * sizeof(uint8_t),
                                  res.get_stream()));

    auto d_num_no_detour_edges = raft::make_device_vector<uint32_t, IdxT>(res, graph_size);
    RAFT_CUDA_TRY(cudaMemsetAsync(
      d_num_no_detour_edges.data_handle(), 0x00, graph_size * sizeof(uint32_t), res.get_stream()));

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

    raft::copy(d_input_graph.data_handle(),
               input_graph_ptr,
               graph_size * input_graph_degree,
               res.get_stream());
    void (*kernel_prune)(const IdxT* const,
                         const uint32_t,
                         const uint32_t,
                         const uint32_t,
                         const uint32_t,
                         const uint32_t,
                         uint8_t* const,
                         uint32_t* const,
                         uint64_t* const);

    constexpr int MAX_DEGREE = 1024;
    if (input_graph_degree <= MAX_DEGREE) {
      kernel_prune = kern_prune<MAX_DEGREE, IdxT>;
    } else {
      RAFT_LOG_ERROR(
        "[ERROR] The degree of input knn graph is too large (%u). "
        "It must be equal to or small than %d.\n",
        input_graph_degree,
        1024);
      exit(-1);
    }
    const uint32_t batch_size =
      std::min(static_cast<uint32_t>(graph_size), static_cast<uint32_t>(256 * 1024));
    const uint32_t num_batch = (graph_size + batch_size - 1) / batch_size;
    const dim3 threads_prune(32, 1, 1);
    const dim3 blocks_prune(batch_size, 1, 1);

    RAFT_CUDA_TRY(
      cudaMemsetAsync(dev_stats.data_handle(), 0, sizeof(uint64_t) * 2, res.get_stream()));

    for (uint32_t i_batch = 0; i_batch < num_batch; i_batch++) {
      kernel_prune<<<blocks_prune, threads_prune, 0, res.get_stream()>>>(
        d_input_graph.data_handle(),
        graph_size,
        input_graph_degree,
        output_graph_degree,
        batch_size,
        i_batch,
        d_detour_count.data_handle(),
        d_num_no_detour_edges.data_handle(),
        dev_stats.data_handle());
      res.sync_stream();
      RAFT_LOG_DEBUG(
        "# Pruning kNN Graph on GPUs (%.1lf %%)\r",
        (double)std::min<IdxT>((i_batch + 1) * batch_size, graph_size) / graph_size * 100);
    }
    res.sync_stream();
    RAFT_LOG_DEBUG("\n");

    raft::copy(detour_count.data_handle(),
               d_detour_count.data_handle(),
               graph_size * input_graph_degree,
               res.get_stream());

    raft::copy(host_stats.data_handle(), dev_stats.data_handle(), 2, res.get_stream());
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
          if (detour_count.data_handle()[k + (input_graph_degree * i)] != num_detour) { continue; }
          pruned_graph.data_handle()[pk + (output_graph_degree * i)] =
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

  auto rev_graph       = raft::make_host_matrix<IdxT, IdxT>(graph_size, output_graph_degree);
  auto rev_graph_count = raft::make_host_vector<uint32_t, IdxT>(graph_size);

  {
    //
    // Make reverse graph
    //
    const double time_make_start = cur_time();

    auto d_rev_graph = raft::make_device_matrix<IdxT, IdxT>(res, graph_size, output_graph_degree);
    RAFT_CUDA_TRY(cudaMemsetAsync(d_rev_graph.data_handle(),
                                  0xff,
                                  graph_size * output_graph_degree * sizeof(IdxT),
                                  res.get_stream()));

    auto d_rev_graph_count = raft::make_device_vector<uint32_t, IdxT>(res, graph_size);
    RAFT_CUDA_TRY(cudaMemsetAsync(
      d_rev_graph_count.data_handle(), 0x00, graph_size * sizeof(uint32_t), res.get_stream()));

    auto dest_nodes   = raft::make_host_vector<IdxT, IdxT>(graph_size);
    auto d_dest_nodes = raft::make_device_vector<IdxT, IdxT>(res, graph_size);

    for (uint64_t k = 0; k < output_graph_degree; k++) {
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        dest_nodes.data_handle()[i] = pruned_graph.data_handle()[k + (output_graph_degree * i)];
      }
      res.sync_stream();

      raft::copy(
        d_dest_nodes.data_handle(), dest_nodes.data_handle(), graph_size, res.get_stream());

      dim3 threads(256, 1, 1);
      dim3 blocks(1024, 1, 1);
      kern_make_rev_graph<<<blocks, threads, 0, res.get_stream()>>>(d_dest_nodes.data_handle(),
                                                                    d_rev_graph.data_handle(),
                                                                    d_rev_graph_count.data_handle(),
                                                                    graph_size,
                                                                    output_graph_degree);
      RAFT_LOG_DEBUG("# Making reverse graph on GPUs: %lu / %u    \r", k, output_graph_degree);
    }

    res.sync_stream();
    RAFT_LOG_DEBUG("\n");

    raft::copy(rev_graph.data_handle(),
               d_rev_graph.data_handle(),
               graph_size * output_graph_degree,
               res.get_stream());
    raft::copy(
      rev_graph_count.data_handle(), d_rev_graph_count.data_handle(), graph_size, res.get_stream());

    const double time_make_end = cur_time();
    RAFT_LOG_DEBUG("# Making reverse graph time: %.1lf sec", time_make_end - time_make_start);
  }

  {
    //
    // Replace some edges with reverse edges
    //
    const double time_replace_start = cur_time();

    const uint64_t num_protected_edges = output_graph_degree / 2;
    RAFT_LOG_DEBUG("# num_protected_edges: %lu", num_protected_edges);

    memcpy(output_graph_ptr,
           pruned_graph.data_handle(),
           sizeof(uint32_t) * graph_size * output_graph_degree);

    constexpr int _omp_chunk = 1024;
#pragma omp parallel for schedule(dynamic, _omp_chunk)
    for (uint64_t j = 0; j < graph_size; j++) {
      for (uint64_t _k = 0; _k < rev_graph_count.data_handle()[j]; _k++) {
        uint64_t k = rev_graph_count.data_handle()[j] - 1 - _k;
        uint64_t i = rev_graph.data_handle()[k + (output_graph_degree * j)];

        uint64_t pos =
          pos_in_array<IdxT>(i, output_graph_ptr + (output_graph_degree * j), output_graph_degree);
        if (pos < num_protected_edges) { continue; }
        uint64_t num_shift = pos - num_protected_edges;
        if (pos == output_graph_degree) {
          num_shift = output_graph_degree - num_protected_edges - 1;
        }
        shift_array<IdxT>(output_graph_ptr + num_protected_edges + (output_graph_degree * j),
                          num_shift);
        output_graph_ptr[num_protected_edges + (output_graph_degree * j)] = i;
      }
      if ((omp_get_thread_num() == 0) && ((j % _omp_chunk) == 0)) {
        RAFT_LOG_DEBUG("# Replacing reverse edges: %lu / %lu    ", j, graph_size);
      }
    }
    RAFT_LOG_DEBUG("\n");

    const double time_replace_end = cur_time();
    RAFT_LOG_DEBUG("# Replacing edges time: %.1lf sec", time_replace_end - time_replace_start);

    /* stats */
    uint64_t num_replaced_edges = 0;
#pragma omp parallel for reduction(+ : num_replaced_edges)
    for (uint64_t i = 0; i < graph_size; i++) {
      for (uint64_t k = 0; k < output_graph_degree; k++) {
        const uint64_t j = pruned_graph.data_handle()[k + (output_graph_degree * i)];
        const uint64_t pos =
          pos_in_array<IdxT>(j, output_graph_ptr + (output_graph_degree * i), output_graph_degree);
        if (pos == output_graph_degree) { num_replaced_edges += 1; }
      }
    }
    RAFT_LOG_DEBUG("# Average number of replaced edges per node: %.2f",
                   (double)num_replaced_edges / graph_size);
  }
}

}  // namespace graph
}  // namespace raft::neighbors::experimental::cagra::detail
