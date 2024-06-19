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
RAFT_KERNEL kern_make_rev_graph(const IdxT* const dest_nodes,     // [graph_size]
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

template <typename IdxT = uint32_t,
          typename g_accessor =
            host_device_accessor<std::experimental::default_accessor<IdxT>, memory_type::host>>
void optimize(raft::resources const& res,
              mdspan<IdxT, matrix_extent<int64_t>, row_major, g_accessor> knn_graph,
              raft::host_matrix_view<IdxT, int64_t, row_major> new_graph)
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

  {
    //
    // Prune kNN graph
    //
    auto d_detour_count =
      raft::make_device_matrix<uint8_t, int64_t>(res, graph_size, input_graph_degree);

    RAFT_CUDA_TRY(cudaMemsetAsync(d_detour_count.data_handle(),
                                  0xff,
                                  graph_size * input_graph_degree * sizeof(uint8_t),
                                  resource::get_cuda_stream(res)));

    auto d_num_no_detour_edges = raft::make_device_vector<uint32_t, int64_t>(res, graph_size);
    RAFT_CUDA_TRY(cudaMemsetAsync(d_num_no_detour_edges.data_handle(),
                                  0x00,
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
          d_detour_count.data_handle(),
          d_num_no_detour_edges.data_handle(),
          dev_stats.data_handle());
      resource::sync_stream(res);
      RAFT_LOG_DEBUG(
        "# Pruning kNN Graph on GPUs (%.1lf %%)\r",
        (double)std::min<IdxT>((i_batch + 1) * batch_size, graph_size) / graph_size * 100);
    }
    resource::sync_stream(res);
    RAFT_LOG_DEBUG("\n");

    host_matrix_view_from_device<uint8_t, int64_t> detour_count(res, d_detour_count.view());

    raft::copy(
      host_stats.data_handle(), dev_stats.data_handle(), 2, resource::get_cuda_stream(res));
    const auto num_keep = host_stats.data_handle()[0];
    const auto num_full = host_stats.data_handle()[1];

    // Create pruned kNN graph
#pragma omp parallel for
    for (uint64_t i = 0; i < graph_size; i++) {
      // Find the `output_graph_degree` smallest detourable count nodes by checking the detourable
      // count of the neighbors while increasing the target detourable count from zero.
      uint64_t pk         = 0;
      uint32_t num_detour = 0;
      while (pk < output_graph_degree) {
        uint32_t next_num_detour = std::numeric_limits<uint32_t>::max();
        for (uint64_t k = 0; k < input_graph_degree; k++) {
          const auto num_detour_k = detour_count.data_handle()[k + (input_graph_degree * i)];
          // Find the detourable count to check in the next iteration
          if (num_detour_k > num_detour) {
            next_num_detour = std::min(static_cast<uint32_t>(num_detour_k), next_num_detour);
          }

          // Store the neighbor index if its detourable count is equal to `num_detour`.
          if (num_detour_k != num_detour) { continue; }
          output_graph_ptr[pk + (output_graph_degree * i)] =
            input_graph_ptr[k + (input_graph_degree * i)];
          pk += 1;
          if (pk >= output_graph_degree) break;
        }
        if (pk >= output_graph_degree) break;

        assert(next_num_detour != std::numeric_limits<uint32_t>::max());
        num_detour = next_num_detour;
      }
      RAFT_EXPECTS(pk == output_graph_degree,
                   "Couldn't find the output_graph_degree (%u) smallest detourable count nodes for "
                   "node %lu in the rank-based node reranking process",
                   output_graph_degree,
                   static_cast<uint64_t>(i));
    }

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

  auto rev_graph       = raft::make_host_matrix<IdxT, int64_t>(graph_size, output_graph_degree);
  auto rev_graph_count = raft::make_host_vector<uint32_t, int64_t>(graph_size);

  {
    //
    // Make reverse graph
    //
    const double time_make_start = cur_time();

    device_matrix_view_from_host<IdxT, int64_t> d_rev_graph(res, rev_graph.view());
    RAFT_CUDA_TRY(cudaMemsetAsync(d_rev_graph.data_handle(),
                                  0xff,
                                  graph_size * output_graph_degree * sizeof(IdxT),
                                  resource::get_cuda_stream(res)));

    auto d_rev_graph_count = raft::make_device_vector<uint32_t, int64_t>(res, graph_size);
    RAFT_CUDA_TRY(cudaMemsetAsync(d_rev_graph_count.data_handle(),
                                  0x00,
                                  graph_size * sizeof(uint32_t),
                                  resource::get_cuda_stream(res)));

    auto dest_nodes   = raft::make_host_vector<IdxT, int64_t>(graph_size);
    auto d_dest_nodes = raft::make_device_vector<IdxT, int64_t>(res, graph_size);

    for (uint64_t k = 0; k < output_graph_degree; k++) {
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        dest_nodes.data_handle()[i] = output_graph_ptr[k + (output_graph_degree * i)];
      }
      resource::sync_stream(res);

      raft::copy(d_dest_nodes.data_handle(),
                 dest_nodes.data_handle(),
                 graph_size,
                 resource::get_cuda_stream(res));

      dim3 threads(256, 1, 1);
      dim3 blocks(1024, 1, 1);
      kern_make_rev_graph<<<blocks, threads, 0, resource::get_cuda_stream(res)>>>(
        d_dest_nodes.data_handle(),
        d_rev_graph.data_handle(),
        d_rev_graph_count.data_handle(),
        graph_size,
        output_graph_degree);
      RAFT_LOG_DEBUG("# Making reverse graph on GPUs: %lu / %u    \r", k, output_graph_degree);
    }

    resource::sync_stream(res);
    RAFT_LOG_DEBUG("\n");

    if (d_rev_graph.allocated_memory()) {
      raft::copy(rev_graph.data_handle(),
                 d_rev_graph.data_handle(),
                 graph_size * output_graph_degree,
                 resource::get_cuda_stream(res));
    }
    raft::copy(rev_graph_count.data_handle(),
               d_rev_graph_count.data_handle(),
               graph_size,
               resource::get_cuda_stream(res));

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

    constexpr int _omp_chunk = 1024;
#pragma omp parallel for schedule(dynamic, _omp_chunk)
    for (uint64_t j = 0; j < graph_size; j++) {
      uint64_t k = std::min(rev_graph_count.data_handle()[j], output_graph_degree);
      while (k) {
        k--;
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
        const uint64_t j = output_graph_ptr[k + (output_graph_degree * i)];
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
}  // namespace raft::neighbors::cagra::detail
