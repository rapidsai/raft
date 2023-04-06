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

template <class T>
__host__ __device__ float compute_norm2(const T* a,
                                        const T* b,
                                        const std::size_t dim,
                                        const float scale)
{
  float sum = 0.f;
  for (std::size_t j = 0; j < dim; j++) {
    const auto diff = a[j] * scale - b[j] * scale;
    sum += diff * diff;
  }
  return sum;
}

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

template <class DATA_T, int blockDim_x, int numElementsPerThread>
__global__ void kern_sort(
  DATA_T** dataset,  // [num_gpus][dataset_chunk_size, dataset_dim]
  uint32_t dataset_size,
  uint32_t dataset_chunk_size,  // (*) num_gpus * dataset_chunk_size >= dataset_size
  uint32_t dataset_dim,
  float scale,
  uint32_t** knn_graph,  // [num_gpus][graph_chunk_size, graph_degree]
  uint32_t graph_size,
  uint32_t graph_chunk_size,  // (*) num_gpus * graph_chunk_size >= graph_size
  uint32_t graph_degree,
  int dev_id)
{
  __shared__ float smem_keys[blockDim_x * numElementsPerThread];
  __shared__ uint32_t smem_vals[blockDim_x * numElementsPerThread];

  uint64_t srcNode     = blockIdx.x + ((uint64_t)graph_chunk_size * dev_id);
  uint64_t srcNode_dev = srcNode / graph_chunk_size;
  uint64_t srcNode_loc = srcNode % graph_chunk_size;
  if (srcNode >= graph_size) { return; }

  const uint32_t num_warps = blockDim_x / 32;
  const uint32_t warp_id   = threadIdx.x / 32;
  const uint32_t lane_id   = threadIdx.x % 32;

  // Compute distance from a src node to its neighbors
  for (int k = warp_id; k < graph_degree; k += num_warps) {
    uint64_t dstNode     = knn_graph[srcNode_dev][k + ((uint64_t)graph_degree * srcNode_loc)];
    uint64_t dstNode_dev = dstNode / graph_chunk_size;
    uint64_t dstNode_loc = dstNode % graph_chunk_size;
    float dist           = 0.0;
    for (int d = lane_id; d < dataset_dim; d += 32) {
      float diff =
        (float)(dataset[srcNode_dev][d + ((uint64_t)dataset_dim * srcNode_loc)]) * scale -
        (float)(dataset[dstNode_dev][d + ((uint64_t)dataset_dim * dstNode_loc)]) * scale;
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
  uint32_t my_vals[numElementsPerThread];
  for (int i = 0; i < numElementsPerThread; i++) {
    int k = i + (numElementsPerThread * threadIdx.x);
    if (k < graph_degree) {
      my_keys[i] = smem_keys[k];
      my_vals[i] = smem_vals[k];
    } else {
      my_keys[i] = FLT_MAX;
      my_vals[i] = 0xffffffffU;
    }
  }
  __syncthreads();

  // Sorting by thread
  uint32_t mask  = 1;
  bool ascending = ((threadIdx.x & mask) == 0);
  for (int j = 0; j < numElementsPerThread; j += 2) {
#pragma unroll
    for (int i = 0; i < numElementsPerThread; i += 2) {
      swap_if_needed<float, uint32_t>(
        my_keys[i], my_keys[i + 1], my_vals[i], my_vals[i + 1], ascending);
    }
#pragma unroll
    for (int i = 1; i < numElementsPerThread - 1; i += 2) {
      swap_if_needed<float, uint32_t>(
        my_keys[i], my_keys[i + 1], my_vals[i], my_vals[i + 1], ascending);
    }
  }

  // Bitonic Sorting
  while (mask < blockDim_x) {
    uint32_t next_mask = mask << 1;

    for (uint32_t curr_mask = mask; curr_mask > 0; curr_mask >>= 1) {
      bool ascending = ((threadIdx.x & curr_mask) == 0) == ((threadIdx.x & next_mask) == 0);
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
          float opp_key    = smem_keys[(threadIdx.x ^ curr_mask) + (blockDim_x * i)];
          uint32_t opp_val = smem_vals[(threadIdx.x ^ curr_mask) + (blockDim_x * i)];
          swap_if_needed<float, uint32_t>(my_keys[i], opp_key, my_vals[i], opp_val, ascending);
        }
      } else {
// intra warp
#pragma unroll
        for (int i = 0; i < numElementsPerThread; i++) {
          float opp_key    = __shfl_xor_sync(0xffffffff, my_keys[i], curr_mask);
          uint32_t opp_val = __shfl_xor_sync(0xffffffff, my_vals[i], curr_mask);
          swap_if_needed<float, uint32_t>(my_keys[i], opp_key, my_vals[i], opp_val, ascending);
        }
      }
    }

    bool ascending = ((threadIdx.x & next_mask) == 0);
#pragma unroll
    for (uint32_t curr_mask = numElementsPerThread / 2; curr_mask > 0; curr_mask >>= 1) {
#pragma unroll
      for (int i = 0; i < numElementsPerThread; i++) {
        int j = i ^ curr_mask;
        if (i > j) continue;
        swap_if_needed<float, uint32_t>(my_keys[i], my_keys[j], my_vals[i], my_vals[j], ascending);
      }
    }
    mask = next_mask;
  }

  // Update knn_graph
  for (int i = 0; i < numElementsPerThread; i++) {
    int k = i + (numElementsPerThread * threadIdx.x);
    if (k < graph_degree) {
      knn_graph[srcNode_dev][k + ((uint64_t)graph_degree * srcNode_loc)] = my_vals[i];
    }
  }
}

template <int MAX_DEGREE>
__global__ void kern_prune(
  uint32_t** knn_graph,  // [num_gpus][graph_chunk_size, graph_degree]
  uint32_t graph_size,
  uint32_t graph_chunk_size,  // (*) num_gpus * graph_chunk_size >= graph_size
  uint32_t graph_degree,
  uint32_t degree,
  int dev_id,
  uint32_t batch_size,
  uint32_t batch_id,
  uint8_t** detour_count,          // [num_gpus][graph_chunk_size, graph_degree]
  uint32_t** num_no_detour_edges,  // [num_gpus][graph_size]
  uint64_t* stats)
{
  __shared__ uint32_t smem_num_detour[MAX_DEGREE];
  uint64_t* num_retain = stats;
  uint64_t* num_full   = stats + 1;

  uint64_t nid = blockIdx.x + (batch_size * batch_id);
  if (nid >= graph_chunk_size) { return; }
  for (uint32_t k = threadIdx.x; k < graph_degree; k += blockDim.x) {
    smem_num_detour[k] = 0;
  }
  __syncthreads();

  uint64_t iA     = nid + ((uint64_t)graph_chunk_size * dev_id);
  uint64_t iA_dev = iA / graph_chunk_size;
  uint64_t iA_loc = iA % graph_chunk_size;
  if (iA >= graph_size) { return; }

  // count number of detours (A->D->B)
  for (uint32_t kAD = 0; kAD < graph_degree - 1; kAD++) {
    uint64_t iD     = knn_graph[iA_dev][kAD + (graph_degree * iA_loc)];
    uint64_t iD_dev = iD / graph_chunk_size;
    uint64_t iD_loc = iD % graph_chunk_size;
    for (uint32_t kDB = threadIdx.x; kDB < graph_degree; kDB += blockDim.x) {
      uint64_t iB_candidate = knn_graph[iD_dev][kDB + ((uint64_t)graph_degree * iD_loc)];
      for (uint32_t kAB = kAD + 1; kAB < graph_degree; kAB++) {
        // if ( kDB < kAB )
        {
          uint64_t iB = knn_graph[iA_dev][kAB + (graph_degree * iA_loc)];
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
    detour_count[iA_dev][k + (graph_degree * iA_loc)] = min(smem_num_detour[k], (uint32_t)255);
    if (smem_num_detour[k] == 0) { num_edges_no_detour++; }
  }
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 1);
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 2);
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 4);
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 8);
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 16);
  num_edges_no_detour = min(num_edges_no_detour, degree);

  if (threadIdx.x == 0) {
    num_no_detour_edges[iA_dev][iA_loc] = num_edges_no_detour;
    atomicAdd((unsigned long long int*)num_retain, (unsigned long long int)num_edges_no_detour);
    if (num_edges_no_detour >= degree) { atomicAdd((unsigned long long int*)num_full, 1); }
  }
}

// unnamed namespace to avoid multiple definition error
namespace {
__global__ void kern_make_rev_graph(const uint32_t i_gpu,
                                    const uint32_t* dest_nodes,  // [global_graph_size]
                                    const uint32_t global_graph_size,
                                    uint32_t* rev_graph,        // [graph_size, degree]
                                    uint32_t* rev_graph_count,  // [graph_size]
                                    const uint32_t graph_size,
                                    const uint32_t degree)
{
  const uint32_t tid  = threadIdx.x + (blockDim.x * blockIdx.x);
  const uint32_t tnum = blockDim.x * gridDim.x;

  for (uint32_t gl_src_id = tid; gl_src_id < global_graph_size; gl_src_id += tnum) {
    uint32_t gl_dest_id = dest_nodes[gl_src_id];
    if (gl_dest_id < graph_size * i_gpu) continue;
    if (gl_dest_id >= graph_size * (i_gpu + 1)) continue;
    if (gl_dest_id >= global_graph_size) continue;

    uint32_t dest_id = gl_dest_id - (graph_size * i_gpu);
    uint32_t pos     = atomicAdd(rev_graph_count + dest_id, 1);
    if (pos < degree) { rev_graph[pos + ((uint64_t)degree * dest_id)] = gl_src_id; }
  }
}
}  // namespace
template <class T>
T*** mgpu_alloc(int n_gpus, uint32_t chunk, uint32_t nelems)
{
  T** arrays;                                      // [n_gpus][chunk, nelems]
  arrays       = (T**)malloc(sizeof(T*) * n_gpus); /* h1 */
  size_t bsize = sizeof(T) * chunk * nelems;
  // RAFT_LOG_DEBUG("[%s, %s, %d] n_gpus: %d, chunk: %u, nelems: %u, bsize: %lu (%lu MiB)\n",
  //         __FILE__, __func__, __LINE__, n_gpus, chunk, nelems, bsize, bsize / 1024 / 1024);
  for (int i_gpu = 0; i_gpu < n_gpus; i_gpu++) {
    RAFT_CUDA_TRY(cudaSetDevice(i_gpu));
    RAFT_CUDA_TRY(cudaMalloc(&(arrays[i_gpu]), bsize)); /* d1 */
  }
  T*** d_arrays;                                       // [n_gpus+1][n_gpus][chunk, nelems]
  d_arrays = (T***)malloc(sizeof(T**) * (n_gpus + 1)); /* h2 */
  bsize    = sizeof(T*) * n_gpus;
  for (int i_gpu = 0; i_gpu < n_gpus; i_gpu++) {
    RAFT_CUDA_TRY(cudaSetDevice(i_gpu));
    RAFT_CUDA_TRY(cudaMalloc(&(d_arrays[i_gpu]), bsize)); /* d2 */
    RAFT_CUDA_TRY(cudaMemcpy(d_arrays[i_gpu], arrays, bsize, cudaMemcpyDefault));
  }
  RAFT_CUDA_TRY(cudaSetDevice(0));
  d_arrays[n_gpus] = arrays;
  return d_arrays;
}

template <class T>
void mgpu_free(T*** d_arrays, int n_gpus)
{
  for (int i_gpu = 0; i_gpu < n_gpus; i_gpu++) {
    RAFT_CUDA_TRY(cudaSetDevice(i_gpu));
    RAFT_CUDA_TRY(cudaFree(d_arrays[n_gpus][i_gpu])); /* d1 */
    RAFT_CUDA_TRY(cudaFree(d_arrays[i_gpu]));         /* d2 */
  }
  RAFT_CUDA_TRY(cudaSetDevice(0));
  free(d_arrays[n_gpus]); /* h1 */
  free(d_arrays);         /* h2 */
}

template <class T>
void mgpu_H2D(T*** d_arrays,     // [n_gpus+1][n_gpus][chunk, nelems]
              const T* h_array,  // [size, nelems]
              int n_gpus,
              uint32_t size,
              uint32_t chunk,  // (*) n_gpus * chunk >= size
              uint32_t nelems)
{
#pragma omp parallel num_threads(n_gpus)
  {
    int i_gpu = omp_get_thread_num();
    RAFT_CUDA_TRY(cudaSetDevice(i_gpu));
    uint32_t _chunk = std::min(size - (chunk * i_gpu), chunk);
    size_t bsize    = sizeof(T) * _chunk * nelems;
    RAFT_CUDA_TRY(cudaMemcpy(d_arrays[n_gpus][i_gpu],
                             h_array + ((uint64_t)chunk * nelems * i_gpu),
                             bsize,
                             cudaMemcpyDefault));
  }
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  RAFT_CUDA_TRY(cudaSetDevice(0));
}

template <class T>
void mgpu_D2H(T*** d_arrays,  // [n_gpus+1][n_gpus][chunk, nelems]
              T* h_array,     // [size, nelems]
              int n_gpus,
              uint32_t size,
              uint32_t chunk,  // (*) n_gpus * chunk >= size
              uint32_t nelems)
{
#pragma omp parallel num_threads(n_gpus)
  {
    int i_gpu = omp_get_thread_num();
    RAFT_CUDA_TRY(cudaSetDevice(i_gpu));
    uint32_t _chunk = std::min(size - (chunk * i_gpu), chunk);
    size_t bsize    = sizeof(T) * _chunk * nelems;
    RAFT_CUDA_TRY(cudaMemcpy(h_array + ((uint64_t)chunk * nelems * i_gpu),
                             d_arrays[n_gpus][i_gpu],
                             bsize,
                             cudaMemcpyDefault));
  }
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  RAFT_CUDA_TRY(cudaSetDevice(0));
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

/** Input arrays can be both host and device*/
template <class DATA_T,
          typename IdxT = uint32_t,
          typename d_accessor =
            host_device_accessor<std::experimental::default_accessor<DATA_T>, memory_type::device>,
          typename g_accessor =
            host_device_accessor<std::experimental::default_accessor<DATA_T>, memory_type::host>>
void prune(raft::device_resources const& res,
           mdspan<const DATA_T, matrix_extent<IdxT>, row_major, d_accessor> dataset,
           mdspan<IdxT, matrix_extent<IdxT>, row_major, g_accessor> knn_graph,
           raft::host_matrix_view<IdxT, IdxT, row_major> new_graph)
{
  RAFT_LOG_DEBUG(
    "# Pruning kNN graph (size=%lu, degree=%lu)\n", knn_graph.extent(0), knn_graph.extent(1));

  RAFT_EXPECTS(
    dataset.extent(0) == knn_graph.extent(0) && knn_graph.extent(0) == new_graph.extent(0),
    "Each input array is expected to have the same number of rows");
  RAFT_EXPECTS(new_graph.extent(1) <= knn_graph.extent(1),
               "output graph cannot have more columns than input graph");
  const uint32_t dataset_size        = dataset.extent(0);
  const uint32_t dataset_dim         = dataset.extent(1);
  const uint32_t input_graph_degree  = knn_graph.extent(1);
  const uint32_t output_graph_degree = new_graph.extent(1);
  const DATA_T* dataset_ptr          = dataset.data_handle();
  uint32_t* input_graph_ptr          = (uint32_t*)knn_graph.data_handle();
  uint32_t* output_graph_ptr         = new_graph.data_handle();
  float scale                  = 1.0f / raft::spatial::knn::detail::utils::config<DATA_T>::kDivisor;
  const std::size_t graph_size = dataset_size;
  size_t array_size;

  // Setup GPUs
  int num_gpus = 0;

  // Setup GPUs
  RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus));
  RAFT_LOG_DEBUG("# num_gpus: %d\n", num_gpus);
  for (int self = 0; self < num_gpus; self++) {
    RAFT_CUDA_TRY(cudaSetDevice(self));
    for (int peer = 0; peer < num_gpus; peer++) {
      if (self == peer) { continue; }
      RAFT_CUDA_TRY(cudaDeviceEnablePeerAccess(peer, 0));
    }
  }
  RAFT_CUDA_TRY(cudaSetDevice(0));

  uint32_t graph_chunk_size     = graph_size;
  uint32_t*** d_input_graph_ptr = NULL;  // [...][num_gpus][graph_chunk_size, input_graph_degree]
  graph_chunk_size              = (graph_size + num_gpus - 1) / num_gpus;
  d_input_graph_ptr = mgpu_alloc<uint32_t>(num_gpus, graph_chunk_size, input_graph_degree);

  uint32_t dataset_chunk_size = dataset_size;
  DATA_T*** d_dataset_ptr     = NULL;  // [num_gpus+1][...][...]
  dataset_chunk_size          = (dataset_size + num_gpus - 1) / num_gpus;
  assert(dataset_chunk_size == graph_chunk_size);
  d_dataset_ptr = mgpu_alloc<DATA_T>(num_gpus, dataset_chunk_size, dataset_dim);

  mgpu_H2D<DATA_T>(
    d_dataset_ptr, dataset_ptr, num_gpus, dataset_size, dataset_chunk_size, dataset_dim);

  //
  // Sorting kNN graph
  //
  double time_sort_start = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN Graph on GPUs ");
  mgpu_H2D<uint32_t>(
    d_input_graph_ptr, input_graph_ptr, num_gpus, graph_size, graph_chunk_size, input_graph_degree);
  void (*kernel_sort)(
    DATA_T**, uint32_t, uint32_t, uint32_t, float, uint32_t**, uint32_t, uint32_t, uint32_t, int);
  constexpr int numElementsPerThread = 4;
  dim3 threads_sort(1, 1, 1);
  if (input_graph_degree <= numElementsPerThread * 32) {
    constexpr int blockDim_x = 32;
    kernel_sort              = kern_sort<DATA_T, blockDim_x, numElementsPerThread>;
    threads_sort.x           = blockDim_x;
  } else if (input_graph_degree <= numElementsPerThread * 64) {
    constexpr int blockDim_x = 64;
    kernel_sort              = kern_sort<DATA_T, blockDim_x, numElementsPerThread>;
    threads_sort.x           = blockDim_x;
  } else if (input_graph_degree <= numElementsPerThread * 128) {
    constexpr int blockDim_x = 128;
    kernel_sort              = kern_sort<DATA_T, blockDim_x, numElementsPerThread>;
    threads_sort.x           = blockDim_x;
  } else if (input_graph_degree <= numElementsPerThread * 256) {
    constexpr int blockDim_x = 256;
    kernel_sort              = kern_sort<DATA_T, blockDim_x, numElementsPerThread>;
    threads_sort.x           = blockDim_x;
  } else {
    fprintf(stderr,
            "[ERROR] The degree of input knn graph is too large (%u). "
            "It must be equal to or small than %d.\n",
            input_graph_degree,
            numElementsPerThread * 256);
    exit(-1);
  }
  dim3 blocks_sort(graph_chunk_size, 1, 1);
  for (int i_gpu = 0; i_gpu < num_gpus; i_gpu++) {
    RAFT_LOG_DEBUG(".");
    RAFT_CUDA_TRY(cudaSetDevice(i_gpu));
    kernel_sort<<<blocks_sort, threads_sort>>>(d_dataset_ptr[i_gpu],
                                               dataset_size,
                                               dataset_chunk_size,
                                               dataset_dim,
                                               scale,
                                               d_input_graph_ptr[i_gpu],
                                               graph_size,
                                               graph_chunk_size,
                                               input_graph_degree,
                                               i_gpu);
  }
  RAFT_CUDA_TRY(cudaSetDevice(0));
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  RAFT_LOG_DEBUG(".");
  mgpu_D2H<uint32_t>(
    d_input_graph_ptr, input_graph_ptr, num_gpus, graph_size, graph_chunk_size, input_graph_degree);
  RAFT_LOG_DEBUG("\n");
  double time_sort_end = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN graph time: %.1lf sec\n", time_sort_end - time_sort_start);

  mgpu_free<DATA_T>(d_dataset_ptr, num_gpus);

  //
  uint8_t* detour_count;  // [graph_size, input_graph_degree]
  array_size   = sizeof(uint8_t) * graph_size * input_graph_degree;
  detour_count = (uint8_t*)malloc(array_size);
  memset(detour_count, 0xff, array_size);

  uint8_t*** d_detour_count = NULL;  // [...][num_gpus][graph_chunk_size, input_graph_degree]
  d_detour_count            = mgpu_alloc<uint8_t>(num_gpus, graph_chunk_size, input_graph_degree);
  mgpu_H2D<uint8_t>(
    d_detour_count, detour_count, num_gpus, graph_size, graph_chunk_size, input_graph_degree);

  //
  uint32_t* num_no_detour_edges;  // [graph_size]
  array_size          = sizeof(uint32_t) * graph_size;
  num_no_detour_edges = (uint32_t*)malloc(array_size);
  memset(num_no_detour_edges, 0, array_size);

  uint32_t*** d_num_no_detour_edges = NULL;  // [...][num_gpus][graph_chunk_size]
  d_num_no_detour_edges             = mgpu_alloc<uint32_t>(num_gpus, graph_chunk_size, 1);
  mgpu_H2D<uint32_t>(
    d_num_no_detour_edges, num_no_detour_edges, num_gpus, graph_size, graph_chunk_size, 1);

  //
  uint64_t** dev_stats  = NULL;  // [num_gpus][2]
  uint64_t** host_stats = NULL;  // [num_gpus][2]
  dev_stats             = (uint64_t**)malloc(sizeof(uint64_t*) * num_gpus);
  host_stats            = (uint64_t**)malloc(sizeof(uint64_t*) * num_gpus);
  array_size            = sizeof(uint64_t) * 2;
  for (int i_gpu = 0; i_gpu < num_gpus; i_gpu++) {
    RAFT_CUDA_TRY(cudaSetDevice(i_gpu));
    RAFT_CUDA_TRY(cudaMalloc(&(dev_stats[i_gpu]), array_size));
    host_stats[i_gpu] = (uint64_t*)malloc(array_size);
  }
  RAFT_CUDA_TRY(cudaSetDevice(0));

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
  double time_prune_start = cur_time();
  uint64_t num_keep       = 0;
  uint64_t num_full       = 0;
  RAFT_LOG_DEBUG("# Pruning kNN Graph on GPUs\r");
  mgpu_H2D<uint32_t>(
    d_input_graph_ptr, input_graph_ptr, num_gpus, graph_size, graph_chunk_size, input_graph_degree);
  void (*kernel_prune)(uint32_t**,
                       uint32_t,
                       uint32_t,
                       uint32_t,
                       uint32_t,
                       int,
                       uint32_t,
                       uint32_t,
                       uint8_t**,
                       uint32_t**,
                       uint64_t*);
  if (input_graph_degree <= 1024) {
    constexpr int MAX_DEGREE = 1024;
    kernel_prune             = kern_prune<MAX_DEGREE>;
  } else {
    fprintf(stderr,
            "[ERROR] The degree of input knn graph is too large (%u). "
            "It must be equal to or small than %d.\n",
            input_graph_degree,
            1024);
    exit(-1);
  }
  uint32_t batch_size = std::min(graph_chunk_size, (uint32_t)256 * 1024);
  uint32_t num_batch  = (graph_chunk_size + batch_size - 1) / batch_size;
  dim3 threads_prune(32, 1, 1);
  dim3 blocks_prune(batch_size, 1, 1);
  for (int i_gpu = 0; i_gpu < num_gpus; i_gpu++) {
    RAFT_CUDA_TRY(cudaSetDevice(i_gpu));
    RAFT_CUDA_TRY(cudaMemset(dev_stats[i_gpu], 0, sizeof(uint64_t) * 2));
  }
  for (uint32_t i_batch = 0; i_batch < num_batch; i_batch++) {
    for (int i_gpu = 0; i_gpu < num_gpus; i_gpu++) {
      RAFT_CUDA_TRY(cudaSetDevice(i_gpu));
      kernel_prune<<<blocks_prune, threads_prune>>>(d_input_graph_ptr[i_gpu],
                                                    graph_size,
                                                    graph_chunk_size,
                                                    input_graph_degree,
                                                    output_graph_degree,
                                                    i_gpu,
                                                    batch_size,
                                                    i_batch,
                                                    d_detour_count[i_gpu],
                                                    d_num_no_detour_edges[i_gpu],
                                                    dev_stats[i_gpu]);
    }
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    fprintf(
      stderr,
      "# Pruning kNN Graph on GPUs (%.1lf %%)\r",
      (double)std::min((i_batch + 1) * batch_size, graph_chunk_size) / graph_chunk_size * 100);
  }
  for (int i_gpu = 0; i_gpu < num_gpus; i_gpu++) {
    RAFT_CUDA_TRY(cudaSetDevice(i_gpu));
    RAFT_CUDA_TRY(
      cudaMemcpy(host_stats[i_gpu], dev_stats[i_gpu], sizeof(uint64_t) * 2, cudaMemcpyDefault));
    num_keep += host_stats[i_gpu][0];
    num_full += host_stats[i_gpu][1];
  }
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  RAFT_CUDA_TRY(cudaSetDevice(0));
  RAFT_LOG_DEBUG("\n");

  mgpu_D2H<uint8_t>(
    d_detour_count, detour_count, num_gpus, graph_size, graph_chunk_size, input_graph_degree);
  mgpu_D2H<uint32_t>(
    d_num_no_detour_edges, num_no_detour_edges, num_gpus, graph_size, graph_chunk_size, 1);

  mgpu_free<uint32_t>(d_input_graph_ptr, num_gpus);
  mgpu_free<uint8_t>(d_detour_count, num_gpus);
  mgpu_free<uint32_t>(d_num_no_detour_edges, num_gpus);

  // Create pruned kNN graph
  array_size                 = sizeof(uint32_t) * graph_size * output_graph_degree;
  uint32_t* pruned_graph_ptr = (uint32_t*)malloc(array_size);
  uint32_t max_detour        = 0;
#pragma omp parallel for reduction(max : max_detour)
  for (uint64_t i = 0; i < graph_size; i++) {
    uint64_t pk = 0;
    for (uint32_t num_detour = 0; num_detour < output_graph_degree; num_detour++) {
      if (max_detour < num_detour) { max_detour = num_detour; /* stats */ }
      for (uint64_t k = 0; k < input_graph_degree; k++) {
        if (detour_count[k + (input_graph_degree * i)] != num_detour) { continue; }
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

  double time_prune_end = cur_time();
  fprintf(stderr,
          "# Pruning time: %.1lf sec, "
          "avg_no_detour_edges_per_node: %.2lf/%u, "
          "nodes_with_no_detour_at_all_edges: %.1lf%%\n",
          time_prune_end - time_prune_start,
          (double)num_keep / graph_size,
          output_graph_degree,
          (double)num_full / graph_size * 100);

  //
  // Make reverse graph
  //
  double time_make_start = cur_time();

  array_size              = sizeof(uint32_t) * graph_size * output_graph_degree;
  uint32_t* rev_graph_ptr = (uint32_t*)malloc(array_size);
  memset(rev_graph_ptr, 0xff, array_size);

  uint32_t*** d_rev_graph_ptr;  // [...][num_gpus][graph_chunk_size, output_graph_degree]
  d_rev_graph_ptr = mgpu_alloc<uint32_t>(num_gpus, graph_chunk_size, output_graph_degree);
  mgpu_H2D<uint32_t>(
    d_rev_graph_ptr, rev_graph_ptr, num_gpus, graph_size, graph_chunk_size, output_graph_degree);

  array_size                = sizeof(uint32_t) * graph_size;
  uint32_t* rev_graph_count = (uint32_t*)malloc(array_size);
  memset(rev_graph_count, 0, array_size);

  uint32_t*** d_rev_graph_count;  // [...][num_gpus][graph_chunk_size, 1]
  d_rev_graph_count = mgpu_alloc<uint32_t>(num_gpus, graph_chunk_size, 1);
  mgpu_H2D<uint32_t>(d_rev_graph_count, rev_graph_count, num_gpus, graph_size, graph_chunk_size, 1);

  uint32_t* dest_nodes;  // [graph_size]
  dest_nodes = (uint32_t*)malloc(sizeof(uint32_t) * graph_size);
  uint32_t** d_dest_nodes;  // [num_gpus][graph_size]
  d_dest_nodes = (uint32_t**)malloc(sizeof(uint32_t*) * num_gpus);
  for (int i_gpu = 0; i_gpu < num_gpus; i_gpu++) {
    RAFT_CUDA_TRY(cudaSetDevice(i_gpu));
    RAFT_CUDA_TRY(cudaMalloc(&(d_dest_nodes[i_gpu]), sizeof(uint32_t) * graph_size));
  }

  for (uint64_t k = 0; k < output_graph_degree; k++) {
#pragma omp parallel for
    for (uint64_t i = 0; i < graph_size; i++) {
      dest_nodes[i] = pruned_graph_ptr[k + (output_graph_degree * i)];
    }
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
#pragma omp parallel num_threads(num_gpus)
    {
      int i_gpu = omp_get_thread_num();
      RAFT_CUDA_TRY(cudaSetDevice(i_gpu));
      RAFT_CUDA_TRY(cudaMemcpy(
        d_dest_nodes[i_gpu], dest_nodes, sizeof(uint32_t) * graph_size, cudaMemcpyHostToDevice));
      dim3 threads(256, 1, 1);
      dim3 blocks(1024, 1, 1);
      kern_make_rev_graph<<<blocks, threads>>>(i_gpu,
                                               d_dest_nodes[i_gpu],
                                               graph_size,
                                               d_rev_graph_ptr[num_gpus][i_gpu],
                                               d_rev_graph_count[num_gpus][i_gpu],
                                               graph_chunk_size,
                                               output_graph_degree);
    }
    RAFT_LOG_DEBUG("# Making reverse graph on GPUs: %lu / %u    \r", k, output_graph_degree);
  }
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  RAFT_CUDA_TRY(cudaSetDevice(0));
  RAFT_LOG_DEBUG("\n");

  mgpu_D2H<uint32_t>(
    d_rev_graph_ptr, rev_graph_ptr, num_gpus, graph_size, graph_chunk_size, output_graph_degree);
  mgpu_D2H<uint32_t>(d_rev_graph_count, rev_graph_count, num_gpus, graph_size, graph_chunk_size, 1);
  mgpu_free<uint32_t>(d_rev_graph_ptr, num_gpus);
  mgpu_free<uint32_t>(d_rev_graph_count, num_gpus);

  double time_make_end = cur_time();
  RAFT_LOG_DEBUG("# Making reverse graph time: %.1lf sec", time_make_end - time_make_start);

  //
  // Replace some edges with reverse edges
  //
  double time_replace_start = cur_time();

  uint64_t num_protected_edges = output_graph_degree / 2;
  RAFT_LOG_DEBUG("# num_protected_edges: %lu", num_protected_edges);

  array_size = sizeof(uint32_t) * graph_size * output_graph_degree;
  memcpy(output_graph_ptr, pruned_graph_ptr, array_size);

  constexpr int _omp_chunk = 1024;
#pragma omp parallel for schedule(dynamic, _omp_chunk)
  for (uint64_t j = 0; j < graph_size; j++) {
    for (uint64_t _k = 0; _k < rev_graph_count[j]; _k++) {
      uint64_t k = rev_graph_count[j] - 1 - _k;
      uint64_t i = rev_graph_ptr[k + (output_graph_degree * j)];

      uint64_t pos = pos_in_array<uint32_t>(
        i, output_graph_ptr + (output_graph_degree * j), output_graph_degree);
      if (pos < num_protected_edges) { continue; }
      uint64_t num_shift = pos - num_protected_edges;
      if (pos == output_graph_degree) { num_shift = output_graph_degree - num_protected_edges - 1; }
      shift_array<uint32_t>(output_graph_ptr + num_protected_edges + (output_graph_degree * j),
                            num_shift);
      output_graph_ptr[num_protected_edges + (output_graph_degree * j)] = i;
    }
    if ((omp_get_thread_num() == 0) && ((j % _omp_chunk) == 0)) {
      RAFT_LOG_DEBUG("# Replacing reverse edges: %lu / %lu    ", j, graph_size);
    }
  }
  RAFT_LOG_DEBUG("\n");
  free(rev_graph_ptr);
  free(rev_graph_count);

  double time_replace_end = cur_time();
  RAFT_LOG_DEBUG("# Replacing edges time: %.1lf sec", time_replace_end - time_replace_start);

  /* stats */
  uint64_t num_replaced_edges = 0;
#pragma omp parallel for reduction(+ : num_replaced_edges)
  for (uint64_t i = 0; i < graph_size; i++) {
    for (uint64_t k = 0; k < output_graph_degree; k++) {
      uint64_t j   = pruned_graph_ptr[k + (output_graph_degree * i)];
      uint64_t pos = pos_in_array<uint32_t>(
        j, output_graph_ptr + (output_graph_degree * i), output_graph_degree);
      if (pos == output_graph_degree) { num_replaced_edges += 1; }
    }
  }
  fprintf(stderr,
          "# Average number of replaced edges per node: %.2f",
          (double)num_replaced_edges / graph_size);
}

}  // namespace graph
}  // namespace raft::neighbors::experimental::cagra::detail
