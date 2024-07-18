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

#include "../nn_descent_types.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/slice.cuh>
#include <raft/neighbors/detail/cagra/device_common.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <raft/util/arch.cuh>  // raft::util::arch::SM_*
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>

#include <mma.h>
#include <omp.h>

#include <limits>
#include <optional>
#include <queue>
#include <random>

namespace raft::neighbors::experimental::nn_descent::detail {

using pinned_memory_resource = thrust::universal_host_pinned_memory_resource;
template <typename T>
using pinned_memory_allocator = thrust::mr::stateless_resource_allocator<T, pinned_memory_resource>;

using DistData_t = float;
constexpr int DEGREE_ON_DEVICE{32};
constexpr int SEGMENT_SIZE{32};
constexpr int counter_interval{100};
template <typename Index_t>
struct InternalID_t;

// InternalID_t uses 1 bit for marking (new or old).
template <>
class InternalID_t<int> {
 private:
  using Index_t = int;
  Index_t id_{std::numeric_limits<Index_t>::max()};

 public:
  __host__ __device__ bool is_new() const { return id_ >= 0; }
  __host__ __device__ Index_t& id_with_flag() { return id_; }
  __host__ __device__ Index_t id() const
  {
    if (is_new()) return id_;
    return -id_ - 1;
  }
  __host__ __device__ void mark_old()
  {
    if (id_ >= 0) id_ = -id_ - 1;
  }
  __host__ __device__ bool operator==(const InternalID_t<int>& other) const
  {
    return id() == other.id();
  }
};

template <typename Index_t>
struct ResultItem;

template <>
class ResultItem<int> {
 private:
  using Index_t = int;
  Index_t id_;
  DistData_t dist_;

 public:
  __host__ __device__ ResultItem()
    : id_(std::numeric_limits<Index_t>::max()), dist_(std::numeric_limits<DistData_t>::max()){};
  __host__ __device__ ResultItem(const Index_t id_with_flag, const DistData_t dist)
    : id_(id_with_flag), dist_(dist){};
  __host__ __device__ bool is_new() const { return id_ >= 0; }
  __host__ __device__ Index_t& id_with_flag() { return id_; }
  __host__ __device__ Index_t id() const
  {
    if (is_new()) return id_;
    return -id_ - 1;
  }
  __host__ __device__ DistData_t& dist() { return dist_; }

  __host__ __device__ void mark_old()
  {
    if (id_ >= 0) id_ = -id_ - 1;
  }

  __host__ __device__ bool operator<(const ResultItem<Index_t>& other) const
  {
    if (dist_ == other.dist_) return id() < other.id();
    return dist_ < other.dist_;
  }
  __host__ __device__ bool operator==(const ResultItem<Index_t>& other) const
  {
    return id() == other.id();
  }
  __host__ __device__ bool operator>=(const ResultItem<Index_t>& other) const
  {
    return !(*this < other);
  }
  __host__ __device__ bool operator<=(const ResultItem<Index_t>& other) const
  {
    return (*this == other) || (*this < other);
  }
  __host__ __device__ bool operator>(const ResultItem<Index_t>& other) const
  {
    return !(*this <= other);
  }
  __host__ __device__ bool operator!=(const ResultItem<Index_t>& other) const
  {
    return !(*this == other);
  }
};

using align32 = raft::Pow2<32>;

template <typename T>
int get_batch_size(const int it_now, const T nrow, const int batch_size)
{
  int it_total = ceildiv(nrow, batch_size);
  return (it_now == it_total - 1) ? nrow - it_now * batch_size : batch_size;
}

// for avoiding bank conflict
template <typename T>
constexpr __host__ __device__ __forceinline__ int skew_dim(int ndim)
{
  // all "4"s are for alignment
  if constexpr (std::is_same<T, float>::value) {
    ndim = ceildiv(ndim, 4) * 4;
    return ndim + (ndim % 32 == 0) * 4;
  }
}

template <typename T>
__device__ __forceinline__ ResultItem<T> xor_swap(ResultItem<T> x, int mask, int dir)
{
  ResultItem<T> y;
  y.dist() = __shfl_xor_sync(raft::warp_full_mask(), x.dist(), mask, raft::warp_size());
  y.id_with_flag() =
    __shfl_xor_sync(raft::warp_full_mask(), x.id_with_flag(), mask, raft::warp_size());
  return x < y == dir ? y : x;
}

__device__ __forceinline__ int xor_swap(int x, int mask, int dir)
{
  int y = __shfl_xor_sync(raft::warp_full_mask(), x, mask, raft::warp_size());
  return x < y == dir ? y : x;
}

// TODO: Move to RAFT utils https://github.com/rapidsai/raft/issues/1827
__device__ __forceinline__ uint bfe(uint lane_id, uint pos)
{
  uint res;
  asm("bfe.u32 %0,%1,%2,%3;" : "=r"(res) : "r"(lane_id), "r"(pos), "r"(1));
  return res;
}

template <typename T>
__device__ __forceinline__ void warp_bitonic_sort(T* element_ptr, const int lane_id)
{
  static_assert(raft::warp_size() == 32);
  auto& element = *element_ptr;
  element       = xor_swap(element, 0x01, bfe(lane_id, 1) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x02, bfe(lane_id, 2) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 2) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x04, bfe(lane_id, 3) ^ bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 3) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 3) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x08, bfe(lane_id, 4) ^ bfe(lane_id, 3));
  element       = xor_swap(element, 0x04, bfe(lane_id, 4) ^ bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 4) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 4) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x10, bfe(lane_id, 4));
  element       = xor_swap(element, 0x08, bfe(lane_id, 3));
  element       = xor_swap(element, 0x04, bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 0));
  return;
}

struct BuildConfig {
  size_t max_dataset_size;
  size_t dataset_dim;
  size_t node_degree{64};
  size_t internal_node_degree{0};
  // If internal_node_degree == 0, the value of node_degree will be assigned to it
  size_t max_iterations{50};
  float termination_threshold{0.0001};
  size_t output_graph_degree{32};
};

template <typename Index_t>
class BloomFilter {
 public:
  BloomFilter(size_t nrow, size_t num_sets_per_list, size_t num_hashs)
    : nrow_(nrow),
      num_sets_per_list_(num_sets_per_list),
      num_hashs_(num_hashs),
      bitsets_(nrow * num_bits_per_set_ * num_sets_per_list)
  {
  }

  void add(size_t list_id, Index_t key)
  {
    if (is_cleared) { is_cleared = false; }
    uint32_t hash         = hash_0(key);
    size_t global_set_idx = list_id * num_bits_per_set_ * num_sets_per_list_ +
                            key % num_sets_per_list_ * num_bits_per_set_;
    bitsets_[global_set_idx + hash % num_bits_per_set_] = 1;
    for (size_t i = 1; i < num_hashs_; i++) {
      hash                                                = hash + hash_1(key);
      bitsets_[global_set_idx + hash % num_bits_per_set_] = 1;
    }
  }

  bool check(size_t list_id, Index_t key)
  {
    bool is_present       = true;
    uint32_t hash         = hash_0(key);
    size_t global_set_idx = list_id * num_bits_per_set_ * num_sets_per_list_ +
                            key % num_sets_per_list_ * num_bits_per_set_;
    is_present &= bitsets_[global_set_idx + hash % num_bits_per_set_];

    if (!is_present) return false;
    for (size_t i = 1; i < num_hashs_; i++) {
      hash = hash + hash_1(key);
      is_present &= bitsets_[global_set_idx + hash % num_bits_per_set_];
      if (!is_present) return false;
    }
    return true;
  }

  void clear()
  {
    if (is_cleared) return;
#pragma omp parallel for
    for (size_t i = 0; i < nrow_ * num_bits_per_set_ * num_sets_per_list_; i++) {
      bitsets_[i] = 0;
    }
    is_cleared = true;
  }

 private:
  uint32_t hash_0(uint32_t value)
  {
    value *= 1103515245;
    value += 12345;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
  }

  uint32_t hash_1(uint32_t value)
  {
    value *= 1664525;
    value += 1013904223;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
  }

  static constexpr int num_bits_per_set_ = 512;
  bool is_cleared{true};
  std::vector<bool> bitsets_;
  size_t nrow_;
  size_t num_sets_per_list_;
  size_t num_hashs_;
};

template <typename Index_t>
struct GnndGraph {
  static constexpr int segment_size = 32;
  InternalID_t<Index_t>* h_graph;

  size_t nrow;
  size_t node_degree;
  int num_samples;
  int num_segments;

  raft::host_matrix<DistData_t, size_t, raft::row_major> h_dists;

  thrust::host_vector<Index_t, pinned_memory_allocator<Index_t>> h_graph_new;
  thrust::host_vector<int2, pinned_memory_allocator<int2>> h_list_sizes_new;

  thrust::host_vector<Index_t, pinned_memory_allocator<Index_t>> h_graph_old;
  thrust::host_vector<int2, pinned_memory_allocator<int2>> h_list_sizes_old;
  BloomFilter<Index_t> bloom_filter;

  GnndGraph(const GnndGraph&)            = delete;
  GnndGraph& operator=(const GnndGraph&) = delete;
  GnndGraph(const size_t nrow,
            const size_t node_degree,
            const size_t internal_node_degree,
            const size_t num_samples);
  void init_random_graph();
  // TODO: Create a generic bloom filter utility https://github.com/rapidsai/raft/issues/1827
  // Use Bloom filter to sample "new" neighbors for local joining
  void sample_graph_new(InternalID_t<Index_t>* new_neighbors, const size_t width);
  void sample_graph(bool sample_new);
  void update_graph(const InternalID_t<Index_t>* new_neighbors,
                    const DistData_t* new_dists,
                    const size_t width,
                    std::atomic<int64_t>& update_counter);
  void sort_lists();
  void clear();
  ~GnndGraph();
};

template <typename Data_t = float, typename Index_t = int, typename epilogue_op = raft::identity_op>
class GNND {
 public:
  GNND(raft::resources const& res, const BuildConfig& build_config);
  GNND(const GNND&)            = delete;
  GNND& operator=(const GNND&) = delete;

  void build(Data_t* data,
             const Index_t nrow,
             Index_t* output_graph,
             bool return_distances,
             DistData_t* output_distances,
             epilogue_op distance_epilogue = raft::identity_op());
  ~GNND()    = default;
  using ID_t = InternalID_t<Index_t>;

 private:
  void add_reverse_edges(Index_t* graph_ptr,
                         Index_t* h_rev_graph_ptr,
                         Index_t* d_rev_graph_ptr,
                         int2* list_sizes,
                         cudaStream_t stream = 0);
  void local_join(cudaStream_t stream = 0, epilogue_op distance_epilogue = raft::identity_op());

  raft::resources const& res;

  BuildConfig build_config_;
  GnndGraph<Index_t> graph_;
  std::atomic<int64_t> update_counter_;

  size_t nrow_;
  size_t ndim_;

  raft::device_matrix<__half, size_t, raft::row_major> d_data_;
  raft::device_vector<DistData_t, size_t> l2_norms_;

  raft::device_matrix<ID_t, size_t, raft::row_major> graph_buffer_;
  raft::device_matrix<DistData_t, size_t, raft::row_major> dists_buffer_;

  // TODO: Investigate using RMM/RAFT types https://github.com/rapidsai/raft/issues/1827
  thrust::host_vector<ID_t, pinned_memory_allocator<ID_t>> graph_host_buffer_;
  thrust::host_vector<DistData_t, pinned_memory_allocator<DistData_t>> dists_host_buffer_;

  raft::device_vector<int, size_t> d_locks_;

  thrust::host_vector<Index_t, pinned_memory_allocator<Index_t>> h_rev_graph_new_;
  thrust::host_vector<Index_t, pinned_memory_allocator<Index_t>> h_graph_old_;
  thrust::host_vector<Index_t, pinned_memory_allocator<Index_t>> h_rev_graph_old_;
  // int2.x is the number of forward edges, int2.y is the number of reverse edges

  raft::device_vector<int2, size_t> d_list_sizes_new_;
  raft::device_vector<int2, size_t> d_list_sizes_old_;
};

constexpr int TILE_ROW_WIDTH = 64;
constexpr int TILE_COL_WIDTH = 128;

constexpr int NUM_SAMPLES = 32;
// For now, the max. number of samples is 32, so the sample cache size is fixed
// to 64 (32 * 2).
constexpr int MAX_NUM_BI_SAMPLES        = 64;
constexpr int SKEWED_MAX_NUM_BI_SAMPLES = skew_dim<float>(MAX_NUM_BI_SAMPLES);
constexpr int BLOCK_SIZE                = 512;
constexpr int WMMA_M                    = 16;
constexpr int WMMA_N                    = 16;
constexpr int WMMA_K                    = 16;

template <typename Data_t>
__device__ __forceinline__ void load_vec(Data_t* vec_buffer,
                                         const Data_t* d_vec,
                                         const int load_dims,
                                         const int padding_dims,
                                         const int lane_id)
{
  if constexpr (std::is_same_v<Data_t, float> or std::is_same_v<Data_t, uint8_t> or
                std::is_same_v<Data_t, int8_t>) {
    constexpr int num_load_elems_per_warp = raft::warp_size();
    for (int step = 0; step < ceildiv(padding_dims, num_load_elems_per_warp); step++) {
      int idx = step * num_load_elems_per_warp + lane_id;
      if (idx < load_dims) {
        vec_buffer[idx] = d_vec[idx];
      } else if (idx < padding_dims) {
        vec_buffer[idx] = 0.0f;
      }
    }
  }
  if constexpr (std::is_same_v<Data_t, __half>) {
    if ((size_t)d_vec % sizeof(float2) == 0 && (size_t)vec_buffer % sizeof(float2) == 0 &&
        load_dims % 4 == 0 && padding_dims % 4 == 0) {
      constexpr int num_load_elems_per_warp = raft::warp_size() * 4;
#pragma unroll
      for (int step = 0; step < ceildiv(padding_dims, num_load_elems_per_warp); step++) {
        int idx_in_vec = step * num_load_elems_per_warp + lane_id * 4;
        if (idx_in_vec + 4 <= load_dims) {
          *(float2*)(vec_buffer + idx_in_vec) = *(float2*)(d_vec + idx_in_vec);
        } else if (idx_in_vec + 4 <= padding_dims) {
          *(float2*)(vec_buffer + idx_in_vec) = float2({0.0f, 0.0f});
        }
      }
    } else {
      constexpr int num_load_elems_per_warp = raft::warp_size();
      for (int step = 0; step < ceildiv(padding_dims, num_load_elems_per_warp); step++) {
        int idx = step * num_load_elems_per_warp + lane_id;
        if (idx < load_dims) {
          vec_buffer[idx] = d_vec[idx];
        } else if (idx < padding_dims) {
          vec_buffer[idx] = 0.0f;
        }
      }
    }
  }
}

// TODO: Replace with RAFT utilities https://github.com/rapidsai/raft/issues/1827
/** Calculate L2 norm, and cast data to __half */
template <typename Data_t>
RAFT_KERNEL preprocess_data_kernel(const Data_t* input_data,
                                   __half* output_data,
                                   int dim,
                                   DistData_t* l2_norms,
                                   size_t list_offset = 0)
{
  extern __shared__ char buffer[];
  __shared__ float l2_norm;
  Data_t* s_vec  = (Data_t*)buffer;
  size_t list_id = list_offset + blockIdx.x;

  load_vec(s_vec, input_data + blockIdx.x * dim, dim, dim, threadIdx.x % raft::warp_size());
  if (threadIdx.x == 0) { l2_norm = 0; }
  __syncthreads();
  int lane_id = threadIdx.x % raft::warp_size();
  for (int step = 0; step < ceildiv(dim, raft::warp_size()); step++) {
    int idx         = step * raft::warp_size() + lane_id;
    float part_dist = 0;
    if (idx < dim) {
      part_dist = s_vec[idx];
      part_dist = part_dist * part_dist;
    }
    __syncwarp();
    for (int offset = raft::warp_size() >> 1; offset >= 1; offset >>= 1) {
      part_dist += __shfl_down_sync(raft::warp_full_mask(), part_dist, offset);
    }
    if (lane_id == 0) { l2_norm += part_dist; }
    __syncwarp();
  }

  for (int step = 0; step < ceildiv(dim, raft::warp_size()); step++) {
    int idx = step * raft::warp_size() + threadIdx.x;
    if (idx < dim) {
      if (l2_norms == nullptr) {
        output_data[list_id * dim + idx] =
          (float)input_data[(size_t)blockIdx.x * dim + idx] / sqrt(l2_norm);
      } else {
        output_data[list_id * dim + idx] = input_data[(size_t)blockIdx.x * dim + idx];
        if (idx == 0) { l2_norms[list_id] = l2_norm; }
      }
    }
  }
}

template <typename Index_t>
RAFT_KERNEL add_rev_edges_kernel(const Index_t* graph,
                                 Index_t* rev_graph,
                                 int num_samples,
                                 int2* list_sizes)
{
  size_t list_id = blockIdx.x;
  int2 list_size = list_sizes[list_id];

  for (int idx = threadIdx.x; idx < list_size.x; idx += blockDim.x) {
    // each node has same number (num_samples) of forward and reverse edges
    size_t rev_list_id = graph[list_id * num_samples + idx];
    // there are already num_samples forward edges
    int idx_in_rev_list = atomicAdd(&list_sizes[rev_list_id].y, 1);
    if (idx_in_rev_list >= num_samples) {
      atomicExch(&list_sizes[rev_list_id].y, num_samples);
    } else {
      rev_graph[rev_list_id * num_samples + idx_in_rev_list] = list_id;
    }
  }
}

template <typename Index_t, typename ID_t = InternalID_t<Index_t>>
__device__ void insert_to_global_graph(ResultItem<Index_t> elem,
                                       size_t list_id,
                                       ID_t* graph,
                                       DistData_t* dists,
                                       int node_degree,
                                       int* locks)
{
  int tx                 = threadIdx.x;
  int lane_id            = tx % raft::warp_size();
  size_t global_idx_base = list_id * node_degree;
  if (elem.id() == list_id) return;

  const int num_segments = ceildiv(node_degree, raft::warp_size());

  int loop_flag = 0;
  do {
    int segment_id = elem.id() % num_segments;
    if (lane_id == 0) {
      loop_flag = atomicCAS(&locks[list_id * num_segments + segment_id], 0, 1) == 0;
    }

    loop_flag = __shfl_sync(raft::warp_full_mask(), loop_flag, 0);

    if (loop_flag == 1) {
      ResultItem<Index_t> knn_list_frag;
      int local_idx     = segment_id * raft::warp_size() + lane_id;
      size_t global_idx = global_idx_base + local_idx;
      if (local_idx < node_degree) {
        knn_list_frag.id_with_flag() = graph[global_idx].id_with_flag();
        knn_list_frag.dist()         = dists[global_idx];
      }

      int pos_to_insert = -1;
      ResultItem<Index_t> prev_elem;

      prev_elem.id_with_flag() =
        __shfl_up_sync(raft::warp_full_mask(), knn_list_frag.id_with_flag(), 1);
      prev_elem.dist() = __shfl_up_sync(raft::warp_full_mask(), knn_list_frag.dist(), 1);

      if (lane_id == 0) {
        prev_elem = ResultItem<Index_t>{std::numeric_limits<Index_t>::min(),
                                        std::numeric_limits<DistData_t>::lowest()};
      }
      if (elem > prev_elem && elem < knn_list_frag) {
        pos_to_insert = segment_id * raft::warp_size() + lane_id;
      } else if (elem == prev_elem || elem == knn_list_frag) {
        pos_to_insert = -2;
      }
      uint mask = __ballot_sync(raft::warp_full_mask(), pos_to_insert >= 0);
      if (mask) {
        uint set_lane_id = __fns(mask, 0, 1);
        pos_to_insert    = __shfl_sync(raft::warp_full_mask(), pos_to_insert, set_lane_id);
      }

      if (pos_to_insert >= 0) {
        int local_idx = segment_id * raft::warp_size() + lane_id;
        if (local_idx > pos_to_insert) {
          local_idx++;
        } else if (local_idx == pos_to_insert) {
          graph[global_idx_base + local_idx].id_with_flag() = elem.id_with_flag();
          dists[global_idx_base + local_idx]                = elem.dist();
          local_idx++;
        }
        size_t global_pos = global_idx_base + local_idx;
        if (local_idx < (segment_id + 1) * raft::warp_size() && local_idx < node_degree) {
          graph[global_pos].id_with_flag() = knn_list_frag.id_with_flag();
          dists[global_pos]                = knn_list_frag.dist();
        }
      }
      __threadfence();
      if (loop_flag && lane_id == 0) { atomicExch(&locks[list_id * num_segments + segment_id], 0); }
    }
  } while (!loop_flag);
}

template <typename Index_t>
__device__ ResultItem<Index_t> get_min_item(const Index_t id,
                                            const int idx_in_list,
                                            const Index_t* neighbs,
                                            const DistData_t* distances,
                                            const bool find_in_row = true)
{
  int lane_id = threadIdx.x % raft::warp_size();

  static_assert(MAX_NUM_BI_SAMPLES == 64);
  int idx[MAX_NUM_BI_SAMPLES / raft::warp_size()];
  float dist[MAX_NUM_BI_SAMPLES / raft::warp_size()] = {std::numeric_limits<DistData_t>::max(),
                                                        std::numeric_limits<DistData_t>::max()};
  idx[0]                                             = lane_id;
  idx[1]                                             = raft::warp_size() + lane_id;

  if (neighbs[idx[0]] != id) {
    dist[0] = find_in_row ? distances[idx_in_list * SKEWED_MAX_NUM_BI_SAMPLES + lane_id]
                          : distances[idx_in_list + lane_id * SKEWED_MAX_NUM_BI_SAMPLES];
  }

  if (neighbs[idx[1]] != id) {
    dist[1] =
      find_in_row
        ? distances[idx_in_list * SKEWED_MAX_NUM_BI_SAMPLES + raft::warp_size() + lane_id]
        : distances[idx_in_list + (raft::warp_size() + lane_id) * SKEWED_MAX_NUM_BI_SAMPLES];
  }

  if (dist[1] < dist[0]) {
    dist[0] = dist[1];
    idx[0]  = idx[1];
  }
  __syncwarp();
  for (int offset = raft::warp_size() >> 1; offset >= 1; offset >>= 1) {
    float other_idx  = __shfl_down_sync(raft::warp_full_mask(), idx[0], offset);
    float other_dist = __shfl_down_sync(raft::warp_full_mask(), dist[0], offset);
    if (other_dist < dist[0]) {
      dist[0] = other_dist;
      idx[0]  = other_idx;
    }
  }

  ResultItem<Index_t> result;
  result.dist()         = __shfl_sync(raft::warp_full_mask(), dist[0], 0);
  result.id_with_flag() = neighbs[__shfl_sync(raft::warp_full_mask(), idx[0], 0)];
  return result;
}

template <typename T>
__device__ __forceinline__ void remove_duplicates(
  T* list_a, int list_a_size, T* list_b, int list_b_size, int& unique_counter, int execute_warp_id)
{
  static_assert(raft::warp_size() == 32);
  if (!(threadIdx.x >= execute_warp_id * raft::warp_size() &&
        threadIdx.x < execute_warp_id * raft::warp_size() + raft::warp_size())) {
    return;
  }
  int lane_id = threadIdx.x % raft::warp_size();
  T elem      = std::numeric_limits<T>::max();
  if (lane_id < list_a_size) { elem = list_a[lane_id]; }
  warp_bitonic_sort(&elem, lane_id);

  if (elem != std::numeric_limits<T>::max()) { list_a[lane_id] = elem; }

  T elem_b = std::numeric_limits<T>::max();

  if (lane_id < list_b_size) { elem_b = list_b[lane_id]; }
  __syncwarp();

  int idx_l    = 0;
  int idx_r    = list_a_size;
  bool existed = false;
  while (idx_l < idx_r) {
    int idx  = (idx_l + idx_r) / 2;
    int elem = list_a[idx];
    if (elem == elem_b) {
      existed = true;
      break;
    }
    if (elem_b > elem) {
      idx_l = idx + 1;
    } else {
      idx_r = idx;
    }
  }
  if (!existed && elem_b != std::numeric_limits<T>::max()) {
    int idx                   = atomicAdd(&unique_counter, 1);
    list_a[list_a_size + idx] = elem_b;
  }
}

// launch_bounds here denote BLOCK_SIZE = 512 and MIN_BLOCKS_PER_SM = 4
// Per
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications,
// MAX_RESIDENT_THREAD_PER_SM = BLOCK_SIZE * BLOCKS_PER_SM = 2048
// For architectures 750 and 860 (890), the values for MAX_RESIDENT_THREAD_PER_SM
// is 1024 and 1536 respectively, which means the bounds don't work anymore
template <typename Index_t,
          typename ID_t        = InternalID_t<Index_t>,
          typename epilogue_op = raft::identity_op>
RAFT_KERNEL
#ifdef __CUDA_ARCH__
#if (__CUDA_ARCH__) == 750 || ((__CUDA_ARCH__) >= 860 && (__CUDA_ARCH__) <= 890)
__launch_bounds__(BLOCK_SIZE)
#else
__launch_bounds__(BLOCK_SIZE, 4)
#endif
#endif
  local_join_kernel(const Index_t* graph_new,
                    const Index_t* rev_graph_new,
                    const int2* sizes_new,
                    const Index_t* graph_old,
                    const Index_t* rev_graph_old,
                    const int2* sizes_old,
                    const int width,
                    const __half* data,
                    const int data_dim,
                    ID_t* graph,
                    DistData_t* dists,
                    int graph_width,
                    int* locks,
                    DistData_t* l2_norms,
                    epilogue_op distance_epilogue)
{
#if (__CUDA_ARCH__ >= 700)
  using namespace nvcuda;
  __shared__ int s_list[MAX_NUM_BI_SAMPLES * 2];

  constexpr int APAD = 8;
  constexpr int BPAD = 8;
  __shared__ __half s_nv[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + APAD];  // New vectors
  __shared__ __half s_ov[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + BPAD];  // Old vectors
  static_assert(sizeof(float) * MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES <=
                sizeof(__half) * MAX_NUM_BI_SAMPLES * (TILE_COL_WIDTH + BPAD));
  // s_distances: MAX_NUM_BI_SAMPLES x SKEWED_MAX_NUM_BI_SAMPLES, reuse the space of s_ov
  float* s_distances    = (float*)&s_ov[0][0];
  int* s_unique_counter = (int*)&s_ov[0][0];

  if (threadIdx.x == 0) {
    s_unique_counter[0] = 0;
    s_unique_counter[1] = 0;
  }

  Index_t* new_neighbors = s_list;
  Index_t* old_neighbors = s_list + MAX_NUM_BI_SAMPLES;

  size_t list_id      = blockIdx.x;
  int2 list_new_size2 = sizes_new[list_id];
  int list_new_size   = list_new_size2.x + list_new_size2.y;
  int2 list_old_size2 = sizes_old[list_id];
  int list_old_size   = list_old_size2.x + list_old_size2.y;

  if (!list_new_size) return;
  int tx = threadIdx.x;

  if (tx < list_new_size2.x) {
    new_neighbors[tx] = graph_new[list_id * width + tx];
  } else if (tx >= list_new_size2.x && tx < list_new_size) {
    new_neighbors[tx] = rev_graph_new[list_id * width + tx - list_new_size2.x];
  }

  if (tx < list_old_size2.x) {
    old_neighbors[tx] = graph_old[list_id * width + tx];
  } else if (tx >= list_old_size2.x && tx < list_old_size) {
    old_neighbors[tx] = rev_graph_old[list_id * width + tx - list_old_size2.x];
  }

  __syncthreads();

  remove_duplicates(new_neighbors,
                    list_new_size2.x,
                    new_neighbors + list_new_size2.x,
                    list_new_size2.y,
                    s_unique_counter[0],
                    0);

  remove_duplicates(old_neighbors,
                    list_old_size2.x,
                    old_neighbors + list_old_size2.x,
                    list_old_size2.y,
                    s_unique_counter[1],
                    1);
  __syncthreads();
  list_new_size = list_new_size2.x + s_unique_counter[0];
  list_old_size = list_old_size2.x + s_unique_counter[1];

  int warp_id             = threadIdx.x / raft::warp_size();
  int lane_id             = threadIdx.x % raft::warp_size();
  constexpr int num_warps = BLOCK_SIZE / raft::warp_size();

  int warp_id_y = warp_id / 4;
  int warp_id_x = warp_id % 4;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0);
  for (int step = 0; step < ceildiv(data_dim, TILE_COL_WIDTH); step++) {
    int num_load_elems = (step == ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                           ? data_dim - step * TILE_COL_WIDTH
                           : TILE_COL_WIDTH;
#pragma unroll
    for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
      int idx = i * num_warps + warp_id;
      if (idx < list_new_size) {
        size_t neighbor_id = new_neighbors[idx];
        size_t idx_in_data = neighbor_id * data_dim;
        load_vec(s_nv[idx],
                 data + idx_in_data + step * TILE_COL_WIDTH,
                 num_load_elems,
                 TILE_COL_WIDTH,
                 lane_id);
      }
    }
    __syncthreads();

    for (int i = 0; i < TILE_COL_WIDTH / WMMA_K; i++) {
      wmma::load_matrix_sync(a_frag, s_nv[warp_id_y * WMMA_M] + i * WMMA_K, TILE_COL_WIDTH + APAD);
      wmma::load_matrix_sync(b_frag, s_nv[warp_id_x * WMMA_N] + i * WMMA_K, TILE_COL_WIDTH + BPAD);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      __syncthreads();
    }
  }

  wmma::store_matrix_sync(
    s_distances + warp_id_y * WMMA_M * SKEWED_MAX_NUM_BI_SAMPLES + warp_id_x * WMMA_N,
    c_frag,
    SKEWED_MAX_NUM_BI_SAMPLES,
    wmma::mem_row_major);
  __syncthreads();

  for (int i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
    auto row_idx = i % SKEWED_MAX_NUM_BI_SAMPLES;
    auto col_idx = i / SKEWED_MAX_NUM_BI_SAMPLES;
    if (row_idx < list_new_size && col_idx < list_new_size) {
      auto r = new_neighbors[row_idx];
      auto c = new_neighbors[col_idx];
      if (l2_norms == nullptr) {
        auto dist_val  = -s_distances[i];
        s_distances[i] = distance_epilogue(dist_val, r, c);
      } else {
        auto dist_val  = l2_norms[r] + l2_norms[c] - 2.0 * s_distances[i];
        s_distances[i] = distance_epilogue(dist_val, r, c);
      }
    } else {
      s_distances[i] = std::numeric_limits<float>::max();
    }
  }
  __syncthreads();

  for (int step = 0; step < ceildiv(list_new_size, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size) continue;
    auto min_elem = get_min_item(s_list[idx_in_list], idx_in_list, new_neighbors, s_distances);
    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }

  if (!list_old_size) return;

  __syncthreads();

  wmma::fill_fragment(c_frag, 0.0);
  for (int step = 0; step < ceildiv(data_dim, TILE_COL_WIDTH); step++) {
    int num_load_elems = (step == ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                           ? data_dim - step * TILE_COL_WIDTH
                           : TILE_COL_WIDTH;
    if (TILE_COL_WIDTH < data_dim) {
#pragma unroll
      for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_new_size) {
          size_t neighbor_id = new_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          load_vec(s_nv[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
      int idx = i * num_warps + warp_id;
      if (idx < list_old_size) {
        size_t neighbor_id = old_neighbors[idx];
        size_t idx_in_data = neighbor_id * data_dim;
        load_vec(s_ov[idx],
                 data + idx_in_data + step * TILE_COL_WIDTH,
                 num_load_elems,
                 TILE_COL_WIDTH,
                 lane_id);
      }
    }
    __syncthreads();

    for (int i = 0; i < TILE_COL_WIDTH / WMMA_K; i++) {
      wmma::load_matrix_sync(a_frag, s_nv[warp_id_y * WMMA_M] + i * WMMA_K, TILE_COL_WIDTH + APAD);
      wmma::load_matrix_sync(b_frag, s_ov[warp_id_x * WMMA_N] + i * WMMA_K, TILE_COL_WIDTH + BPAD);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      __syncthreads();
    }
  }

  wmma::store_matrix_sync(
    s_distances + warp_id_y * WMMA_M * SKEWED_MAX_NUM_BI_SAMPLES + warp_id_x * WMMA_N,
    c_frag,
    SKEWED_MAX_NUM_BI_SAMPLES,
    wmma::mem_row_major);
  __syncthreads();

  for (int i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
    auto row_idx = i % SKEWED_MAX_NUM_BI_SAMPLES;
    auto col_idx = i / SKEWED_MAX_NUM_BI_SAMPLES;
    if (row_idx < list_old_size && col_idx < list_new_size) {
      auto r = old_neighbors[row_idx];
      auto c = new_neighbors[col_idx];
      if (l2_norms == nullptr) {
        auto dist_val  = -s_distances[i];
        s_distances[i] = distance_epilogue(dist_val, r, c);
      } else {
        auto dist_val  = l2_norms[r] + l2_norms[c] - 2.0 * s_distances[i];
        s_distances[i] = distance_epilogue(dist_val, r, c);
      }
    } else {
      s_distances[i] = std::numeric_limits<float>::max();
    }
  }
  __syncthreads();

  for (int step = 0; step < ceildiv(MAX_NUM_BI_SAMPLES * 2, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size && idx_in_list < MAX_NUM_BI_SAMPLES) continue;
    if (idx_in_list >= MAX_NUM_BI_SAMPLES + list_old_size && idx_in_list < MAX_NUM_BI_SAMPLES * 2)
      continue;
    ResultItem<Index_t> min_elem{std::numeric_limits<Index_t>::max(),
                                 std::numeric_limits<DistData_t>::max()};
    if (idx_in_list < MAX_NUM_BI_SAMPLES) {
      auto temp_min_item =
        get_min_item(s_list[idx_in_list], idx_in_list, old_neighbors, s_distances);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    } else {
      auto temp_min_item = get_min_item(
        s_list[idx_in_list], idx_in_list - MAX_NUM_BI_SAMPLES, new_neighbors, s_distances, false);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    }

    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }
#endif
}

namespace {
template <typename Index_t>
int insert_to_ordered_list(InternalID_t<Index_t>* list,
                           DistData_t* dist_list,
                           const int width,
                           const InternalID_t<Index_t> neighb_id,
                           const DistData_t dist)
{
  if (dist > dist_list[width - 1]) { return width; }

  int idx_insert      = width;
  bool position_found = false;
  for (int i = 0; i < width; i++) {
    if (list[i].id() == neighb_id.id()) { return width; }
    if (!position_found && dist_list[i] > dist) {
      idx_insert     = i;
      position_found = true;
    }
  }
  if (idx_insert == width) return idx_insert;

  memmove(list + idx_insert + 1, list + idx_insert, sizeof(*list) * (width - idx_insert - 1));
  memmove(dist_list + idx_insert + 1,
          dist_list + idx_insert,
          sizeof(*dist_list) * (width - idx_insert - 1));

  list[idx_insert]      = neighb_id;
  dist_list[idx_insert] = dist;
  return idx_insert;
};

}  // namespace

template <typename Index_t>
GnndGraph<Index_t>::GnndGraph(const size_t nrow,
                              const size_t node_degree,
                              const size_t internal_node_degree,
                              const size_t num_samples)
  : nrow(nrow),
    node_degree(node_degree),
    num_samples(num_samples),
    bloom_filter(nrow, internal_node_degree / segment_size, 3),
    h_dists{raft::make_host_matrix<DistData_t, size_t, raft::row_major>(nrow, node_degree)},
    h_graph_new(nrow * num_samples),
    h_list_sizes_new(nrow),
    h_graph_old(nrow * num_samples),
    h_list_sizes_old{nrow}
{
  // node_degree must be a multiple of segment_size;
  assert(node_degree % segment_size == 0);
  assert(internal_node_degree % segment_size == 0);

  num_segments = node_degree / segment_size;
  // To save the CPU memory, graph should be allocated by external function
  h_graph = nullptr;
}

// This is the only operation on the CPU that cannot be overlapped.
// So it should be as fast as possible.
template <typename Index_t>
void GnndGraph<Index_t>::sample_graph_new(InternalID_t<Index_t>* new_neighbors, const size_t width)
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    auto list_new         = h_graph_new.data() + i * num_samples;
    h_list_sizes_new[i].x = 0;
    h_list_sizes_new[i].y = 0;

    for (size_t j = 0; j < width; j++) {
      auto new_neighb_id = new_neighbors[i * width + j].id();
      if ((size_t)new_neighb_id >= nrow) break;
      if (bloom_filter.check(i, new_neighb_id)) { continue; }
      bloom_filter.add(i, new_neighb_id);
      new_neighbors[i * width + j].mark_old();
      list_new[h_list_sizes_new[i].x++] = new_neighb_id;
      if (h_list_sizes_new[i].x == num_samples) break;
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::init_random_graph()
{
  for (size_t seg_idx = 0; seg_idx < static_cast<size_t>(num_segments); seg_idx++) {
    // random sequence (range: 0~nrow)
    // segment_x stores neighbors which id % num_segments == x
    std::vector<Index_t> rand_seq(nrow / num_segments);
    std::iota(rand_seq.begin(), rand_seq.end(), 0);
    auto gen = std::default_random_engine{seg_idx};
    std::shuffle(rand_seq.begin(), rand_seq.end(), gen);

#pragma omp parallel for
    for (size_t i = 0; i < nrow; i++) {
      size_t base_idx      = i * node_degree + seg_idx * segment_size;
      auto h_neighbor_list = h_graph + base_idx;
      auto h_dist_list     = h_dists.data_handle() + base_idx;
      for (size_t j = 0; j < static_cast<size_t>(segment_size); j++) {
        size_t idx = base_idx + j;
        Index_t id = rand_seq[idx % rand_seq.size()] * num_segments + seg_idx;
        if ((size_t)id == i) {
          id = rand_seq[(idx + segment_size) % rand_seq.size()] * num_segments + seg_idx;
        }
        h_neighbor_list[j].id_with_flag() = id;
        h_dist_list[j]                    = std::numeric_limits<DistData_t>::max();
      }
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::sample_graph(bool sample_new)
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    h_list_sizes_old[i].x = 0;
    h_list_sizes_old[i].y = 0;
    h_list_sizes_new[i].x = 0;
    h_list_sizes_new[i].y = 0;

    auto list     = h_graph + i * node_degree;
    auto list_old = h_graph_old.data() + i * num_samples;
    auto list_new = h_graph_new.data() + i * num_samples;
    for (int j = 0; j < segment_size; j++) {
      for (int k = 0; k < num_segments; k++) {
        auto neighbor = list[k * segment_size + j];
        if ((size_t)neighbor.id() >= nrow) continue;
        if (!neighbor.is_new()) {
          if (h_list_sizes_old[i].x < num_samples) {
            list_old[h_list_sizes_old[i].x++] = neighbor.id();
          }
        } else if (sample_new) {
          if (h_list_sizes_new[i].x < num_samples) {
            list[k * segment_size + j].mark_old();
            list_new[h_list_sizes_new[i].x++] = neighbor.id();
          }
        }
        if (h_list_sizes_old[i].x == num_samples && h_list_sizes_new[i].x == num_samples) { break; }
      }
      if (h_list_sizes_old[i].x == num_samples && h_list_sizes_new[i].x == num_samples) { break; }
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::update_graph(const InternalID_t<Index_t>* new_neighbors,
                                      const DistData_t* new_dists,
                                      const size_t width,
                                      std::atomic<int64_t>& update_counter)
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    for (size_t j = 0; j < width; j++) {
      auto new_neighb_id = new_neighbors[i * width + j];
      auto new_dist      = new_dists[i * width + j];
      if (new_dist == std::numeric_limits<DistData_t>::max()) break;
      if ((size_t)new_neighb_id.id() == i) continue;
      int seg_idx    = new_neighb_id.id() % num_segments;
      auto list      = h_graph + i * node_degree + seg_idx * segment_size;
      auto dist_list = h_dists.data_handle() + i * node_degree + seg_idx * segment_size;
      int insert_pos =
        insert_to_ordered_list(list, dist_list, segment_size, new_neighb_id, new_dist);
      if (i % counter_interval == 0 && insert_pos != segment_size) { update_counter++; }
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::sort_lists()
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    std::vector<std::pair<DistData_t, Index_t>> new_list;
    for (size_t j = 0; j < node_degree; j++) {
      new_list.emplace_back(h_dists.data_handle()[i * node_degree + j],
                            h_graph[i * node_degree + j].id());
    }
    std::sort(new_list.begin(), new_list.end());
    for (size_t j = 0; j < node_degree; j++) {
      h_graph[i * node_degree + j].id_with_flag() = new_list[j].second;
      h_dists.data_handle()[i * node_degree + j]  = new_list[j].first;
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::clear()
{
  bloom_filter.clear();
}

template <typename Index_t>
GnndGraph<Index_t>::~GnndGraph()
{
  assert(h_graph == nullptr);
}

template <typename Data_t, typename Index_t, typename epilogue_op>
GNND<Data_t, Index_t, epilogue_op>::GNND(raft::resources const& res,
                                         const BuildConfig& build_config)
  : res(res),
    build_config_(build_config),
    graph_(build_config.max_dataset_size,
           align32::roundUp(build_config.node_degree),
           align32::roundUp(build_config.internal_node_degree ? build_config.internal_node_degree
                                                              : build_config.node_degree),
           NUM_SAMPLES),
    nrow_(build_config.max_dataset_size),
    ndim_(build_config.dataset_dim),
    d_data_{raft::make_device_matrix<__half, size_t, raft::row_major>(
      res, nrow_, build_config.dataset_dim)},
    l2_norms_{raft::make_device_vector<DistData_t, size_t>(res, nrow_)},
    graph_buffer_{
      raft::make_device_matrix<ID_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    dists_buffer_{
      raft::make_device_matrix<DistData_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    graph_host_buffer_(nrow_ * DEGREE_ON_DEVICE),
    dists_host_buffer_(nrow_ * DEGREE_ON_DEVICE),
    d_locks_{raft::make_device_vector<int, size_t>(res, nrow_)},
    h_rev_graph_new_(nrow_ * NUM_SAMPLES),
    h_graph_old_(nrow_ * NUM_SAMPLES),
    h_rev_graph_old_(nrow_ * NUM_SAMPLES),
    d_list_sizes_new_{raft::make_device_vector<int2, size_t>(res, nrow_)},
    d_list_sizes_old_{raft::make_device_vector<int2, size_t>(res, nrow_)}
{
  static_assert(NUM_SAMPLES <= 32);

  thrust::fill(thrust::device,
               dists_buffer_.data_handle(),
               dists_buffer_.data_handle() + dists_buffer_.size(),
               std::numeric_limits<float>::max());
  thrust::fill(thrust::device,
               reinterpret_cast<Index_t*>(graph_buffer_.data_handle()),
               reinterpret_cast<Index_t*>(graph_buffer_.data_handle()) + graph_buffer_.size(),
               std::numeric_limits<Index_t>::max());
  thrust::fill(thrust::device, d_locks_.data_handle(), d_locks_.data_handle() + d_locks_.size(), 0);
};

template <typename Data_t, typename Index_t, typename epilogue_op>
void GNND<Data_t, Index_t, epilogue_op>::add_reverse_edges(Index_t* graph_ptr,
                                                           Index_t* h_rev_graph_ptr,
                                                           Index_t* d_rev_graph_ptr,
                                                           int2* list_sizes,
                                                           cudaStream_t stream)
{
  add_rev_edges_kernel<<<nrow_, raft::warp_size(), 0, stream>>>(
    graph_ptr, d_rev_graph_ptr, NUM_SAMPLES, list_sizes);
  raft::copy(
    h_rev_graph_ptr, d_rev_graph_ptr, nrow_ * NUM_SAMPLES, raft::resource::get_cuda_stream(res));
}

template <typename Data_t, typename Index_t, typename epilogue_op>
void GNND<Data_t, Index_t, epilogue_op>::local_join(cudaStream_t stream,
                                                    epilogue_op distance_epilogue)
{
  thrust::fill(thrust::device.on(stream),
               dists_buffer_.data_handle(),
               dists_buffer_.data_handle() + dists_buffer_.size(),
               std::numeric_limits<float>::max());
  local_join_kernel<<<nrow_, BLOCK_SIZE, 0, stream>>>(
    thrust::raw_pointer_cast(graph_.h_graph_new.data()),
    thrust::raw_pointer_cast(h_rev_graph_new_.data()),
    d_list_sizes_new_.data_handle(),
    thrust::raw_pointer_cast(h_graph_old_.data()),
    thrust::raw_pointer_cast(h_rev_graph_old_.data()),
    d_list_sizes_old_.data_handle(),
    NUM_SAMPLES,
    d_data_.data_handle(),
    ndim_,
    graph_buffer_.data_handle(),
    dists_buffer_.data_handle(),
    DEGREE_ON_DEVICE,
    d_locks_.data_handle(),
    l2_norms_.data_handle(),
    distance_epilogue);
}

template <typename Data_t, typename Index_t, typename epilogue_op>
void GNND<Data_t, Index_t, epilogue_op>::build(Data_t* data,
                                               const Index_t nrow,
                                               Index_t* output_graph,
                                               bool return_distances,
                                               DistData_t* output_distances,
                                               epilogue_op distance_epilogue)
{
  using input_t = typename std::remove_const<Data_t>::type;

  cudaStream_t stream = raft::resource::get_cuda_stream(res);
  nrow_               = nrow;
  graph_.h_graph      = (InternalID_t<Index_t>*)output_graph;

  cudaPointerAttributes data_ptr_attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&data_ptr_attr, data));
  size_t batch_size = (data_ptr_attr.devicePointer == nullptr) ? 100000 : nrow_;

  raft::spatial::knn::detail::utils::batch_load_iterator vec_batches{
    data, static_cast<size_t>(nrow_), build_config_.dataset_dim, batch_size, stream};
  for (auto const& batch : vec_batches) {
    preprocess_data_kernel<<<
      batch.size(),
      raft::warp_size(),
      sizeof(Data_t) * ceildiv(build_config_.dataset_dim, static_cast<size_t>(raft::warp_size())) *
        raft::warp_size(),
      stream>>>(batch.data(),
                d_data_.data_handle(),
                build_config_.dataset_dim,
                l2_norms_.data_handle(),
                batch.offset());
  }

  thrust::fill(thrust::device.on(stream),
               (Index_t*)graph_buffer_.data_handle(),
               (Index_t*)graph_buffer_.data_handle() + graph_buffer_.size(),
               std::numeric_limits<Index_t>::max());

  graph_.clear();
  graph_.init_random_graph();
  graph_.sample_graph(true);

  auto update_and_sample = [&](bool update_graph) {
    if (update_graph) {
      update_counter_ = 0;
      graph_.update_graph(thrust::raw_pointer_cast(graph_host_buffer_.data()),
                          thrust::raw_pointer_cast(dists_host_buffer_.data()),
                          DEGREE_ON_DEVICE,
                          update_counter_);
      if (update_counter_ < build_config_.termination_threshold * nrow_ *
                              build_config_.dataset_dim / counter_interval) {
        update_counter_ = -1;
      }
    }
    graph_.sample_graph(false);
  };

  for (size_t it = 0; it < build_config_.max_iterations; it++) {
    raft::copy(d_list_sizes_new_.data_handle(),
               thrust::raw_pointer_cast(graph_.h_list_sizes_new.data()),
               nrow_,
               raft::resource::get_cuda_stream(res));
    raft::copy(thrust::raw_pointer_cast(h_graph_old_.data()),
               thrust::raw_pointer_cast(graph_.h_graph_old.data()),
               nrow_ * NUM_SAMPLES,
               raft::resource::get_cuda_stream(res));
    raft::copy(d_list_sizes_old_.data_handle(),
               thrust::raw_pointer_cast(graph_.h_list_sizes_old.data()),
               nrow_,
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    std::thread update_and_sample_thread(update_and_sample, it);

    RAFT_LOG_DEBUG("# GNND iteraton: %lu / %lu", it + 1, build_config_.max_iterations);

    // Reuse dists_buffer_ to save GPU memory. graph_buffer_ cannot be reused, because it
    // contains some information for local_join.
    static_assert(DEGREE_ON_DEVICE * sizeof(*(dists_buffer_.data_handle())) >=
                  NUM_SAMPLES * sizeof(*(graph_buffer_.data_handle())));
    add_reverse_edges(thrust::raw_pointer_cast(graph_.h_graph_new.data()),
                      thrust::raw_pointer_cast(h_rev_graph_new_.data()),
                      (Index_t*)dists_buffer_.data_handle(),
                      d_list_sizes_new_.data_handle(),
                      stream);
    add_reverse_edges(thrust::raw_pointer_cast(h_graph_old_.data()),
                      thrust::raw_pointer_cast(h_rev_graph_old_.data()),
                      (Index_t*)dists_buffer_.data_handle(),
                      d_list_sizes_old_.data_handle(),
                      stream);

    // Tensor operations from `mma.h` are guarded with archicteture
    // __CUDA_ARCH__ >= 700. Since RAFT supports compilation for ARCH 600,
    // we need to ensure that `local_join_kernel` (which uses tensor) operations
    // is not only not compiled, but also a runtime error is presented to the user
    auto kernel       = preprocess_data_kernel<input_t>;
    void* kernel_ptr  = reinterpret_cast<void*>(kernel);
    auto runtime_arch = raft::util::arch::kernel_virtual_arch(kernel_ptr);
    auto wmma_range =
      raft::util::arch::SM_range(raft::util::arch::SM_70(), raft::util::arch::SM_future());

    if (wmma_range.contains(runtime_arch)) {
      local_join(stream, distance_epilogue);
    } else {
      THROW("NN_DESCENT cannot be run for __CUDA_ARCH__ < 700");
    }

    update_and_sample_thread.join();

    if (update_counter_ == -1) { break; }
    raft::copy(thrust::raw_pointer_cast(graph_host_buffer_.data()),
               graph_buffer_.data_handle(),
               nrow_ * DEGREE_ON_DEVICE,
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    raft::copy(thrust::raw_pointer_cast(dists_host_buffer_.data()),
               dists_buffer_.data_handle(),
               nrow_ * DEGREE_ON_DEVICE,
               raft::resource::get_cuda_stream(res));

    graph_.sample_graph_new(thrust::raw_pointer_cast(graph_host_buffer_.data()), DEGREE_ON_DEVICE);
  }

  graph_.update_graph(thrust::raw_pointer_cast(graph_host_buffer_.data()),
                      thrust::raw_pointer_cast(dists_host_buffer_.data()),
                      DEGREE_ON_DEVICE,
                      update_counter_);
  raft::resource::sync_stream(res);
  graph_.sort_lists();

  // Reuse graph_.h_dists as the buffer for shrink the lists in graph
  static_assert(sizeof(decltype(*(graph_.h_dists.data_handle()))) >= sizeof(Index_t));

  if (return_distances) {
    auto graph_d_dists = raft::make_device_matrix<DistData_t, int64_t, raft::row_major>(
      res, nrow_, build_config_.node_degree);
    raft::copy(graph_d_dists.data_handle(),
               graph_.h_dists.data_handle(),
               nrow_ * build_config_.node_degree,
               raft::resource::get_cuda_stream(res));

    auto output_dist_view = raft::make_device_matrix_view<DistData_t, int64_t, raft::row_major>(
      output_distances, nrow_, build_config_.output_graph_degree);

    raft::matrix::slice_coordinates coords{static_cast<int64_t>(0),
                                           static_cast<int64_t>(0),
                                           static_cast<int64_t>(nrow_),
                                           static_cast<int64_t>(build_config_.output_graph_degree)};
    raft::matrix::slice<DistData_t, int64_t, raft::row_major>(
      res, raft::make_const_mdspan(graph_d_dists.view()), output_dist_view, coords);
  }

  Index_t* graph_shrink_buffer = (Index_t*)graph_.h_dists.data_handle();

#pragma omp parallel for
  for (size_t i = 0; i < (size_t)nrow_; i++) {
    for (size_t j = 0; j < build_config_.node_degree; j++) {
      size_t idx = i * graph_.node_degree + j;
      int id     = graph_.h_graph[idx].id();
      if (id < static_cast<int>(nrow_)) {
        graph_shrink_buffer[i * build_config_.node_degree + j] = id;
      } else {
        graph_shrink_buffer[i * build_config_.node_degree + j] =
          raft::neighbors::cagra::detail::device::xorshift64(idx) % nrow_;
      }
    }
  }
  graph_.h_graph = nullptr;

#pragma omp parallel for
  for (size_t i = 0; i < (size_t)nrow_; i++) {
    for (size_t j = 0; j < build_config_.node_degree; j++) {
      output_graph[i * build_config_.node_degree + j] =
        graph_shrink_buffer[i * build_config_.node_degree + j];
    }
  }
}

template <typename T,
          typename IdxT        = uint32_t,
          typename epilogue_op = raft::identity_op,
          typename Accessor =
            host_device_accessor<std::experimental::default_accessor<T>, memory_type::host>>
void build(raft::resources const& res,
           const index_params& params,
           mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> dataset,
           index<IdxT>& idx,
           epilogue_op distance_epilogue = raft::identity_op())
{
  RAFT_EXPECTS(dataset.extent(0) < std::numeric_limits<int>::max() - 1,
               "The dataset size for GNND should be less than %d",
               std::numeric_limits<int>::max() - 1);
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  if (intermediate_degree >= static_cast<size_t>(dataset.extent(0))) {
    RAFT_LOG_WARN(
      "Intermediate graph degree cannot be larger than dataset size, reducing it to %lu",
      dataset.extent(0));
    intermediate_degree = dataset.extent(0) - 1;
  }
  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  // The elements in each knn-list are partitioned into different buckets, and we need more buckets
  // to mitigate bucket collisions. `intermediate_degree` is OK to larger than
  // extended_graph_degree.
  size_t extended_graph_degree =
    align32::roundUp(static_cast<size_t>(graph_degree * (graph_degree <= 32 ? 1.0 : 1.3)));
  size_t extended_intermediate_degree = align32::roundUp(
    static_cast<size_t>(intermediate_degree * (intermediate_degree <= 32 ? 1.0 : 1.3)));

  auto int_graph = raft::make_host_matrix<int, int64_t, row_major>(
    dataset.extent(0), static_cast<int64_t>(extended_graph_degree));

  BuildConfig build_config{.max_dataset_size      = static_cast<size_t>(dataset.extent(0)),
                           .dataset_dim           = static_cast<size_t>(dataset.extent(1)),
                           .node_degree           = extended_graph_degree,
                           .internal_node_degree  = extended_intermediate_degree,
                           .max_iterations        = params.max_iterations,
                           .termination_threshold = params.termination_threshold,
                           .output_graph_degree   = params.graph_degree};

  GNND<const T, int, epilogue_op> nnd(res, build_config);

  if (idx.distances().has_value() || !params.return_distances) {
    nnd.build(dataset.data_handle(),
              dataset.extent(0),
              int_graph.data_handle(),
              params.return_distances,
              idx.distances()
                .value_or(raft::make_device_matrix<float, int64_t>(res, 0, 0).view())
                .data_handle(),
              distance_epilogue);
  } else {
    RAFT_EXPECTS(!params.return_distances,
                 "Distance view not allocated. Using return_distances set to true requires "
                 "distance view to be allocated.");
  }

#pragma omp parallel for
  for (size_t i = 0; i < static_cast<size_t>(dataset.extent(0)); i++) {
    for (size_t j = 0; j < graph_degree; j++) {
      auto graph                  = idx.graph().data_handle();
      graph[i * graph_degree + j] = int_graph.data_handle()[i * extended_graph_degree + j];
    }
  }
}

template <typename T,
          typename IdxT        = uint32_t,
          typename epilogue_op = raft::identity_op,
          typename Accessor =
            host_device_accessor<std::experimental::default_accessor<T>, memory_type::host>>
index<IdxT> build(raft::resources const& res,
                  const index_params& params,
                  mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> dataset,
                  epilogue_op distance_epilogue = raft::identity_op())
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  index<IdxT> idx{
    res, dataset.extent(0), static_cast<int64_t>(graph_degree), params.return_distances};

  build(res, params, dataset, idx, distance_epilogue);

  return idx;
}

}  // namespace raft::neighbors::experimental::nn_descent::detail
