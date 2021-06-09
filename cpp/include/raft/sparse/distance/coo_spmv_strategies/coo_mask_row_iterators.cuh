/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "../common.h"
#include "../utils.cuh"

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

namespace raft {
namespace sparse {
namespace distance {

template <typename value_idx>
class mask_row_it {
 public:
  mask_row_it(const value_idx *full_indptr_, const value_idx &n_rows_,
              value_idx *mask_row_idx_ = NULL)
    : full_indptr(full_indptr_), mask_row_idx(mask_row_idx_), n_rows(n_rows_) {}

  __device__ inline value_idx get_row_idx(const int &n_blocks_nnz_b) {
    if (mask_row_idx != NULL) {
      return mask_row_idx[blockIdx.x / n_blocks_nnz_b];
    } else {
      return blockIdx.x / n_blocks_nnz_b;
    }
  }

  __device__ inline void get_row_offsets(
    const value_idx &row_idx, value_idx &start_offset, value_idx &stop_offset,
    const value_idx &n_blocks_nnz_b, bool &first_a_chunk, bool &last_a_chunk) {
    start_offset = full_indptr[row_idx];
    stop_offset = full_indptr[row_idx + 1] - 1;
  }

  __device__ constexpr inline void get_indices_boundary(
    const value_idx *indices, value_idx &indices_len, value_idx &start_offset,
    value_idx &stop_offset, value_idx &start_index, value_idx &stop_index,
    bool &first_a_chunk, bool &last_a_chunk) {
    // do nothing;
  }

  __device__ constexpr inline bool check_indices_bounds(
    value_idx &start_index_a, value_idx &stop_index_a, value_idx &index_b) {
    return true;
  }

  const value_idx *full_indptr, &n_rows;
  value_idx *mask_row_idx;
};

template <typename value_idx>
__global__ void fill_chunk_indices_kernel(value_idx *n_chunks_per_row,
                                          value_idx *chunk_indices,
                                          value_idx n_rows) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_rows) {
    auto start = n_chunks_per_row[tid];
    auto end = n_chunks_per_row[tid + 1];

// auto row_idx = mask_row_idx[tid];
#pragma unroll
    for (int i = start; i < end; i++) {
      chunk_indices[i] = tid;
    }
  }
}

template <typename value_idx>
class chunked_mask_row_it : public mask_row_it<value_idx> {
 public:
  chunked_mask_row_it(const value_idx *full_indptr_, const value_idx &n_rows_,
                      value_idx *mask_row_idx_, int row_chunk_size_,
                      const cudaStream_t stream_)
    : mask_row_it<value_idx>(full_indptr_, n_rows_, mask_row_idx_),
      row_chunk_size(row_chunk_size_),
      n_chunks_per_row(n_rows_ + 1, 0),
      stream(stream_) {}

  void init() {
    auto policy = rmm::exec_policy(stream);

    n_chunks_per_row_functor chunk_functor(this->full_indptr, row_chunk_size);
    thrust::transform(policy, this->mask_row_idx,
                      this->mask_row_idx + this->n_rows,
                      n_chunks_per_row.begin() + 1, chunk_functor);

    thrust::inclusive_scan(policy, n_chunks_per_row.begin() + 1,
                           n_chunks_per_row.end(),
                           n_chunks_per_row.begin() + 1);

    n_chunks_per_row_ptr = n_chunks_per_row.data().get();
    raft::update_host(&total_row_blocks, n_chunks_per_row_ptr + this->n_rows, 1,
                      stream);

    fill_chunk_indices();
  }

  __device__ inline value_idx get_row_idx(const int &n_blocks_nnz_b) {
    return this->mask_row_idx[chunk_indices_ptr[blockIdx.x / n_blocks_nnz_b]];
  }

  __device__ inline void get_row_offsets(
    const value_idx &row_idx, value_idx &start_offset, value_idx &stop_offset,
    const int &n_blocks_nnz_b, bool &first_a_chunk, bool &last_a_chunk) {
    auto chunk_index = blockIdx.x / n_blocks_nnz_b;
    auto chunk_val = chunk_indices_ptr[chunk_index];
    auto prev_n_chunks = n_chunks_per_row_ptr[chunk_val];
    auto relative_chunk = chunk_index - prev_n_chunks;
    first_a_chunk = relative_chunk == 0;

    start_offset = this->full_indptr[row_idx] + relative_chunk * row_chunk_size;
    stop_offset = start_offset + row_chunk_size;

    auto final_stop_offset = this->full_indptr[row_idx + 1];

    last_a_chunk = stop_offset >= final_stop_offset;
    stop_offset = last_a_chunk ? final_stop_offset - 1 : stop_offset - 1;
  }

  __device__ inline void get_indices_boundary(
    const value_idx *indices, value_idx &row_idx, value_idx &start_offset,
    value_idx &stop_offset, value_idx &start_index, value_idx &stop_index,
    bool &first_a_chunk, bool &last_a_chunk) {
    start_index = first_a_chunk ? start_index : indices[start_offset - 1] + 1;
    stop_index = last_a_chunk ? stop_index : indices[stop_offset];
  }

  __device__ inline bool check_indices_bounds(value_idx &start_index_a,
                                              value_idx &stop_index_a,
                                              value_idx &index_b) {
    return (index_b >= start_index_a && index_b <= stop_index_a);
  }

  value_idx total_row_blocks;

  const cudaStream_t stream;
  rmm::device_vector<value_idx> n_chunks_per_row, chunk_indices;
  value_idx *n_chunks_per_row_ptr, *chunk_indices_ptr;
  value_idx row_chunk_size;

  struct n_chunks_per_row_functor {
   public:
    n_chunks_per_row_functor(const value_idx *indptr_,
                             value_idx row_chunk_size_)
      : indptr(indptr_), row_chunk_size(row_chunk_size_) {}

    __host__ __device__ value_idx operator()(const value_idx &i) {
      auto degree = indptr[i + 1] - indptr[i];
      return raft::ceildiv(degree, (value_idx)row_chunk_size);
    }

    const value_idx *indptr;
    value_idx row_chunk_size;
  };

  void fill_chunk_indices() {
    auto n_threads = std::min(this->n_rows, 256);
    auto n_blocks = raft::ceildiv(this->n_rows, (value_idx)n_threads);

    chunk_indices = rmm::device_vector<value_idx>(total_row_blocks);

    chunk_indices_ptr = chunk_indices.data().get();

    fill_chunk_indices_kernel<value_idx><<<n_blocks, n_threads, 0, stream>>>(
      n_chunks_per_row_ptr, chunk_indices_ptr, this->n_rows);
  }
};

}  // namespace distance
}  // namespace sparse
}  // namespace raft