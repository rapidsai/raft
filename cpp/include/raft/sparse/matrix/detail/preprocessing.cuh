/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/map_reduce.cuh>
#include <raft/matrix/init.cuh>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/neighbors/cross_component_nn.cuh>
#include <raft/sparse/op/sort.cuh>

#include <thrust/reduce.h>

namespace raft::sparse::matrix::detail {

__global__ void _scan(int* rows, int nnz, int* counts)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nnz) { return; }
  if (index == 0) {
    counts[index] = 1;
    return;
  }
  if (index < nnz) {
    int curr_id = rows[index];
    int old_id  = rows[index - 1];
    if (curr_id != old_id) {
      counts[index] = 1;
    } else {
      counts[index] = 0;
    }
  }
}

__global__ void _fit_compute_occurs(int* cols, int nnz, int* counts, int* feats, int vocabSize)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if ((index < nnz) && (counts[index] == 1)) {
    int targetVal = cols[index];
    int vocab     = targetVal % vocabSize;
    while (targetVal == cols[index]) {
      feats[vocab] = feats[vocab] + 1;
      index++;
      if (index >= nnz) { return; }
    }
  }
}

__global__ void _transform(int* rows,
                           int* columns,
                           float* values,
                           int* feat_id_count,
                           int* counts,
                           float* out_values,
                           int num_rows,
                           float avgRowLen,
                           float k,
                           float b,
                           int nnz,
                           int vocabSize,
                           bool bm25 = false)
{
  int start_index = blockIdx.x * blockDim.x + threadIdx.x;
  int index       = start_index;
  if (index < nnz && counts[index] == 1) {
    int row_length = 0;
    int targetVal  = rows[index];
    while (targetVal == rows[index]) {
      row_length = row_length + values[index];
      index++;
      if (index >= nnz) { break; }
    }
    index = start_index;
    float result;
    while (targetVal == rows[index]) {
      int col       = columns[index];
      int vocab     = col % vocabSize;
      float tf      = (float)values[index] / row_length;
      double idf_in = (double)num_rows / feat_id_count[vocab];
      float idf     = (float)raft::log<double>(idf_in);
      result        = tf * idf;
      if (bm25) {
        float bm = ((k + 1) * tf) / (k * ((1.0f - b) + b * (row_length / avgRowLen)) + tf);
        result   = idf * bm;
      }
      out_values[index] = result;
      index++;
      if (index >= nnz) { break; }
    }
  }
}

template <typename ValueType, typename IndexType>
void convert_csr_to_coo(raft::resources& handle,
                        raft::device_csr_matrix<ValueType,
                                                IndexType,
                                                IndexType,
                                                IndexType,
                                                raft::device_uvector_policy,
                                                raft::PRESERVING> csr_in,
                        raft::device_vector_view<IndexType, int64_t> rows,
                        raft::device_vector_view<IndexType, int64_t> columns,
                        raft::device_vector_view<ValueType, int64_t> values)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto nnz            = csr_in.structure_view().get_nnz();
  auto indptr         = csr_in.structure_view().get_indptr();
  auto indices        = csr_in.structure_view().get_indices();
  auto vals           = csr_in.view().get_elements();

  raft::sparse::convert::csr_to_coo(
    indptr.data(), (int)indptr.size(), rows.data_handle(), (int)nnz, stream);
  raft::copy(columns.data_handle(), indices.data(), (int)nnz, stream);
  raft::copy(values.data_handle(), vals.data(), (int)nnz, stream);
}

}  // namespace raft::sparse::matrix::detail