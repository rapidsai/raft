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

/**
 * This function creates a representation of input data (rows) that identifies when
 * value changes in the input data. This function assumes data is sorted.
 *
 * @param[in] rows
 *   The input data
 * @param[in] nnz
 *   The size of the input data.
 * @param[in] counts
 *   The resulting representation of the index value changes of the input. Should be
 *   the same size as the input (nnz)
 */
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

/**
 * This function counts the occurrences of the input array. Uses modulo logic as a
 * rudimentary hash (should be changed with better hash function).
 *
 * @param[in] cols
 *   The input data
 * @param[in] nnz
 *   The size of the input data.
 * @param[in] counts
 *   The resulting representation of the index value changes of the input. Should be
 *   the same size as the input (nnz)
 * @param[in] feats
 *   The array that will house the occurrence counts
 * @param[in] vocabSize
 *   The size of the occurrence counts array (feats).
 */
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

/**
 * This function calculates tfidf or bm25, depending on options supplied, from the
 * values input array.
 *
 * @param[in] rows
 *   The input rows.
 * @param[in] columns
 *   The input columns (features).
 * @param[in] values
 *   The input values.
 * @param[in] feat_id_count
 *   The array holding the feature(column) occurrence counts for all fitted inputs.
 * @param[in] counts
 *   The array representing value changes in rows input.
 * @param[in] out_values
 *   The array that will store calculated values, should be size NNZ.
 * @param[in] vocabSize
 *   The number of the features (columns).
 * @param[in] num_rows
 *   Total number of rows for all fitted inputs.
 * @param[in] avgRowLen
 *   The average length of a row (sum of all values for each row).
 * @param[in] k
 *   The bm25 formula variable. Helps with optimization.
 * @param[in] b
 *   The bm25 formula variable. Helps with optimization.
 * @param[in] nnz
 *   The size of the input arrays (rows, columns, values).
 * @param[in] bm25
 *   Boolean that activates bm25 calculation instead of tfidf
 */
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
      row_length += values[index];
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

/**
 * This function converts a raft csr matrix in to a coo (rows, columns,values)
 * representation.
 *
 * @param[in] handle
 *   The input data
 * @param[in] csr_in
 *   The input raft csr matrix.
 * @param[in] rows
 *   The output rows from the csr conversion.
 * @param[in] columns
 *   The output columns from the csr conversion.
 * @param[in] values
 *   The output values from the csr conversion.
 */
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
