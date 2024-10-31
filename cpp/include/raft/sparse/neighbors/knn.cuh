/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

/**
 * DISCLAIMER: this file is deprecated: use knn.cuh instead
 */

#pragma once

#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
#pragma message(__FILE__                                                    \
                  " is deprecated and will be removed in a future release." \
                  " Please use the sparse/spatial version instead.")
#endif

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/neighbors/brute_force.cuh>
#include <raft/sparse/op/sort.cuh>

namespace raft::sparse::neighbors {

/**
 * Search the sparse kNN for the k-nearest neighbors of a set of sparse query vectors
 * using some distance implementation
 * @param[in] idxIndptr csr indptr of the index matrix (size n_idx_rows + 1)
 * @param[in] idxIndices csr column indices array of the index matrix (size n_idx_nnz)
 * @param[in] idxData csr data array of the index matrix (size idxNNZ)
 * @param[in] idxNNZ number of non-zeros for sparse index matrix
 * @param[in] n_idx_rows number of data samples in index matrix
 * @param[in] n_idx_cols
 * @param[in] queryIndptr csr indptr of the query matrix (size n_query_rows + 1)
 * @param[in] queryIndices csr indices array of the query matrix (size queryNNZ)
 * @param[in] queryData csr data array of the query matrix (size queryNNZ)
 * @param[in] queryNNZ number of non-zeros for sparse query matrix
 * @param[in] n_query_rows number of data samples in query matrix
 * @param[in] n_query_cols number of features in query matrix
 * @param[out] output_indices dense matrix for output indices (size n_query_rows * k)
 * @param[out] output_dists dense matrix for output distances (size n_query_rows * k)
 * @param[in] k the number of neighbors to query
 * @param[in] handle CUDA resource::get_cuda_stream(handle) to order operations with respect to
 * @param[in] batch_size_index maximum number of rows to use from index matrix per batch
 * @param[in] batch_size_query maximum number of rows to use from query matrix per batch
 * @param[in] metric distance metric/measure to use
 * @param[in] metricArg potential argument for metric (currently unused)
 */
template <typename value_idx = int, typename value_t = float>
void brute_force_knn(const value_idx* idxIndptr,
                     const value_idx* idxIndices,
                     const value_t* idxData,
                     size_t idxNNZ,
                     int n_idx_rows,
                     int n_idx_cols,
                     const value_idx* queryIndptr,
                     const value_idx* queryIndices,
                     const value_t* queryData,
                     size_t queryNNZ,
                     int n_query_rows,
                     int n_query_cols,
                     value_idx* output_indices,
                     value_t* output_dists,
                     int k,
                     raft::resources const& handle,
                     size_t batch_size_index             = 2 << 14,  // approx 1M
                     size_t batch_size_query             = 2 << 14,
                     raft::distance::DistanceType metric = raft::distance::DistanceType::L2Expanded,
                     float metricArg                     = 0)
{
  brute_force::knn<value_idx, value_t>(idxIndptr,
                                       idxIndices,
                                       idxData,
                                       idxNNZ,
                                       n_idx_rows,
                                       n_idx_cols,
                                       queryIndptr,
                                       queryIndices,
                                       queryData,
                                       queryNNZ,
                                       n_query_rows,
                                       n_query_cols,
                                       output_indices,
                                       output_dists,
                                       k,
                                       handle,
                                       batch_size_index,
                                       batch_size_query,
                                       metric,
                                       metricArg);
}

/**
 * Search the sparse kNN for the k-nearest neighbors of a set of sparse query vectors
 * using some distance implementation
 * @param[in] csr_idx index csr matrix
 * @param[in] csr_query query csr matrix
 * @param[out] output_indices dense matrix for output indices (size n_query_rows * k)
 * @param[out] output_dists dense matrix for output distances (size n_query_rows * k)
 * @param[in] k the number of neighbors to query
 * @param[in] handle CUDA resource::get_cuda_stream(handle) to order operations with respect to
 * @param[in] batch_size_index maximum number of rows to use from index matrix per batch
 * @param[in] batch_size_query maximum number of rows to use from query matrix per batch
 * @param[in] metric distance metric/measure to use
 * @param[in] metricArg potential argument for metric (currently unused)
 */
template <typename value_idx = int, typename value_t = float>
void brute_force_knn(raft::device_csr_matrix<value_t,
                                             value_idx,
                                             value_idx,
                                             value_idx,
                                             raft::device_uvector_policy,
                                             raft::PRESERVING> csr_idx,
                     raft::device_csr_matrix<value_t,
                                             value_idx,
                                             value_idx,
                                             value_idx,
                                             raft::device_uvector_policy,
                                             raft::PRESERVING> csr_query,
                     device_vector_view<value_idx> output_indices,
                     device_vector_view<value_t> output_dists,
                     int k,
                     raft::resources const& handle,
                     size_t batch_size_index             = 2 << 14,  // approx 1M
                     size_t batch_size_query             = 2 << 14,
                     raft::distance::DistanceType metric = raft::distance::DistanceType::L2Expanded,
                     float metricArg                     = 0)
{
  auto idxIndptr  = csr_idx.structure_view().get_indptr();
  auto idxIndices = csr_idx.structure_view().get_indices();
  auto idxData    = csr_idx.view().get_elements();

  auto queryIndptr  = csr_query.structure_view().get_indptr();
  auto queryIndices = csr_query.structure_view().get_indices();
  auto queryData    = csr_query.view().get_elements();

  brute_force::knn<value_idx, value_t>(idxIndptr.data(),
                                       idxIndices.data(),
                                       idxData.data(),
                                       idxIndices.size(),
                                       idxIndptr.size() - 1,
                                       csr_idx.structure_view().get_n_cols(),
                                       queryIndptr.data(),
                                       queryIndices.data(),
                                       queryData.data(),
                                       queryIndices.size(),
                                       queryIndptr.size() - 1,
                                       csr_query.structure_view().get_n_cols(),
                                       output_indices.data_handle(),
                                       output_dists.data_handle(),
                                       k,
                                       handle,
                                       batch_size_index,
                                       batch_size_query,
                                       metric,
                                       metricArg);
}

/**
 * Search the sparse kNN for the k-nearest neighbors of a set of sparse query vectors
 * using some distance implementation
 * @param[in] coo_idx index coo matrix
 * @param[in] coo_query query coo matrix
 * @param[out] output_indices dense matrix for output indices (size n_query_rows * k)
 * @param[out] output_dists dense matrix for output distances (size n_query_rows * k)
 * @param[in] k the number of neighbors to query
 * @param[in] handle CUDA resource::get_cuda_stream(handle) to order operations with respect to
 * @param[in] batch_size_index maximum number of rows to use from index matrix per batch
 * @param[in] batch_size_query maximum number of rows to use from query matrix per batch
 * @param[in] metric distance metric/measure to use
 * @param[in] metricArg potential argument for metric (currently unused)
 */
template <typename value_idx = int, typename value_t = float>
void brute_force_knn(raft::device_coo_matrix<value_t,
                                             value_idx,
                                             value_idx,
                                             value_idx,
                                             raft::device_uvector_policy,
                                             raft::PRESERVING> coo_idx,
                     raft::device_coo_matrix<value_t,
                                             value_idx,
                                             value_idx,
                                             value_idx,
                                             raft::device_uvector_policy,
                                             raft::PRESERVING> coo_query,
                     device_vector_view<value_idx> output_indices,
                     device_vector_view<value_t> output_dists,
                     int k,
                     raft::resources const& handle,
                     size_t batch_size_index             = 2 << 14,  // approx 1M
                     size_t batch_size_query             = 2 << 14,
                     raft::distance::DistanceType metric = raft::distance::DistanceType::L2Expanded,
                     float metricArg                     = 0)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto idxRows = coo_idx.structure_view().get_rows();
  auto idxCols = coo_idx.structure_view().get_cols();
  auto idxData = coo_idx.view().get_elements();

  auto queryRows = coo_query.structure_view().get_rows();
  auto queryCols = coo_query.structure_view().get_cols();
  auto queryData = coo_query.view().get_elements();

  raft::sparse::op::coo_sort(int(idxRows.size()),
                             int(idxCols.size()),
                             int(idxData.size()),
                             idxRows.data(),
                             idxCols.data(),
                             idxRows.data(),
                             stream);

  raft::sparse::op::coo_sort(int(queryRows.size()),
                             int(queryCols.size()),
                             int(queryData.size()),
                             queryRows.data(),
                             queryCols.data(),
                             queryData.data(),
                             stream);
  // + 1 is to account for the 0 at the beginning of the csr representation
  auto idxRowsCsr = raft::make_device_vector<value_idx, int64_t>(
    handle, coo_query.structure_view().get_n_rows() + 1);
  auto queryRowsCsr = raft::make_device_vector<value_idx, int64_t>(
    handle, coo_query.structure_view().get_n_rows() + 1);

  raft::sparse::convert::sorted_coo_to_csr(idxRows.data(),
                                           int(idxRows.size()),
                                           idxRowsCsr.data_handle(),
                                           coo_idx.structure_view().get_n_rows() + 1,
                                           stream);

  raft::sparse::convert::sorted_coo_to_csr(queryRows.data(),
                                           int(queryRows.size()),
                                           queryRowsCsr.data_handle(),
                                           coo_query.structure_view().get_n_rows() + 1,
                                           stream);

  brute_force::knn<value_idx, value_t>(idxRowsCsr.data_handle(),
                                       idxCols.data(),
                                       idxData.data(),
                                       idxCols.size(),
                                       idxRowsCsr.size() - 1,
                                       coo_idx.structure_view().get_n_cols(),
                                       queryRowsCsr.data_handle(),
                                       queryCols.data(),
                                       queryData.data(),
                                       queryCols.size(),
                                       queryRowsCsr.size() - 1,
                                       coo_query.structure_view().get_n_cols(),
                                       output_indices.data_handle(),
                                       output_dists.data_handle(),
                                       k,
                                       handle,
                                       batch_size_index,
                                       batch_size_query,
                                       metric,
                                       metricArg);
}

};  // namespace raft::sparse::neighbors
