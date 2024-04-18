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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/sparse/neighbors/brute_force.cuh>

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
template <typename value_idx = int, typename value_t = float, int TPB_X = 32>
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

};  // namespace raft::sparse::neighbors
