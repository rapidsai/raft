/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#ifndef __SILHOUETTE_SCORE_H
#define __SILHOUETTE_SCORE_H

#pragma once

#include <raft/stats/detail/batched/silhouette_score.cuh>
#include <raft/stats/detail/silhouette_score.cuh>

namespace raft {
    namespace stats {

/**
 * @brief main function that returns the average silhouette score for a given set of data and its
 * clusterings
 * @tparam DataT: type of the data samples
 * @tparam LabelT: type of the labels
 * @param handle: raft handle for managing expensive resources
 * @param X_in: pointer to the input Data samples array (nRows x nCols)
 * @param nRows: number of data samples
 * @param nCols: number of features
 * @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
 * @param nLabels: number of Labels
 * @param silhouette_scorePerSample: pointer to the array that is optionally taken in as input and
 * is populated with the silhouette score for every sample (1 x nRows)
 * @param stream: the cuda stream where to launch this kernel
 * @param metric: the numerical value that maps to the type of distance metric to be used in the
 * calculations
 */
        template <typename DataT, typename LabelT>
        DataT silhouette_score(
                const raft::handle_t& handle,
                DataT* X_in,
                int nRows,
                int nCols,
                LabelT* labels,
                int nLabels,
                DataT* silhouette_scorePerSample,
                cudaStream_t stream,
                raft::distance::DistanceType metric = raft::distance::DistanceType::L2Unexpanded)
        {
            return detail::silhouette_score(
                    handle, X_in, nRows, nCols, labels, nLabels, silhouette_scorePerSample, stream, metric);
        }

        template <typename value_t, typename value_idx, typename label_idx>
        value_t silhouette_score_batched(
                const raft::handle_t& handle,
                value_t* X,
                value_idx n_rows,
                value_idx n_cols,
                label_idx* y,
                label_idx n_labels,
                value_t* scores,
                value_idx chunk,
                raft::distance::DistanceType metric = raft::distance::DistanceType::L2Unexpanded)
        {
            return batched::detail::silhouette_score(
                    handle, X, n_rows, n_cols, y, n_labels, scores, chunk, metric);
        }

    };  // namespace stats
};  // namespace raft

#endif