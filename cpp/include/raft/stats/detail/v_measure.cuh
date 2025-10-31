/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file v_measure.cuh
 */

#include <raft/stats/homogeneity_score.cuh>

namespace raft {
namespace stats {
namespace detail {

/**
 * @brief Function to calculate the v-measure between two clusters
 *
 * @param truthClusterArray: the array of truth classes of type T
 * @param predClusterArray: the array of predicted classes of type T
 * @param size: the size of the data points of type int
 * @param lowerLabelRange: the lower bound of the range of labels
 * @param upperLabelRange: the upper bound of the range of labels
 * @param stream: the cudaStream object
 * @param beta: v_measure parameter
 */
template <typename T>
double v_measure(const T* truthClusterArray,
                 const T* predClusterArray,
                 int size,
                 T lowerLabelRange,
                 T upperLabelRange,
                 cudaStream_t stream,
                 double beta = 1.0)
{
  double computedHomogeity, computedCompleteness, computedVMeasure;

  computedHomogeity = raft::stats::homogeneity_score(
    truthClusterArray, predClusterArray, size, lowerLabelRange, upperLabelRange, stream);
  computedCompleteness = raft::stats::homogeneity_score(
    predClusterArray, truthClusterArray, size, lowerLabelRange, upperLabelRange, stream);

  if (computedCompleteness + computedHomogeity == 0.0)
    computedVMeasure = 0.0;
  else
    computedVMeasure = ((1 + beta) * computedHomogeity * computedCompleteness /
                        (beta * computedHomogeity + computedCompleteness));

  return computedVMeasure;
}

};  // end namespace detail
};  // end namespace stats
};  // end namespace raft
