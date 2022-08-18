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

#ifndef __CONTINGENCY_MATRIX_H
#define __CONTINGENCY_MATRIX_H

#pragma once

#include <raft/core/mdarray.hpp>
#include <raft/stats/detail/contingencyMatrix.cuh>

namespace raft {
namespace stats {

/**
 * @brief use this to allocate output matrix size
 * size of matrix = (maxLabel - minLabel + 1)^2 * sizeof(int)
 * @param groundTruth: device 1-d array for ground truth (num of rows)
 * @param nSamples: number of elements in input array
 * @param stream: cuda stream for execution
 * @param minLabel: [out] calculated min value in input array
 * @param maxLabel: [out] calculated max value in input array
 */
template <typename T>
void getInputClassCardinality(
  const T* groundTruth, const int nSamples, cudaStream_t stream, T& minLabel, T& maxLabel)
{
  detail::getInputClassCardinality(groundTruth, nSamples, stream, minLabel, maxLabel);
}

/**
 * @brief use this to allocate output matrix size
 * size of matrix = (maxLabel - minLabel + 1)^2 * sizeof(int)
 * @param handle: the raft handle.
 * @param groundTruth: device 1-d array for ground truth (num of rows)
 * @param nSamples: number of elements in input array
 * @param minLabel: [out] calculated min value in input array
 * @param maxLabel: [out] calculated max value in input array
 */
template <typename T, typename IdxType, typename LayoutPolicy, typename AccessorPolicy>
void getInputClassCardinality(
  const raft::handle_t& handle,
  raft::mdspan<const T, raft::vector_extent<IdxType>, LayoutPolicy, AccessorPolicy> groundTruth,
  const raft::host_scalar_view<T>& minLabel,
  const raft::host_scalar_view<T>& maxLabel)
{
  detail::getInputClassCardinality(groundTruth.data_handle(),
                                   groundTruth.extent(0),
                                   handle.get_stream(),
                                   *minLabel.data_handle(),
                                   *maxLabel.data_handle());
}

/**
 * @brief Calculate workspace size for running contingency matrix calculations
 * @tparam T label type
 * @tparam OutT output matrix type
 * @param nSamples: number of elements in input array
 * @param groundTruth: device 1-d array for ground truth (num of rows)
 * @param stream: cuda stream for execution
 * @param minLabel: Optional, min value in input array
 * @param maxLabel: Optional, max value in input array
 */
template <typename T, typename OutT = int>
size_t getContingencyMatrixWorkspaceSize(int nSamples,
                                         const T* groundTruth,
                                         cudaStream_t stream,
                                         T minLabel = std::numeric_limits<T>::max(),
                                         T maxLabel = std::numeric_limits<T>::max())
{
  return detail::getContingencyMatrixWorkspaceSize(
    nSamples, groundTruth, stream, minLabel, maxLabel);
}

/**
 * @brief Calculate workspace size for running contingency matrix calculations
 * @tparam T label type
 * @tparam OutT output matrix type
 * @param handle: the raft handle.
 * @param groundTruth: device 1-d array for ground truth (num of rows)
 * @param minLabel: Optional, min value in input array
 * @param maxLabel: Optional, max value in input array
 */
template <typename T,
          typename OutT = int,
          typename IdxType,
          typename LayoutPolicy,
          typename AccessorPolicy>
size_t getContingencyMatrixWorkspaceSize(
  const raft::handle_t& handle,
  raft::mdspan<const T, raft::vector_extent<IdxType>, LayoutPolicy, AccessorPolicy> groundTruth,
  T minLabel = std::numeric_limits<T>::max(),
  T maxLabel = std::numeric_limits<T>::max())
{
  return detail::getContingencyMatrixWorkspaceSize(
    groundTruth.extent(0), groundTruth.data_handle(), handle.get_stream(), minLabel, maxLabel);
}

/**
 * @brief contruct contingency matrix given input ground truth and prediction
 *        labels. Users should call function getInputClassCardinality to find
 *        and allocate memory for output. Similarly workspace requirements
 *        should be checked using function getContingencyMatrixWorkspaceSize
 * @tparam T label type
 * @tparam OutT output matrix type
 * @param groundTruth: device 1-d array for ground truth (num of rows)
 * @param predictedLabel: device 1-d array for prediction (num of columns)
 * @param nSamples: number of elements in input array
 * @param outMat: output buffer for contingecy matrix
 * @param stream: cuda stream for execution
 * @param workspace: Optional, workspace memory allocation
 * @param workspaceSize: Optional, size of workspace memory
 * @param minLabel: Optional, min value in input ground truth array
 * @param maxLabel: Optional, max value in input ground truth array
 */
template <typename T, typename OutT = int>
void contingencyMatrix(const T* groundTruth,
                       const T* predictedLabel,
                       int nSamples,
                       OutT* outMat,
                       cudaStream_t stream,
                       void* workspace      = nullptr,
                       size_t workspaceSize = 0,
                       T minLabel           = std::numeric_limits<T>::max(),
                       T maxLabel           = std::numeric_limits<T>::max())
{
  detail::contingencyMatrix<T, OutT>(groundTruth,
                                     predictedLabel,
                                     nSamples,
                                     outMat,
                                     stream,
                                     workspace,
                                     workspaceSize,
                                     minLabel,
                                     maxLabel);
}

/**
 * @brief contruct contingency matrix given input ground truth and prediction
 *        labels. Users should call function getInputClassCardinality to find
 *        and allocate memory for output. Similarly workspace requirements
 *        should be checked using function getContingencyMatrixWorkspaceSize
 * @tparam T label type
 * @tparam OutT output matrix type
 * @tparam IdxType Index type of matrix extent.
 * @tparam LayoutPolicy Layout type of the input data.
 * @tparam AccessorPolicy Accessor for the input and output, must be valid accessor on
 *                        device.
 * @param handle: the raft handle.
 * @param groundTruth: device 1-d array for ground truth (num of rows)
 * @param predictedLabel: device 1-d array for prediction (num of columns)
 * @param outMat: output buffer for contingecy matrix
 * @param workspace: Optional, workspace memory allocation
 * @param workspaceSize: Optional, size of workspace memory
 * @param minLabel: Optional, min value in input ground truth array
 * @param maxLabel: Optional, max value in input ground truth array
 */
template <typename T,
          typename OutT = int,
          typename IdxType,
          typename LayoutPolicy,
          typename AccessorPolicy>
void contingencyMatrix(
  const raft::handle_t& handle,
  raft::mdspan<const T, raft::vector_extent<IdxType>, LayoutPolicy, AccessorPolicy> groundTruth,
  raft::mdspan<const T, raft::vector_extent<IdxType>, LayoutPolicy, AccessorPolicy> predictedLabel,
  raft::mdspan<OutT, raft::matrix_extent<IdxType>, LayoutPolicy> outMat,
  void* workspace      = nullptr,
  size_t workspaceSize = 0,
  T minLabel           = std::numeric_limits<T>::max(),
  T maxLabel           = std::numeric_limits<T>::max())
{
  detail::contingencyMatrix<T, OutT>(groundTruth.data_handle(),
                                     predictedLabel.data_handle(),
                                     groundTruth.extent(0),
                                     outMat.data_handle(),
                                     handle.get_stream(),
                                     workspace,
                                     workspaceSize,
                                     minLabel,
                                     maxLabel);
}

};  // namespace stats
};  // namespace raft

#endif