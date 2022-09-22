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

#ifndef __KL_DIVERGENCE_H
#define __KL_DIVERGENCE_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/stats/detail/kl_divergence.cuh>

namespace raft {
namespace stats {

/**
 * @brief Function to calculate KL Divergence
 * <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">more info on KL
 * Divergence</a>
 *
 * @tparam DataT: Data type of the input array
 * @param modelPDF: the model array of probability density functions of type DataT
 * @param candidatePDF: the candidate array of probability density functions of type DataT
 * @param size: the size of the data points of type int
 * @param stream: the cudaStream object
 */
template <typename DataT>
DataT kl_divergence(const DataT* modelPDF, const DataT* candidatePDF, int size, cudaStream_t stream)
{
  return detail::kl_divergence(modelPDF, candidatePDF, size, stream);
}

/**
 * @brief Function to calculate KL Divergence
 * <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">more info on KL
 * Divergence</a>
 *
 * @tparam DataT: Data type of the input array
 * @tparam IdxType index type
 * @param handle the raft handle
 * @param modelPDF: the model array of probability density functions of type DataT
 * @param candidatePDF: the candidate array of probability density functions of type DataT
 */
template <typename DataT, typename IdxType>
DataT kl_divergence(const raft::handle_t& handle,
                    raft::device_vector_view<const DataT, IdxType> modelPDF,
                    raft::device_vector_view<const DataT, IdxType> candidatePDF)
{
  RAFT_EXPECTS(modelPDF.size() == candidatePDF.size(), "Size mismatch");
  RAFT_EXPECTS(modelPDF.is_exhaustive(), "modelPDF must be contiguous");
  RAFT_EXPECTS(candidatePDF.is_exhaustive(), "candidatePDF must be contiguous");
  return detail::kl_divergence(
    modelPDF.data_handle(), candidatePDF.data_handle(), modelPDF.extent(0), handle.get_stream());
}

};  // end namespace stats
};  // end namespace raft

#endif
