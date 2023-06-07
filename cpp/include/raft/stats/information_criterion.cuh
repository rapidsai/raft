/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
 * @file information_criterion.cuh
 * @brief These information criteria are used to evaluate the quality of models
 *        by balancing the quality of the fit and the number of parameters.
 *
 * See:
 *  - AIC: https://en.wikipedia.org/wiki/Akaike_information_criterion
 *  - AICc: https://en.wikipedia.org/wiki/Akaike_information_criterion#AICc
 *  - BIC: https://en.wikipedia.org/wiki/Bayesian_information_criterion
 */

#ifndef __INFORMATION_CRIT_H
#define __INFORMATION_CRIT_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/detail/batched/information_criterion.cuh>
#include <raft/stats/stats_types.hpp>

namespace raft {
namespace stats {

/**
 * Compute the given type of information criterion
 *
 * @note: it is safe to do the computation in-place (i.e give same pointer
 *        as input and output)
 *
 * @param[out] d_ic             Information criterion to be returned for each
 *                              series (device)
 * @param[in]  d_loglikelihood  Log-likelihood for each series (device)
 * @param[in]  ic_type          Type of criterion to compute. See IC_Type
 * @param[in]  n_params         Number of parameters in the model
 * @param[in]  batch_size       Number of series in the batch
 * @param[in]  n_samples        Number of samples in each series
 * @param[in]  stream           CUDA stream
 */
template <typename ScalarT, typename IdxT>
void information_criterion_batched(ScalarT* d_ic,
                                   const ScalarT* d_loglikelihood,
                                   IC_Type ic_type,
                                   IdxT n_params,
                                   IdxT batch_size,
                                   IdxT n_samples,
                                   cudaStream_t stream)
{
  batched::detail::information_criterion(
    d_ic, d_loglikelihood, ic_type, n_params, batch_size, n_samples, stream);
}

/**
 * @defgroup stats_information_criterion Information Criterion
 * @{
 */

/**
 * Compute the given type of information criterion
 *
 * @note: it is safe to do the computation in-place (i.e give same pointer
 *        as input and output)
 * See:
 *  - AIC: https://en.wikipedia.org/wiki/Akaike_information_criterion
 *  - AICc: https://en.wikipedia.org/wiki/Akaike_information_criterion#AICc
 *  - BIC: https://en.wikipedia.org/wiki/Bayesian_information_criterion
 *
 * @tparam value_t data type
 * @tparam idx_t index type
 * @param[in]  handle           the raft handle
 * @param[in]  d_loglikelihood  Log-likelihood for each series (device) length: batch_size
 * @param[out] d_ic             Information criterion to be returned for each
 *                              series (device) length: batch_size
 * @param[in]  ic_type          Type of criterion to compute. See IC_Type
 * @param[in]  n_params         Number of parameters in the model
 * @param[in]  n_samples        Number of samples in each series
 */
template <typename value_t, typename idx_t>
void information_criterion_batched(raft::resources const& handle,
                                   raft::device_vector_view<const value_t, idx_t> d_loglikelihood,
                                   raft::device_vector_view<value_t, idx_t> d_ic,
                                   IC_Type ic_type,
                                   idx_t n_params,
                                   idx_t n_samples)
{
  RAFT_EXPECTS(d_ic.size() == d_loglikelihood.size(), "Size mismatch");
  RAFT_EXPECTS(d_ic.is_exhaustive(), "d_ic must be contiguous");
  RAFT_EXPECTS(d_loglikelihood.is_exhaustive(), "d_loglikelihood must be contiguous");
  batched::detail::information_criterion(d_ic.data_handle(),
                                         d_loglikelihood.data_handle(),
                                         ic_type,
                                         n_params,
                                         d_ic.extent(0),
                                         n_samples,
                                         resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_information_criterion

}  // namespace stats
}  // namespace raft
#endif
