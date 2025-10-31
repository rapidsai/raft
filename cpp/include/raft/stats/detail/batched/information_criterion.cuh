/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <raft/linalg/unary_op.cuh>
#include <raft/stats/stats_types.hpp>

#include <cmath>

namespace raft {
namespace stats {
namespace batched {
namespace detail {

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
void information_criterion(ScalarT* d_ic,
                           const ScalarT* d_loglikelihood,
                           IC_Type ic_type,
                           IdxT n_params,
                           IdxT batch_size,
                           IdxT n_samples,
                           cudaStream_t stream)
{
  ScalarT ic_base{};
  ScalarT N = static_cast<ScalarT>(n_params);
  ScalarT T = static_cast<ScalarT>(n_samples);
  switch (ic_type) {
    case AIC: ic_base = (ScalarT)2.0 * N; break;
    case AICc:
      ic_base = (ScalarT)2.0 * (N + (N * (N + (ScalarT)1.0)) / (T - N - (ScalarT)1.0));
      break;
    case BIC: ic_base = std::log(T) * N; break;
  }
  /* Compute information criterion from log-likelihood and base term */
  raft::linalg::unaryOp(
    d_ic,
    d_loglikelihood,
    batch_size,
    [=] __device__(ScalarT loglike) { return ic_base - (ScalarT)2.0 * loglike; },
    stream);
}

}  // namespace detail
}  // namespace batched
}  // namespace stats
}  // namespace raft
