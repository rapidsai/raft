/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

/* Adapted from scikit-learn
 * https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/_samples_generator.py
 */

#ifndef __MAKE_REGRESSION_H
#define __MAKE_REGRESSION_H

#pragma once

#include "detail/make_regression.cuh"

#include <raft/core/mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <algorithm>
#include <optional>

namespace raft::random {

/**
 * @brief GPU-equivalent of sklearn.datasets.make_regression as documented at:
 * https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
 *
 * @tparam  DataT  Scalar type
 * @tparam  IdxT   Index type
 *
 * @param[in]   handle          RAFT handle
 * @param[out]  out             Row-major (samples, features) matrix to store
 *                              the problem data
 * @param[out]  values          Row-major (samples, targets) matrix to store
 *                              the values for the regression problem
 * @param[in]   n_rows          Number of samples
 * @param[in]   n_cols          Number of features
 * @param[in]   n_informative   Number of informative features (non-zero
 *                              coefficients)
 * @param[in]   stream          CUDA stream
 * @param[out]  coef            Row-major (features, targets) matrix to store
 *                              the coefficients used to generate the values
 *                              for the regression problem. If nullptr is
 *                              given, nothing will be written
 * @param[in]   n_targets       Number of targets (generated values per sample)
 * @param[in]   bias            A scalar that will be added to the values
 * @param[in]   effective_rank  The approximate rank of the data matrix (used
 *                              to create correlations in the data). -1 is the
 *                              code to use well-conditioned data
 * @param[in]   tail_strength   The relative importance of the fat noisy tail
 *                              of the singular values profile if
 *                              effective_rank is not -1
 * @param[in]   noise           Standard deviation of the Gaussian noise
 *                              applied to the output
 * @param[in]   shuffle         Shuffle the samples and the features
 * @param[in]   seed            Seed for the random number generator
 * @param[in]   type            Random generator type
 */
template <typename DataT, typename IdxT>
void make_regression(raft::resources const& handle,
                     DataT* out,
                     DataT* values,
                     IdxT n_rows,
                     IdxT n_cols,
                     IdxT n_informative,
                     cudaStream_t stream,
                     DataT* coef         = nullptr,
                     IdxT n_targets      = (IdxT)1,
                     DataT bias          = (DataT)0.0,
                     IdxT effective_rank = (IdxT)-1,
                     DataT tail_strength = (DataT)0.5,
                     DataT noise         = (DataT)0.0,
                     bool shuffle        = true,
                     uint64_t seed       = 0ULL,
                     GeneratorType type  = GenPC)
{
  detail::make_regression_caller(handle,
                                 out,
                                 values,
                                 n_rows,
                                 n_cols,
                                 n_informative,
                                 stream,
                                 coef,
                                 n_targets,
                                 bias,
                                 effective_rank,
                                 tail_strength,
                                 noise,
                                 shuffle,
                                 seed,
                                 type);
}

/**
 * @defgroup make_regression Generate Dataset for Regression Model
 * @{
 */

/**
 * @brief GPU-equivalent of sklearn.datasets.make_regression as documented at:
 * https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
 *
 * @tparam  DataT  Scalar type
 * @tparam  IdxT   Index type
 *
 * @param[in]   handle          RAFT handle
 * @param[out]  out             Row-major (samples, features) matrix to store
 *                              the problem data
 * @param[out]  values          Row-major (samples, targets) matrix to store
 *                              the values for the regression problem
 * @param[in]   n_informative   Number of informative features (non-zero
 *                              coefficients)
 * @param[out]  coef            If present, a row-major (features, targets) matrix
 *                              to store the coefficients used to generate the values
 *                              for the regression problem
 * @param[in]   bias            A scalar that will be added to the values
 * @param[in]   effective_rank  The approximate rank of the data matrix (used
 *                              to create correlations in the data). -1 is the
 *                              code to use well-conditioned data
 * @param[in]   tail_strength   The relative importance of the fat noisy tail
 *                              of the singular values profile if
 *                              effective_rank is not -1
 * @param[in]   noise           Standard deviation of the Gaussian noise
 *                              applied to the output
 * @param[in]   shuffle         Shuffle the samples and the features
 * @param[in]   seed            Seed for the random number generator
 * @param[in]   type            Random generator type
 */
template <typename DataT, typename IdxT>
void make_regression(raft::resources const& handle,
                     raft::device_matrix_view<DataT, IdxT, raft::row_major> out,
                     raft::device_matrix_view<DataT, IdxT, raft::row_major> values,
                     IdxT n_informative,
                     std::optional<raft::device_matrix_view<DataT, IdxT, raft::row_major>> coef,
                     DataT bias          = DataT{},
                     IdxT effective_rank = static_cast<IdxT>(-1),
                     DataT tail_strength = DataT{0.5},
                     DataT noise         = DataT{},
                     bool shuffle        = true,
                     uint64_t seed       = 0ULL,
                     GeneratorType type  = GenPC)
{
  const auto n_samples = out.extent(0);
  assert(values.extent(0) == n_samples);
  const auto n_features = out.extent(1);
  const auto n_targets  = values.extent(1);

  const bool have_coef = coef.has_value();
  if (have_coef) {
    const auto coef_ref = *coef;
    assert(coef_ref.extent(0) == n_features);
    assert(coef_ref.extent(1) == n_targets);
  }
  DataT* coef_ptr = have_coef ? (*coef).data_handle() : nullptr;

  detail::make_regression_caller(handle,
                                 out.data_handle(),
                                 values.data_handle(),
                                 n_samples,
                                 n_features,
                                 n_informative,
                                 resource::get_cuda_stream(handle),
                                 coef_ptr,
                                 n_targets,
                                 bias,
                                 effective_rank,
                                 tail_strength,
                                 noise,
                                 shuffle,
                                 seed,
                                 type);
}

/** @} */  // end group make_regression

}  // namespace raft::random

#endif
