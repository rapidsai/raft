/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/pca.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace raft::linalg {

/**
 * @defgroup pca PCA operations
 * @{
 */

/**
 * @brief perform fit operation for PCA. Generates eigenvectors, explained vars, singular vals, etc.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @param[in] handle raft::resources
 * @param[in] prms PCA parameters (n_components, algorithm, whiten, etc.)
 * @param[inout] input the data is fitted to PCA. Size n_rows x n_cols (col-major). Modified
 * temporarily during computation.
 * @param[out] components the principal components of the input data. Size n_components x n_cols
 * (col-major).
 * @param[out] explained_var explained variances (eigenvalues) of the principal components. Size
 * n_components.
 * @param[out] explained_var_ratio the ratio of the explained variance and total variance. Size
 * n_components.
 * @param[out] singular_vals singular values of the data. Size n_components.
 * @param[out] mu mean of all the features (all the columns in the data). Size n_cols.
 * @param[out] noise_vars variance of the noise. Scalar.
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
template <typename math_t, typename idx_t = std::size_t>
void pca_fit(raft::resources const& handle,
             const paramsPCA& prms,
             raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
             raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
             raft::device_vector_view<math_t, idx_t> explained_var,
             raft::device_vector_view<math_t, idx_t> explained_var_ratio,
             raft::device_vector_view<math_t, idx_t> singular_vals,
             raft::device_vector_view<math_t, idx_t> mu,
             raft::device_scalar_view<math_t> noise_vars,
             bool flip_signs_based_on_U = false)
{
  auto stream = resource::get_cuda_stream(handle);

  paramsPCA prms_with_dims = prms;
  prms_with_dims.n_rows    = static_cast<std::size_t>(input.extent(0));
  prms_with_dims.n_cols    = static_cast<std::size_t>(input.extent(1));

  detail::pcaFit(handle,
                 input.data_handle(),
                 components.data_handle(),
                 explained_var.data_handle(),
                 explained_var_ratio.data_handle(),
                 singular_vals.data_handle(),
                 mu.data_handle(),
                 noise_vars.data_handle(),
                 prms_with_dims,
                 stream,
                 flip_signs_based_on_U);
}

/**
 * @brief perform fit and transform operations for PCA. Generates transformed data,
 * eigenvectors, explained vars, singular vals, etc.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @param[in] handle raft::resources
 * @param[in] prms PCA parameters (n_components, algorithm, whiten, etc.)
 * @param[inout] input the data is fitted to PCA. Size n_rows x n_cols (col-major). Modified
 * temporarily during computation.
 * @param[out] trans_input the transformed data. Size n_rows x n_components (col-major).
 * @param[out] components the principal components of the input data. Size n_components x n_cols
 * (col-major).
 * @param[out] explained_var explained variances (eigenvalues) of the principal components. Size
 * n_components.
 * @param[out] explained_var_ratio the ratio of the explained variance and total variance. Size
 * n_components.
 * @param[out] singular_vals singular values of the data. Size n_components.
 * @param[out] mu mean of all the features (all the columns in the data). Size n_cols.
 * @param[out] noise_vars variance of the noise. Scalar.
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
template <typename math_t, typename idx_t = std::size_t>
void pca_fit_transform(raft::resources const& handle,
                       const paramsPCA& prms,
                       raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
                       raft::device_matrix_view<math_t, idx_t, raft::col_major> trans_input,
                       raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                       raft::device_vector_view<math_t, idx_t> explained_var,
                       raft::device_vector_view<math_t, idx_t> explained_var_ratio,
                       raft::device_vector_view<math_t, idx_t> singular_vals,
                       raft::device_vector_view<math_t, idx_t> mu,
                       raft::device_scalar_view<math_t> noise_vars,
                       bool flip_signs_based_on_U = false)
{
  auto stream = resource::get_cuda_stream(handle);

  paramsPCA prms_with_dims = prms;
  prms_with_dims.n_rows    = static_cast<std::size_t>(input.extent(0));
  prms_with_dims.n_cols    = static_cast<std::size_t>(input.extent(1));

  detail::pcaFitTransform(handle,
                          input.data_handle(),
                          trans_input.data_handle(),
                          components.data_handle(),
                          explained_var.data_handle(),
                          explained_var_ratio.data_handle(),
                          singular_vals.data_handle(),
                          mu.data_handle(),
                          noise_vars.data_handle(),
                          prms_with_dims,
                          stream,
                          flip_signs_based_on_U);
}

/**
 * @brief performs inverse transform operation for PCA. Transforms the transformed data back to
 * original data.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @param[in] handle raft::resources
 * @param[in] prms PCA parameters (n_components, algorithm, whiten, etc.)
 * @param[in] trans_input the transformed data. Size n_rows x n_components (col-major).
 * @param[in] components the principal components of the input data. Size n_components x n_cols
 * (col-major).
 * @param[in] singular_vals singular values of the data. Size n_components.
 * @param[in] mu mean of features (every column). Size n_cols.
 * @param[out] output the reconstructed data. Size n_rows x n_cols (col-major).
 */
template <typename math_t, typename idx_t = std::size_t>
void pca_inverse_transform(raft::resources const& handle,
                           const paramsPCA& prms,
                           raft::device_matrix_view<math_t, idx_t, raft::col_major> trans_input,
                           raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                           raft::device_vector_view<math_t, idx_t> singular_vals,
                           raft::device_vector_view<math_t, idx_t> mu,
                           raft::device_matrix_view<math_t, idx_t, raft::col_major> output)
{
  auto stream = resource::get_cuda_stream(handle);

  paramsPCA prms_with_dims = prms;
  prms_with_dims.n_rows    = static_cast<std::size_t>(output.extent(0));
  prms_with_dims.n_cols    = static_cast<std::size_t>(output.extent(1));

  detail::pcaInverseTransform(handle,
                              trans_input.data_handle(),
                              components.data_handle(),
                              singular_vals.data_handle(),
                              mu.data_handle(),
                              output.data_handle(),
                              prms_with_dims,
                              stream);
}

/**
 * @brief performs transform operation for PCA. Transforms the data to eigenspace.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @param[in] handle raft::resources
 * @param[in] prms PCA parameters (n_components, algorithm, whiten, etc.)
 * @param[inout] input the data to be transformed. Size n_rows x n_cols (col-major). Modified
 * temporarily during computation (mean-centered then restored).
 * @param[in] components principal components of the input data. Size n_components x n_cols
 * (col-major).
 * @param[in] singular_vals singular values of the data. Size n_components.
 * @param[in] mu mean value of the input data. Size n_cols.
 * @param[out] trans_input the transformed data. Size n_rows x n_components (col-major).
 */
template <typename math_t, typename idx_t = std::size_t>
void pca_transform(raft::resources const& handle,
                   const paramsPCA& prms,
                   raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
                   raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                   raft::device_vector_view<math_t, idx_t> singular_vals,
                   raft::device_vector_view<math_t, idx_t> mu,
                   raft::device_matrix_view<math_t, idx_t, raft::col_major> trans_input)
{
  auto stream = resource::get_cuda_stream(handle);

  paramsPCA prms_with_dims = prms;
  prms_with_dims.n_rows    = static_cast<std::size_t>(input.extent(0));
  prms_with_dims.n_cols    = static_cast<std::size_t>(input.extent(1));

  detail::pcaTransform(handle,
                       input.data_handle(),
                       components.data_handle(),
                       trans_input.data_handle(),
                       singular_vals.data_handle(),
                       mu.data_handle(),
                       prms_with_dims,
                       stream);
}

/** @} */  // end group pca

};  // end namespace raft::linalg
