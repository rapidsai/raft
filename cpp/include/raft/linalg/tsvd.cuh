/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/tsvd.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace raft::linalg {

/**
 * @defgroup tsvd Truncated SVD operations
 * @{
 */

/**
 * @brief perform fit operation for tSVD. Generates eigenvectors, singular vals, etc.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @param[in] handle raft::resources
 * @param[in] prms data structure that includes all the parameters from input size to algorithm.
 * @param[inout] input the data is fitted to tSVD. Size n_rows x n_cols (col-major).
 * @param[out] components the principal components of the input data. Size n_components x n_cols
 * (col-major).
 * @param[out] singular_vals singular values of the data. Size n_components.
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
template <typename math_t, typename idx_t>
void tsvd_fit(raft::resources const& handle,
              const paramsTSVD& prms,
              raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
              raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
              raft::device_vector_view<math_t, idx_t> singular_vals,
              bool flip_signs_based_on_U = false)
{
  auto stream = resource::get_cuda_stream(handle);

  paramsTSVD prms_with_dims = prms;
  prms_with_dims.n_rows     = static_cast<std::size_t>(input.extent(0));
  prms_with_dims.n_cols     = static_cast<std::size_t>(input.extent(1));

  detail::tsvdFit(handle,
                  input.data_handle(),
                  components.data_handle(),
                  singular_vals.data_handle(),
                  prms_with_dims,
                  stream,
                  flip_signs_based_on_U);
}

/**
 * @brief performs fit and transform operations for tSVD. Generates transformed data,
 * eigenvectors, explained vars, singular vals, etc.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @param[in] handle raft::resources
 * @param[in] prms data structure that includes all the parameters from input size to algorithm.
 * @param[inout] input the data is fitted to tSVD. Size n_rows x n_cols (col-major).
 * @param[out] trans_input the transformed data. Size n_rows x n_components (col-major).
 * @param[out] components the principal components of the input data. Size n_components x n_cols
 * (col-major).
 * @param[out] explained_var explained variances (eigenvalues) of the principal components. Size
 * n_components.
 * @param[out] explained_var_ratio the ratio of the explained variance and total variance. Size
 * n_components.
 * @param[out] singular_vals singular values of the data. Size n_components.
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
template <typename math_t, typename idx_t>
void tsvd_fit_transform(raft::resources const& handle,
                        const paramsTSVD& prms,
                        raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
                        raft::device_matrix_view<math_t, idx_t, raft::col_major> trans_input,
                        raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                        raft::device_vector_view<math_t, idx_t> explained_var,
                        raft::device_vector_view<math_t, idx_t> explained_var_ratio,
                        raft::device_vector_view<math_t, idx_t> singular_vals,
                        bool flip_signs_based_on_U = false)
{
  auto stream = resource::get_cuda_stream(handle);

  paramsTSVD prms_with_dims = prms;
  prms_with_dims.n_rows     = static_cast<std::size_t>(input.extent(0));
  prms_with_dims.n_cols     = static_cast<std::size_t>(input.extent(1));

  detail::tsvdFitTransform(handle,
                           input.data_handle(),
                           trans_input.data_handle(),
                           components.data_handle(),
                           explained_var.data_handle(),
                           explained_var_ratio.data_handle(),
                           singular_vals.data_handle(),
                           prms_with_dims,
                           stream,
                           flip_signs_based_on_U);
}

/**
 * @brief performs transform operation for tSVD. Transforms the data to eigenspace.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @param[in] handle raft::resources
 * @param[in] prms data structure that includes all the parameters from input size to algorithm.
 * @param[in] input the data to be transformed. Size n_rows x n_cols (col-major).
 * @param[in] components principal components of the input data. Size n_components x n_cols
 * (col-major).
 * @param[out] trans_input output that is transformed version of input. Size n_rows x n_components
 * (col-major).
 */
template <typename math_t, typename idx_t>
void tsvd_transform(raft::resources const& handle,
                    const paramsTSVD& prms,
                    raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
                    raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                    raft::device_matrix_view<math_t, idx_t, raft::col_major> trans_input)
{
  auto stream = resource::get_cuda_stream(handle);

  paramsTSVD prms_with_dims = prms;
  prms_with_dims.n_rows     = static_cast<std::size_t>(input.extent(0));
  prms_with_dims.n_cols     = static_cast<std::size_t>(input.extent(1));

  detail::tsvdTransform(handle,
                        input.data_handle(),
                        components.data_handle(),
                        trans_input.data_handle(),
                        prms_with_dims,
                        stream);
}

/**
 * @brief performs inverse transform operation for tSVD. Transforms the transformed data back to
 * original data.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @param[in] handle raft::resources
 * @param[in] prms data structure that includes all the parameters from input size to algorithm.
 * @param[in] trans_input the transformed data. Size n_rows x n_components (col-major).
 * @param[in] components transpose of the principal components. Size n_components x n_cols
 * (col-major).
 * @param[out] output the reconstructed data. Size n_rows x n_cols (col-major).
 */
template <typename math_t, typename idx_t>
void tsvd_inverse_transform(raft::resources const& handle,
                            const paramsTSVD& prms,
                            raft::device_matrix_view<math_t, idx_t, raft::col_major> trans_input,
                            raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                            raft::device_matrix_view<math_t, idx_t, raft::col_major> output)
{
  auto stream = resource::get_cuda_stream(handle);

  paramsTSVD prms_with_dims = prms;
  prms_with_dims.n_rows     = static_cast<std::size_t>(output.extent(0));
  prms_with_dims.n_cols     = static_cast<std::size_t>(output.extent(1));

  detail::tsvdInverseTransform(handle,
                               trans_input.data_handle(),
                               components.data_handle(),
                               output.data_handle(),
                               prms_with_dims,
                               stream);
}

/**
 * @brief Eigendecomposition helper for tSVD/PCA. Computes eigenvectors and eigenvalues
 * of a symmetric matrix using either divide-and-conquer or Jacobi method.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam enum_solver solver enum type
 * @param[in] handle raft::resources
 * @param[in] prms tSVD parameters (controls algorithm, tolerance, iterations)
 * @param[inout] in symmetric input matrix [n_cols x n_cols] (col-major). Overwritten.
 * @param[out] components eigenvectors [n_cols x n_cols] (col-major)
 * @param[out] explained_var eigenvalues [n_cols]
 */
template <typename math_t, typename idx_t, typename enum_solver = solver>
void cal_eig(raft::resources const& handle,
             const paramsTSVDTemplate<enum_solver>& prms,
             raft::device_matrix_view<math_t, idx_t, raft::col_major> in,
             raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
             raft::device_vector_view<math_t, idx_t> explained_var)
{
  auto stream = resource::get_cuda_stream(handle);

  paramsTSVDTemplate<enum_solver> prms_with_dims = prms;
  prms_with_dims.n_rows                          = static_cast<std::size_t>(in.extent(0));
  prms_with_dims.n_cols                          = static_cast<std::size_t>(in.extent(1));

  detail::calEig(handle,
                 in.data_handle(),
                 components.data_handle(),
                 explained_var.data_handle(),
                 prms_with_dims,
                 stream);
}

/**
 * @brief Sign flip for PCA and tSVD. Stabilizes the sign of column-major eigenvectors.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @param[in] handle raft::resources
 * @param[in] input input data matrix [n_samples x n_features] (col-major)
 * @param[inout] components components matrix [n_components x n_features] (col-major)
 * @param[in] center whether to mean-center input before computing signs
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
template <typename math_t, typename idx_t>
void sign_flip_components(raft::resources const& handle,
                          raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
                          raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                          bool center,
                          bool flip_signs_based_on_U = false)
{
  auto stream       = resource::get_cuda_stream(handle);
  auto n_samples    = static_cast<std::size_t>(input.extent(0));
  auto n_features   = static_cast<std::size_t>(input.extent(1));
  auto n_components = static_cast<std::size_t>(components.extent(0));

  detail::signFlipComponents(handle,
                             input.data_handle(),
                             components.data_handle(),
                             n_samples,
                             n_features,
                             n_components,
                             stream,
                             center,
                             flip_signs_based_on_U);
}

/** @} */  // end group tsvd

};  // end namespace raft::linalg
