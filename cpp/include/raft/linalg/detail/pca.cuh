/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/pca_types.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/linalg/tsvd.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/ratio.cuh>
#include <raft/matrix/sqrt.cuh>
#include <raft/stats/cov.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

namespace raft::linalg::detail {

template <typename math_t, typename idx_t>
void truncCompExpVars(raft::resources const& handle,
                      raft::device_matrix_view<math_t, idx_t, raft::col_major> in,
                      raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                      raft::device_vector_view<math_t, idx_t> explained_var,
                      raft::device_vector_view<math_t, idx_t> explained_var_ratio,
                      raft::device_scalar_view<math_t, idx_t> noise_vars,
                      const paramsTSVD& prms)
{
  auto stream = resource::get_cuda_stream(handle);

  auto n_cols       = in.extent(0);
  auto n_components = components.extent(0);

  auto len = static_cast<std::size_t>(n_cols * n_cols);
  rmm::device_uvector<math_t> components_all(len, stream);
  rmm::device_uvector<math_t> explained_var_all(static_cast<std::size_t>(n_cols), stream);
  rmm::device_uvector<math_t> explained_var_ratio_all(static_cast<std::size_t>(n_cols), stream);

  detail::calEig<math_t, idx_t>(
    handle,
    in,
    raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
      components_all.data(), n_cols, n_cols),
    raft::make_device_vector_view<math_t, idx_t>(explained_var_all.data(), n_cols),
    prms);
  raft::matrix::trunc_zero_origin(
    handle,
    raft::make_device_matrix_view<const math_t, idx_t, raft::col_major>(
      components_all.data(), n_cols, n_cols),
    raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
      components.data_handle(), n_components, n_cols));
  raft::matrix::ratio(handle,
                      raft::make_device_matrix_view<const math_t, idx_t, raft::col_major>(
                        explained_var_all.data(), n_cols, idx_t(1)),
                      raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
                        explained_var_ratio_all.data(), n_cols, idx_t(1)));
  raft::matrix::trunc_zero_origin(
    handle,
    raft::make_device_matrix_view<const math_t, idx_t, raft::col_major>(
      explained_var_all.data(), n_cols, idx_t(1)),
    raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
      explained_var.data_handle(), n_components, idx_t(1)));
  raft::matrix::trunc_zero_origin(
    handle,
    raft::make_device_matrix_view<const math_t, idx_t, raft::col_major>(
      explained_var_ratio_all.data(), n_cols, idx_t(1)),
    raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
      explained_var_ratio.data_handle(), n_components, idx_t(1)));

  if (static_cast<std::size_t>(n_components) < static_cast<std::size_t>(n_cols) &&
      static_cast<std::size_t>(n_components) < prms.n_rows) {
    raft::stats::mean<true>(noise_vars.data_handle(),
                            explained_var_all.data() + static_cast<std::size_t>(n_components),
                            std::size_t{1},
                            static_cast<std::size_t>(n_cols - n_components),
                            false,
                            stream);
  } else {
    raft::matrix::fill(
      handle,
      raft::make_device_vector_view<math_t, idx_t>(noise_vars.data_handle(), idx_t(1)),
      math_t{0});
  }
}

/**
 * @brief perform fit operation for the pca. Generates eigenvectors, explained vars, singular vals,
 * etc.
 * @param[in] handle: raft::resources
 * @param[inout] input: the data is fitted to PCA. Size n_rows x n_cols (col-major).
 * @param[out] components: the principal components. Size n_components x n_cols (col-major).
 * @param[out] explained_var: explained variances. Size n_components.
 * @param[out] explained_var_ratio: ratio of explained to total variance. Size n_components.
 * @param[out] singular_vals: singular values. Size n_components.
 * @param[out] mu: mean of all features. Size n_cols.
 * @param[out] noise_vars: noise variance scalar.
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
template <typename math_t, typename idx_t>
void pcaFit(raft::resources const& handle,
            raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
            raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
            raft::device_vector_view<math_t, idx_t> explained_var,
            raft::device_vector_view<math_t, idx_t> explained_var_ratio,
            raft::device_vector_view<math_t, idx_t> singular_vals,
            raft::device_vector_view<math_t, idx_t> mu,
            raft::device_scalar_view<math_t, idx_t> noise_vars,
            const paramsPCA& prms,
            bool flip_signs_based_on_U = false)
{
  auto stream        = resource::get_cuda_stream(handle);
  auto cublas_handle = raft::resource::get_cublas_handle(handle);

  auto n_rows = input.extent(0);
  auto n_cols = input.extent(1);

  ASSERT(n_cols > 1, "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");
  ASSERT(prms.n_components > 0,
         "Parameter n_components: number of components cannot be less than one");

  auto n_components = static_cast<idx_t>(prms.n_components);
  if (n_components > n_cols) n_components = n_cols;

  raft::stats::mean<false>(mu.data_handle(), input.data_handle(), n_cols, n_rows, false, stream);

  auto len = static_cast<std::size_t>(n_cols * n_cols);
  rmm::device_uvector<math_t> cov(len, stream);

  raft::stats::cov<false>(
    handle, cov.data(), input.data_handle(), mu.data_handle(), n_cols, n_rows, true, true, stream);

  paramsPCA prms_with_rows = prms;
  prms_with_rows.n_rows    = static_cast<std::size_t>(n_rows);
  prms_with_rows.n_cols    = static_cast<std::size_t>(n_cols);

  detail::truncCompExpVars(
    handle,
    raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(cov.data(), n_cols, n_cols),
    components,
    explained_var,
    explained_var_ratio,
    noise_vars,
    prms_with_rows);

  math_t scalar = (n_rows - 1);
  raft::matrix::weighted_sqrt(handle,
                              raft::make_device_matrix_view<const math_t, idx_t, raft::row_major>(
                                explained_var.data_handle(), idx_t(1), n_components),
                              raft::make_device_matrix_view<math_t, idx_t, raft::row_major>(
                                singular_vals.data_handle(), idx_t(1), n_components),
                              raft::make_host_scalar_view(&scalar),
                              true);

  raft::stats::meanAdd<false, true>(
    input.data_handle(), input.data_handle(), mu.data_handle(), n_cols, n_rows, stream);

  detail::signFlipComponents(handle, input, components, true, flip_signs_based_on_U);
}

/**
 * @brief performs transform operation for the pca. Transforms the data to eigenspace.
 * @param[in] handle: raft::resources
 * @param[inout] input: the data to transform. Size n_rows x n_cols (col-major). Modified
 * temporarily (mean-centered then restored).
 * @param[in] components: principal components. Size n_components x n_cols (col-major).
 * @param[out] trans_input: the transformed data. Size n_rows x n_components (col-major).
 * @param[in] singular_vals: singular values. Size n_components.
 * @param[in] mu: mean of features. Size n_cols.
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 */
template <typename math_t, typename idx_t>
void pcaTransform(raft::resources const& handle,
                  raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
                  raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                  raft::device_matrix_view<math_t, idx_t, raft::col_major> trans_input,
                  raft::device_vector_view<math_t, idx_t> singular_vals,
                  raft::device_vector_view<math_t, idx_t> mu,
                  const paramsPCA& prms)
{
  auto stream = resource::get_cuda_stream(handle);

  auto n_rows       = input.extent(0);
  auto n_cols       = input.extent(1);
  auto n_components = components.extent(0);

  ASSERT(n_cols > 1, "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(n_rows > 0, "Parameter n_rows: number of rows cannot be less than one");
  ASSERT(n_components > 0, "Parameter n_components: number of components cannot be less than one");

  auto components_len = static_cast<std::size_t>(n_cols * n_components);
  rmm::device_uvector<math_t> components_copy{components_len, stream};
  raft::copy(components_copy.data(), components.data_handle(), components_len, stream);

  if (prms.whiten) {
    math_t scalar = math_t(sqrt(n_rows - 1));
    raft::linalg::scalarMultiply(
      components_copy.data(), components_copy.data(), scalar, components_len, stream);
    raft::linalg::binary_div_skip_zero<raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_device_matrix_view<math_t, idx_t, raft::row_major>(
        components_copy.data(), n_cols, n_components),
      raft::make_device_vector_view<const math_t, idx_t>(singular_vals.data_handle(),
                                                         n_components));
  }

  raft::stats::meanCenter<false, true>(
    input.data_handle(), input.data_handle(), mu.data_handle(), n_cols, n_rows, stream);
  detail::tsvdTransform(handle,
                        input,
                        raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
                          components_copy.data(), n_components, n_cols),
                        trans_input,
                        prms);
  raft::stats::meanAdd<false, true>(
    input.data_handle(), input.data_handle(), mu.data_handle(), n_cols, n_rows, stream);
}

/**
 * @brief performs inverse transform operation for the pca. Transforms the transformed data back to
 * original data.
 * @param[in] handle: raft::resources
 * @param[in] trans_input: the transformed data. Size n_rows x n_components (col-major).
 * @param[in] components: principal components. Size n_components x n_cols (col-major).
 * @param[in] singular_vals: singular values. Size n_components.
 * @param[in] mu: mean of features. Size n_cols.
 * @param[out] input: the reconstructed data. Size n_rows x n_cols (col-major).
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 */
template <typename math_t, typename idx_t>
void pcaInverseTransform(raft::resources const& handle,
                         raft::device_matrix_view<math_t, idx_t, raft::col_major> trans_input,
                         raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                         raft::device_vector_view<math_t, idx_t> singular_vals,
                         raft::device_vector_view<math_t, idx_t> mu,
                         raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
                         const paramsPCA& prms)
{
  auto stream = resource::get_cuda_stream(handle);

  auto n_rows       = input.extent(0);
  auto n_cols       = input.extent(1);
  auto n_components = components.extent(0);

  ASSERT(n_cols > 1, "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(n_rows > 0, "Parameter n_rows: number of rows cannot be less than one");
  ASSERT(n_components > 0, "Parameter n_components: number of components cannot be less than one");

  auto components_len = static_cast<std::size_t>(n_cols * n_components);
  rmm::device_uvector<math_t> components_copy{components_len, stream};
  raft::copy(components_copy.data(), components.data_handle(), components_len, stream);

  if (prms.whiten) {
    math_t sqrt_n_samples = sqrt(n_rows - 1);
    math_t scalar         = n_rows - 1 > 0 ? math_t(1 / sqrt_n_samples) : 0;
    raft::linalg::scalarMultiply(
      components_copy.data(), components_copy.data(), scalar, components_len, stream);
    raft::linalg::binary_mult_skip_zero<raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_device_matrix_view<math_t, idx_t, raft::row_major>(
        components_copy.data(), n_cols, n_components),
      raft::make_device_vector_view<const math_t, idx_t>(singular_vals.data_handle(),
                                                         n_components));
  }

  detail::tsvdInverseTransform(handle,
                               trans_input,
                               raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
                                 components_copy.data(), n_components, n_cols),
                               input,
                               prms);
  raft::stats::meanAdd<false, true>(
    input.data_handle(), input.data_handle(), mu.data_handle(), n_cols, n_rows, stream);
}

/**
 * @brief perform fit and transform operations for the pca. Generates transformed data,
 * eigenvectors, explained vars, singular vals, etc.
 * @param[in] handle: raft::resources
 * @param[inout] input: the data is fitted to PCA. Size n_rows x n_cols (col-major).
 * @param[out] trans_input: the transformed data. Size n_rows x n_components (col-major).
 * @param[out] components: the principal components. Size n_components x n_cols (col-major).
 * @param[out] explained_var: explained variances. Size n_components.
 * @param[out] explained_var_ratio: ratio of explained to total variance. Size n_components.
 * @param[out] singular_vals: singular values. Size n_components.
 * @param[out] mu: mean of all features. Size n_cols.
 * @param[out] noise_vars: noise variance scalar.
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
template <typename math_t, typename idx_t>
void pcaFitTransform(raft::resources const& handle,
                     raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
                     raft::device_matrix_view<math_t, idx_t, raft::col_major> trans_input,
                     raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                     raft::device_vector_view<math_t, idx_t> explained_var,
                     raft::device_vector_view<math_t, idx_t> explained_var_ratio,
                     raft::device_vector_view<math_t, idx_t> singular_vals,
                     raft::device_vector_view<math_t, idx_t> mu,
                     raft::device_scalar_view<math_t, idx_t> noise_vars,
                     const paramsPCA& prms,
                     bool flip_signs_based_on_U = false)
{
  detail::pcaFit(handle,
                 input,
                 components,
                 explained_var,
                 explained_var_ratio,
                 singular_vals,
                 mu,
                 noise_vars,
                 prms,
                 flip_signs_based_on_U);
  detail::pcaTransform(handle, input, components, trans_input, singular_vals, mu, prms);
}

};  // end namespace raft::linalg::detail
