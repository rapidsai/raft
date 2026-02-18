/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_resources.hpp>
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

template <typename math_t>
void truncCompExpVars(raft::resources const& handle,
                      math_t* in,
                      math_t* components,
                      math_t* explained_var,
                      math_t* explained_var_ratio,
                      math_t* noise_vars,
                      const paramsTSVD& prms,
                      cudaStream_t stream)
{
  auto len = prms.n_cols * prms.n_cols;
  rmm::device_uvector<math_t> components_all(len, stream);
  rmm::device_uvector<math_t> explained_var_all(prms.n_cols, stream);
  rmm::device_uvector<math_t> explained_var_ratio_all(prms.n_cols, stream);

  detail::calEig<math_t>(handle, in, components_all.data(), explained_var_all.data(), prms, stream);
  raft::matrix::trunc_zero_origin(
    handle,
    raft::make_device_matrix_view<const math_t, std::size_t, raft::col_major>(
      components_all.data(), prms.n_cols, prms.n_cols),
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      components, prms.n_components, prms.n_cols));
  raft::matrix::ratio(handle,
                      raft::make_device_matrix_view<const math_t, std::size_t, raft::col_major>(
                        explained_var_all.data(), prms.n_cols, std::size_t(1)),
                      raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
                        explained_var_ratio_all.data(), prms.n_cols, std::size_t(1)));
  raft::matrix::trunc_zero_origin(
    handle,
    raft::make_device_matrix_view<const math_t, std::size_t, raft::col_major>(
      explained_var_all.data(), prms.n_cols, std::size_t(1)),
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      explained_var, prms.n_components, std::size_t(1)));
  raft::matrix::trunc_zero_origin(
    handle,
    raft::make_device_matrix_view<const math_t, std::size_t, raft::col_major>(
      explained_var_ratio_all.data(), prms.n_cols, std::size_t(1)),
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      explained_var_ratio, prms.n_components, std::size_t(1)));

  // Compute the scalar noise_vars defined as (pseudocode)
  // (n_components < min(n_cols, n_rows)) ? explained_var_all[n_components:].mean() : 0
  if (prms.n_components < prms.n_cols && prms.n_components < prms.n_rows) {
    raft::stats::mean<true>(noise_vars,
                            explained_var_all.data() + prms.n_components,
                            std::size_t{1},
                            prms.n_cols - prms.n_components,
                            false,
                            stream);
  } else {
    raft::matrix::fill(
      handle,
      raft::make_device_vector_view<math_t, std::size_t>(noise_vars, std::size_t(1)),
      math_t{0});
  }
}

/**
 * @brief perform fit operation for the pca. Generates eigenvectors, explained vars, singular vals,
 * etc.
 * @param[in] handle: raft::resources
 * @param[in] input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is
 * indicated in prms.
 * @param[out] components: the principal components of the input data. Size n_cols * n_components.
 * @param[out] explained_var: explained variances (eigenvalues) of the principal components. Size
 * n_components * 1.
 * @param[out] explained_var_ratio: the ratio of the explained variance and total variance. Size
 * n_components * 1.
 * @param[out] singular_vals: singular values of the data. Size n_components * 1
 * @param[out] mu: mean of all the features (all the columns in the data). Size n_cols * 1.
 * @param[out] noise_vars: variance of the noise. Size 1 * 1 (scalar).
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void pcaFit(raft::resources const& handle,
            math_t* input,
            math_t* components,
            math_t* explained_var,
            math_t* explained_var_ratio,
            math_t* singular_vals,
            math_t* mu,
            math_t* noise_vars,
            const paramsPCA& prms,
            cudaStream_t stream,
            bool flip_signs_based_on_U = false)
{
  auto cublas_handle = raft::resource::get_cublas_handle(handle);

  ASSERT(prms.n_cols > 1, "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");
  ASSERT(prms.n_components > 0,
         "Parameter n_components: number of components cannot be less than one");

  auto n_components = prms.n_components;
  if (n_components > prms.n_cols) n_components = prms.n_cols;

  raft::stats::mean<false>(mu, input, prms.n_cols, prms.n_rows, false, stream);

  auto len = prms.n_cols * prms.n_cols;
  rmm::device_uvector<math_t> cov(len, stream);

  raft::stats::cov<false>(
    handle, cov.data(), input, mu, prms.n_cols, prms.n_rows, true, true, stream);
  detail::truncCompExpVars(
    handle, cov.data(), components, explained_var, explained_var_ratio, noise_vars, prms, stream);

  math_t scalar = (prms.n_rows - 1);
  raft::matrix::weighted_sqrt(
    handle,
    raft::make_device_matrix_view<const math_t, std::size_t, raft::row_major>(
      explained_var, std::size_t(1), n_components),
    raft::make_device_matrix_view<math_t, std::size_t, raft::row_major>(
      singular_vals, std::size_t(1), n_components),
    raft::make_host_scalar_view(&scalar),
    true);

  raft::stats::meanAdd<false, true>(input, input, mu, prms.n_cols, prms.n_rows, stream);

  detail::signFlipComponents(handle,
                             input,
                             components,
                             prms.n_rows,
                             prms.n_cols,
                             prms.n_components,
                             stream,
                             true,
                             flip_signs_based_on_U);
}

/**
 * @brief performs transform operation for the pca. Transforms the data to eigenspace.
 * @param[in] handle: raft::resources
 * @param[in] input: the data is transformed. Size n_rows x n_components.
 * @param[in] components: principal components of the input data. Size n_cols * n_components.
 * @param[out] trans_input:  the transformed data. Size n_rows * n_components.
 * @param[in] singular_vals: singular values of the data. Size n_components * 1.
 * @param[in] mu: mean value of the input data
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void pcaTransform(raft::resources const& handle,
                  math_t* input,
                  math_t* components,
                  math_t* trans_input,
                  math_t* singular_vals,
                  math_t* mu,
                  const paramsPCA& prms,
                  cudaStream_t stream)
{
  ASSERT(prms.n_cols > 1, "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 0, "Parameter n_rows: number of rows cannot be less than one");
  ASSERT(prms.n_components > 0,
         "Parameter n_components: number of components cannot be less than one");

  auto components_len = prms.n_cols * prms.n_components;
  rmm::device_uvector<math_t> components_copy{components_len, stream};
  raft::copy(components_copy.data(), components, prms.n_cols * prms.n_components, stream);

  if (prms.whiten) {
    math_t scalar = math_t(sqrt(prms.n_rows - 1));
    raft::linalg::scalarMultiply(components_copy.data(),
                                 components_copy.data(),
                                 scalar,
                                 prms.n_cols * prms.n_components,
                                 stream);
    raft::linalg::binary_div_skip_zero<raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_device_matrix_view<math_t, std::size_t, raft::row_major>(
        components_copy.data(), prms.n_cols, prms.n_components),
      raft::make_device_vector_view<const math_t, std::size_t>(singular_vals, prms.n_components));
  }

  raft::stats::meanCenter<false, true>(input, input, mu, prms.n_cols, prms.n_rows, stream);
  detail::tsvdTransform(handle, input, components_copy.data(), trans_input, prms, stream);
  raft::stats::meanAdd<false, true>(input, input, mu, prms.n_cols, prms.n_rows, stream);
}

/**
 * @brief performs inverse transform operation for the pca. Transforms the transformed data back to
 * original data.
 * @param[in] handle: raft::resources
 * @param[in] trans_input: the data is fitted to PCA. Size n_rows x n_components.
 * @param[in] components: transpose of the principal components of the input data. Size n_components
 * * n_cols.
 * @param[in] singular_vals: singular values of the data. Size n_components * 1
 * @param[in] mu: mean of features (every column).
 * @param[out] input: the data is fitted to PCA. Size n_rows x n_cols.
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void pcaInverseTransform(raft::resources const& handle,
                         math_t* trans_input,
                         math_t* components,
                         math_t* singular_vals,
                         math_t* mu,
                         math_t* input,
                         const paramsPCA& prms,
                         cudaStream_t stream)
{
  ASSERT(prms.n_cols > 1, "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 0, "Parameter n_rows: number of rows cannot be less than one");
  ASSERT(prms.n_components > 0,
         "Parameter n_components: number of components cannot be less than one");

  auto components_len = prms.n_cols * prms.n_components;
  rmm::device_uvector<math_t> components_copy{components_len, stream};
  raft::copy(components_copy.data(), components, prms.n_cols * prms.n_components, stream);

  if (prms.whiten) {
    math_t sqrt_n_samples = sqrt(prms.n_rows - 1);
    math_t scalar         = prms.n_rows - 1 > 0 ? math_t(1 / sqrt_n_samples) : 0;
    raft::linalg::scalarMultiply(components_copy.data(),
                                 components_copy.data(),
                                 scalar,
                                 prms.n_cols * prms.n_components,
                                 stream);
    raft::linalg::binary_mult_skip_zero<raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_device_matrix_view<math_t, std::size_t, raft::row_major>(
        components_copy.data(), prms.n_cols, prms.n_components),
      raft::make_device_vector_view<const math_t, std::size_t>(singular_vals, prms.n_components));
  }

  detail::tsvdInverseTransform(handle, trans_input, components_copy.data(), input, prms, stream);
  raft::stats::meanAdd<false, true>(input, input, mu, prms.n_cols, prms.n_rows, stream);
}

/**
 * @brief perform fit and transform operations for the pca. Generates transformed data,
 * eigenvectors, explained vars, singular vals, etc.
 * @param[in] handle: raft::resources
 * @param[in] input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is
 * indicated in prms.
 * @param[out] trans_input: the transformed data. Size n_rows * n_components.
 * @param[out] components: the principal components of the input data. Size n_cols * n_components.
 * @param[out] explained_var: explained variances (eigenvalues) of the principal components. Size
 * n_components * 1.
 * @param[out] explained_var_ratio: the ratio of the explained variance and total variance. Size
 * n_components * 1.
 * @param[out] singular_vals: singular values of the data. Size n_components * 1
 * @param[out] mu: mean of all the features (all the columns in the data). Size n_cols * 1.
 * @param[out] noise_vars: variance of the noise. Size 1 * 1 (scalar).
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void pcaFitTransform(raft::resources const& handle,
                     math_t* input,
                     math_t* trans_input,
                     math_t* components,
                     math_t* explained_var,
                     math_t* explained_var_ratio,
                     math_t* singular_vals,
                     math_t* mu,
                     math_t* noise_vars,
                     const paramsPCA& prms,
                     cudaStream_t stream,
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
                 stream,
                 flip_signs_based_on_U);
  detail::pcaTransform(handle, input, components, trans_input, singular_vals, mu, prms, stream);
}

};  // end namespace raft::linalg::detail
