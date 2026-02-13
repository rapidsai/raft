/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/pca.cuh"

namespace raft::linalg {

template <typename math_t, typename enum_solver = solver>
void truncCompExpVars(const raft::handle_t& handle,
                      math_t* in,
                      math_t* components,
                      math_t* explained_var,
                      math_t* explained_var_ratio,
                      math_t* noise_vars,
                      const paramsTSVDTemplate<enum_solver>& prms,
                      cudaStream_t stream)
{
  detail::truncCompExpVars(
    handle, in, components, explained_var, explained_var_ratio, noise_vars, prms, stream);
}

/**
 * @brief perform fit operation for the pca. Generates eigenvectors, explained vars, singular vals,
 * etc.
 * @param[in] handle: cuml handle object
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
void pcaFit(const raft::handle_t& handle,
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
}

/**
 * @brief perform fit and transform operations for the pca. Generates transformed data,
 * eigenvectors, explained vars, singular vals, etc.
 * @param[in] handle: cuml handle object
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
void pcaFitTransform(const raft::handle_t& handle,
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
  detail::pcaFitTransform(handle,
                          input,
                          trans_input,
                          components,
                          explained_var,
                          explained_var_ratio,
                          singular_vals,
                          mu,
                          noise_vars,
                          prms,
                          stream,
                          flip_signs_based_on_U);
}

// TODO: implement pcaGetCovariance function
template <typename math_t>
void pcaGetCovariance()
{
  detail::pcaGetCovariance<math_t>();
}

// TODO: implement pcaGetPrecision function
template <typename math_t>
void pcaGetPrecision()
{
  detail::pcaGetPrecision<math_t>();
}

/**
 * @brief performs inverse transform operation for the pca. Transforms the transformed data back to
 * original data.
 * @param[in] handle: the internal cuml handle object
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
void pcaInverseTransform(const raft::handle_t& handle,
                         math_t* trans_input,
                         math_t* components,
                         math_t* singular_vals,
                         math_t* mu,
                         math_t* input,
                         const paramsPCA& prms,
                         cudaStream_t stream)
{
  detail::pcaInverseTransform(
    handle, trans_input, components, singular_vals, mu, input, prms, stream);
}

// TODO: implement pcaScore function
template <typename math_t>
void pcaScore()
{
  detail::pcaScore<math_t>();
}

// TODO: implement pcaScoreSamples function
template <typename math_t>
void pcaScoreSamples()
{
  detail::pcaScoreSamples<math_t>();
}

/**
 * @brief performs transform operation for the pca. Transforms the data to eigenspace.
 * @param[in] handle: the internal cuml handle object
 * @param[in] input: the data is transformed. Size n_rows x n_components.
 * @param[in] components: principal components of the input data. Size n_cols * n_components.
 * @param[out] trans_input:  the transformed data. Size n_rows * n_components.
 * @param[in] singular_vals: singular values of the data. Size n_components * 1.
 * @param[in] mu: mean value of the input data
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void pcaTransform(const raft::handle_t& handle,
                  math_t* input,
                  math_t* components,
                  math_t* trans_input,
                  math_t* singular_vals,
                  math_t* mu,
                  const paramsPCA& prms,
                  cudaStream_t stream)
{
  detail::pcaTransform(handle, input, components, trans_input, singular_vals, mu, prms, stream);
}

};  // end namespace raft::linalg
