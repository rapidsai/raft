/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/tsvd.cuh"

namespace raft::linalg {

template <typename math_t>
void calCompExpVarsSvd(const raft::handle_t& handle,
                       math_t* in,
                       math_t* components,
                       math_t* singular_vals,
                       math_t* explained_vars,
                       math_t* explained_var_ratio,
                       const paramsTSVD& prms,
                       cudaStream_t stream)
{
  detail::calCompExpVarsSvd(
    handle, in, components, singular_vals, explained_vars, explained_var_ratio, prms, stream);
}

template <typename math_t, typename enum_solver = solver>
void calEig(const raft::handle_t& handle,
            math_t* in,
            math_t* components,
            math_t* explained_var,
            const paramsTSVDTemplate<enum_solver>& prms,
            cudaStream_t stream)
{
  detail::calEig(handle, in, components, explained_var, prms, stream);
}

/**
 * @defgroup sign flip for PCA and tSVD. This is used to stabilize the sign of column major eigen
 * vectors
 * @param handle: resource handle
 * @param components: components matrix, used to determine the sign of max absolute value
 * @param input: input data
 * @param n_rows: number of rows of components matrix
 * @param n_cols: number of columns of components matrix
 * @param n_samples: number of samples (number of rows of input)
 * @param stream: cuda stream
 * @param flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 * @{
 */
template <typename math_t>
void signFlipComponents(const raft::handle_t& handle,
                        math_t* input,
                        math_t* components,
                        std::size_t n_samples,
                        std::size_t n_features,
                        std::size_t n_components,
                        cudaStream_t stream,
                        bool center,
                        bool flip_signs_based_on_U = false)
{
  detail::signFlipComponents(handle,
                             input,
                             components,
                             n_samples,
                             n_features,
                             n_components,
                             stream,
                             center,
                             flip_signs_based_on_U);
}

/**
 * @defgroup sign flip for PCA and tSVD. This is used to stabilize the sign of column major eigen
 * vectors
 * @param input: input matrix that will be used to determine the sign.
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param components: components matrix.
 * @param n_cols_comp: number of columns of components matrix
 * @param stream cuda stream
 * @{
 */
template <typename math_t>
void signFlip(math_t* input,
              std::size_t n_rows,
              std::size_t n_cols,
              math_t* components,
              std::size_t n_cols_comp,
              cudaStream_t stream)
{
  detail::signFlip(input, n_rows, n_cols, components, n_cols_comp, stream);
}

/**
 * @brief perform fit operation for the tsvd. Generates eigenvectors, explained vars, singular vals,
 * etc.
 * @param[in] handle: the internal cuml handle object
 * @param[in] input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is
 * indicated in prms.
 * @param[out] components: the principal components of the input data. Size n_cols * n_components.
 * @param[out] singular_vals: singular values of the data. Size n_components * 1
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void tsvdFit(const raft::handle_t& handle,
             math_t* input,
             math_t* components,
             math_t* singular_vals,
             const paramsTSVD& prms,
             cudaStream_t stream,
             bool flip_signs_based_on_U = false)
{
  detail::tsvdFit(handle, input, components, singular_vals, prms, stream, flip_signs_based_on_U);
}

/**
 * @brief performs fit and transform operations for the tsvd. Generates transformed data,
 * eigenvectors, explained vars, singular vals, etc.
 * @param[in] handle: the internal cuml handle object
 * @param[in] input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is
 * indicated in prms.
 * @param[out] trans_input: the transformed data. Size n_rows * n_components.
 * @param[out] components: the principal components of the input data. Size n_cols * n_components.
 * @param[out] explained_var: explained variances (eigenvalues) of the principal components. Size
 * n_components * 1.
 * @param[out] explained_var_ratio: the ratio of the explained variance and total variance. Size
 * n_components * 1.
 * @param[out] singular_vals: singular values of the data. Size n_components * 1
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void tsvdFitTransform(const raft::handle_t& handle,
                      math_t* input,
                      math_t* trans_input,
                      math_t* components,
                      math_t* explained_var,
                      math_t* explained_var_ratio,
                      math_t* singular_vals,
                      const paramsTSVD& prms,
                      cudaStream_t stream,
                      bool flip_signs_based_on_U = false)
{
  detail::tsvdFitTransform(handle,
                           input,
                           trans_input,
                           components,
                           explained_var,
                           explained_var_ratio,
                           singular_vals,
                           prms,
                           stream,
                           flip_signs_based_on_U);
}

/**
 * @brief performs transform operation for the tsvd. Transforms the data to eigenspace.
 * @param[in] handle the internal cuml handle object
 * @param[in] input: the data is transformed. Size n_rows x n_components.
 * @param[in] components: principal components of the input data. Size n_cols * n_components.
 * @param[out] trans_input: output that is transformed version of input
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void tsvdTransform(const raft::handle_t& handle,
                   math_t* input,
                   math_t* components,
                   math_t* trans_input,
                   const paramsTSVD& prms,
                   cudaStream_t stream)
{
  detail::tsvdTransform(handle, input, components, trans_input, prms, stream);
}

/**
 * @brief performs inverse transform operation for the tsvd. Transforms the transformed data back to
 * original data.
 * @param[in] handle the internal cuml handle object
 * @param[in] trans_input: the data is fitted to PCA. Size n_rows x n_components.
 * @param[in] components: transpose of the principal components of the input data. Size n_components
 * * n_cols.
 * @param[out] input: the data is fitted to PCA. Size n_rows x n_cols.
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void tsvdInverseTransform(const raft::handle_t& handle,
                          math_t* trans_input,
                          math_t* components,
                          math_t* input,
                          const paramsTSVD& prms,
                          cudaStream_t stream)
{
  detail::tsvdInverseTransform(handle, trans_input, components, input, prms, stream);
}

};  // end namespace raft::linalg
