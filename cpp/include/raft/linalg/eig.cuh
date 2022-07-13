/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#ifndef __EIG_H
#define __EIG_H

#pragma once

#include "detail/eig.cuh"

#include <raft/core/mdarray.hpp>

namespace raft {
namespace linalg {

/**
 * @defgroup eig Eigen Decomposition Methods
 * @{
 */

/**
 * @brief eig decomp with divide and conquer method for the column-major
 * symmetric matrices
 * @param handle raft handle
 * @param in the input buffer (symmetric matrix that has real eig values and
 * vectors.
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @param stream cuda stream
 */
template <typename math_t>
void eigDC(const raft::handle_t& handle,
           const math_t* in,
           std::size_t n_rows,
           std::size_t n_cols,
           math_t* eig_vectors,
           math_t* eig_vals,
           cudaStream_t stream)
{
  detail::eigDC(handle, in, n_rows, n_cols, eig_vectors, eig_vals, stream);
}

using detail::COPY_INPUT;
using detail::EigVecMemUsage;
using detail::OVERWRITE_INPUT;

/**
 * @brief eig sel decomp with divide and conquer method for the column-major
 * symmetric matrices
 * @param handle raft handle
 * @param in the input buffer (symmetric matrix that has real eig values and
 * vectors.
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param n_eig_vals: number of eigenvectors to be generated
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @param memUsage: the memory selection for eig vector output
 * @param stream cuda stream
 */
template <typename math_t>
void eigSelDC(const raft::handle_t& handle,
              math_t* in,
              std::size_t n_rows,
              std::size_t n_cols,
              std::size_t n_eig_vals,
              math_t* eig_vectors,
              math_t* eig_vals,
              EigVecMemUsage memUsage,
              cudaStream_t stream)
{
  detail::eigSelDC(handle, in, n_rows, n_cols, n_eig_vals, eig_vectors, eig_vals, memUsage, stream);
}

/**
 * @brief overloaded function for eig decomp with Jacobi method for the
 * column-major symmetric matrices (in parameter)
 * @param handle: raft handle
 * @param in: input matrix
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @param stream: stream on which this function will be run
 * @param tol: error tolerance for the jacobi method. Algorithm stops when the
 * error is below tol
 * @param sweeps: number of sweeps in the Jacobi algorithm. The more the better
 * accuracy.
 */
template <typename math_t>
void eigJacobi(const raft::handle_t& handle,
               const math_t* in,
               std::size_t n_rows,
               std::size_t n_cols,
               math_t* eig_vectors,
               math_t* eig_vals,
               cudaStream_t stream,
               math_t tol = 1.e-7,
               int sweeps = 15)
{
  detail::eigJacobi(handle, in, n_rows, n_cols, eig_vectors, eig_vals, stream, tol, sweeps);
}

/**
 * @brief eig decomp with divide and conquer method for the column-major
 * symmetric matrices
 * @tparam ElementType the data-type of input and output
 * @param handle raft::handle_t
 * @param in input raft::matrix_view (symmetric matrix that has real eig values and
 * vectors)
 * @param eig_vectors: eigenvectors output of type raft::matrix_view
 * @param eig_vals: eigen values output of type raft::vector_view
 * @param memUsage: the memory selection for eig vector output
 */
template <typename ElementType>
void eig_dc(const raft::handle_t& handle,
            const raft::matrix_view<ElementType, raft::col_major> in,
            raft::matrix_view<ElementType, raft::col_major> eig_vectors,
            raft::vector_view<ElementType> eig_vals)
{
  RAFT_EXPECTS(in.is_contiguous(), "Input must be contiguous");
  RAFT_EXPECTS(eig_vectors.is_contiguous(), "Eigen Vectors must be contiguous");
  RAFT_EXPECTS(eig_vals.is_contiguous(), "Eigen Values must be contiguous");
  RAFT_EXPECTS(in.size() == eig_vectors.size(), "Size mismatch between Input and Eigen Vectors");
  RAFT_EXPECTS(eig_vals.size() == in.extent(1), "Size mismatch between Input and Eigen Values");

  eigDC(handle,
        in.data(),
        in.extent(0),
        in.extent(1),
        eig_vectors.data(),
        eig_vals.data(),
        handle.get_stream());
}

/**
 * @brief eig decomp to select top-n eigen values with divide and conquer method
 *        for the column-major symmetric matrices
 * @tparam ElementType the data-type of input and output
 * @param handle raft::handle_t
 * @param in input raft::matrix_view (symmetric matrix that has real eig values and
 * vectors)
 * @param n_eig_vals: number of eigenvectors to be generated
 * @param eig_vectors: eigenvectors output of type raft::matrix_view
 * @param eig_vals: eigen values output of type raft::vector_view
 */
template <typename ElementType>
void eig_dc_select(const raft::handle_t& handle,
                   const raft::matrix_view<ElementType, raft::col_major> in,
                   std::size_t n_eig_vals,
                   raft::matrix_view<ElementType, raft::col_major> eig_vectors,
                   raft::vector_view<ElementType> eig_vals,
                   EigVecMemUsage memUsage)
{
  RAFT_EXPECTS(in.is_contiguous(), "Input must be contiguous");
  RAFT_EXPECTS(eig_vectors.is_contiguous(), "Eigen Vectors must be contiguous");
  RAFT_EXPECTS(eig_vals.is_contiguous(), "Eigen Values must be contiguous");
  RAFT_EXPECTS(eig_vectors.size() == n_eig_vals * in.extent(0),
               "Size mismatch between Input and Eigen Vectors");
  RAFT_EXPECTS(eig_vals.size() == n_eig_vals, "Size mismatch between Input and Eigen Values");

  raft::linalg::eigSelDC(handle,
                         in.data(),
                         in.extent(0),
                         in.extent(1),
                         n_eig_vals,
                         eig_vectors.data(),
                         eig_vals.data(),
                         memUsage,
                         handle.get_stream());
}

/**
 * @brief overloaded function for eig decomp with Jacobi method for the
 * column-major symmetric matrices (in parameter)
 * @tparam ElementType the data-type of input and output
 * @param handle raft::handle_t
 * @param in input raft::matrix_view (symmetric matrix that has real eig values and
 * vectors)
 * @param eig_vectors: eigenvectors output of type raft::matrix_view
 * @param eig_vals: eigen values output of type raft::vector_view
 * @param tol: error tolerance for the jacobi method. Algorithm stops when the
 * error is below tol
 * @param sweeps: number of sweeps in the Jacobi algorithm. The more the better
 * accuracy.
 */
template <typename ElementType>
void eig_jacobi(const raft::handle_t& handle,
                const raft::matrix_view<ElementType, raft::col_major> in,
                raft::matrix_view<ElementType, raft::col_major> eig_vectors,
                raft::vector_view<ElementType> eig_vals,
                ElementType tol = 1.e-7,
                int sweeps      = 15)
{
  RAFT_EXPECTS(in.is_contiguous(), "Input must be contiguous");
  RAFT_EXPECTS(eig_vectors.is_contiguous(), "Eigen Vectors must be contiguous");
  RAFT_EXPECTS(eig_vals.is_contiguous(), "Eigen Values must be contiguous");
  RAFT_EXPECTS(in.size() == eig_vectors.size(), "Size mismatch between Input and Eigen Vectors");
  RAFT_EXPECTS(eig_vals.size() == in.extent(1), "Size mismatch between Input and Eigen Values");

  eigJacobi(handle,
            in.data(),
            in.extent(0),
            in.extent(1),
            eig_vectors.data(),
            eig_vals.data(),
            handle.get_stream(),
            tol,
            sweeps);
}

/** @} */  // end of eig

};  // end namespace linalg
};  // end namespace raft

#endif