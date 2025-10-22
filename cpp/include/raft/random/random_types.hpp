/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace raft::random {

/**
 * \ingroup multi_variable_gaussian
 * @{
 */

/**
 * @brief Matrix decomposition method for `multi_variable_gaussian` to use.
 *
 * `multi_variable_gaussian` can use any of the following methods.
 *
 * - `CHOLESKY`: Uses Cholesky decomposition on the normal equations.
 *   This may be faster than the other two methods, but less accurate.
 *
 * - `JACOBI`: Uses the singular value decomposition (SVD) computed with
 *   cuSOLVER's gesvdj algorithm, which is based on the Jacobi method
 *   (sweeps of plane rotations).  This exposes more parallelism
 *   for small and medium size matrices than the QR option below.
 *
 * - `QR`: Uses the SVD computed with cuSOLVER's gesvd algorithm,
 *   which is based on the QR algorithm.
 */
enum class multi_variable_gaussian_decomposition_method { CHOLESKY, JACOBI, QR };

/** @} */

};  // end of namespace raft::random
