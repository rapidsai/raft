/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
