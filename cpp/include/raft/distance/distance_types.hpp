/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

namespace raft {
namespace distance {

/**
 * @brief Describes how precise and fast distance should be computed.
 */
enum class Compute_options {
  /** The choice of speed and accuracy is left to the implementation.
   *
   *  This will use Fast_Similar_Precision by default. If the environment
   *  variable `NVIDIA_TF32_OVERRIDE` is set, this will default to
   *  Fast_Reduced_Precision.
   *
   * */
  Unspecified,
  /** Use the most numerically accurate option.
   * */
  Precise,
  /** Use fast computation with similar precision.
   *
   *  - If possible, expand the norm computation for two points into the sum of
   *  norms minus an inner product:
   *
   *  || x - y ||^2 = || x ||^2 + || y ||^2 - 2 <x , y>
   *
   *  The inner product becomes a matrix multiplication for many points.
   *
   *  - If possible, execute the matrix multiplication using 3xtfloat, as
   *  described in [0].
   *
   *  [0] Ootomo H, Yokota R. Recovering single precision accuracy from Tensor Cores
   *  while surpassing the FP32 theoretical peak performance. The International
   *  Journal of High Performance Computing Applications. 2022;36(4):475-491.
   *  doi:10.1177/10943420221090256
   *
   * */
  Fast_Similar_Precision,
  /** Use reduced precision to speed up computation.
   *
   *  1. Use inner product expansion, as described above.
   *  2. Use tensor float precision instead of fp32 precision.
   *
   * */
  Fast_Reduced_Precision
};

/**
 * @brief Describes how the L2 norm should be computed.
 *
 */
struct L2_options {
  /** If true, compute squared L2 norm. */
  bool squared;
  /** Specify speed and precision of computation. */
  Compute_options compute_options;
};

/** enum to tell how to compute distance */
enum DistanceType : unsigned short {

  /** evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk) */
  L2Expanded = 0,
  /** same as above, but inside the epilogue, perform square root operation */
  L2SqrtExpanded = 1,
  /** cosine distance */
  CosineExpanded = 2,
  /** L1 distance */
  L1 = 3,
  /** evaluate as dist_ij += (x_ik - y-jk)^2 */
  L2Unexpanded = 4,
  /** same as above, but inside the epilogue, perform square root operation */
  L2SqrtUnexpanded = 5,
  /** basic inner product **/
  InnerProduct = 6,
  /** Chebyshev (Linf) distance **/
  Linf = 7,
  /** Canberra distance **/
  Canberra = 8,
  /** Generalized Minkowski distance **/
  LpUnexpanded = 9,
  /** Correlation distance **/
  CorrelationExpanded = 10,
  /** Jaccard distance **/
  JaccardExpanded = 11,
  /** Hellinger distance **/
  HellingerExpanded = 12,
  /** Haversine distance **/
  Haversine = 13,
  /** Bray-Curtis distance **/
  BrayCurtis = 14,
  /** Jensen-Shannon distance**/
  JensenShannon = 15,
  /** Hamming distance **/
  HammingUnexpanded = 16,
  /** KLDivergence **/
  KLDivergence = 17,
  /** RusselRao **/
  RusselRaoExpanded = 18,
  /** Dice-Sorensen distance **/
  DiceExpanded = 19,
  /** Precomputed (special value) **/
  Precomputed = 100
};

/**
 * Whether minimal distance corresponds to similar elements (using the given metric).
 */
inline bool is_min_close(DistanceType metric)
{
  bool select_min;
  switch (metric) {
    case DistanceType::InnerProduct:
      // Similarity metrics have the opposite meaning, i.e. nearest neighbors are those with larger
      // similarity (See the same logic at cpp/include/raft/sparse/spatial/detail/knn.cuh:362
      // {perform_k_selection})
      select_min = false;
      break;
    default: select_min = true;
  }
  return select_min;
}

namespace kernels {
enum KernelType { LINEAR, POLYNOMIAL, RBF, TANH };

/**
 * Parameters for kernel matrices.
 * The following kernels are implemented:
 * - LINEAR \f[ K(x_1,x_2) = <x_1,x_2>, \f] where \f$< , >\f$ is the dot product
 * - POLYNOMIAL \f[ K(x_1, x_2) = (\gamma <x_1,x_2> + \mathrm{coef0})^\mathrm{degree} \f]
 * - RBF \f[ K(x_1, x_2) = \exp(- \gamma |x_1-x_2|^2) \f]
 * - TANH \f[ K(x_1, x_2) = \tanh(\gamma <x_1,x_2> + \mathrm{coef0}) \f]
 */
struct KernelParams {
  // Kernel function parameters
  KernelType kernel;  //!< Type of the kernel function
  int degree;         //!< Degree of polynomial kernel (ignored by others)
  double gamma;       //!< multiplier in the
  double coef0;       //!< additive constant in poly and tanh kernels
};
}  // end namespace kernels

};  // namespace distance
};  // end namespace raft
