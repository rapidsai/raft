/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/operators.hpp>                          // raft::identity_op
#include <raft/distance/detail/distance_ops/all_ops.cuh>    // ops::*
#include <raft/distance/detail/distance_ops/cutlass.cuh>    // ops::has_cutlass_op
#include <raft/distance/detail/kernels/rbf_fin_op.cuh>      // rbf_fin_op
#include <raft/distance/detail/pairwise_matrix/params.cuh>  // pairwise_matrix_params
#include <raft/util/raft_explicit.hpp>                      // RAFT_EXPLICIT

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

namespace raft::distance::detail {

template <typename OpT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void pairwise_matrix_dispatch(OpT distance_op,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              const DataT* x,
                              const DataT* y,
                              const DataT* x_norm,
                              const DataT* y_norm,
                              OutT* out,
                              FinOpT fin_op,
                              cudaStream_t stream,
                              bool is_row_major) RAFT_EXPLICIT;

};  // namespace raft::distance::detail

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_distance_detail_pairwise_matrix_dispatch(                     \
  OpT, DataT, AccT, OutT, FinOpT, IdxT)                                                \
  extern template void raft::distance::detail::                                        \
    pairwise_matrix_dispatch<OpT<DataT, AccT, IdxT>, DataT, AccT, OutT, FinOpT, IdxT>( \
      OpT<DataT, AccT, IdxT> distance_op,                                              \
      IdxT m,                                                                          \
      IdxT n,                                                                          \
      IdxT k,                                                                          \
      const DataT* x,                                                                  \
      const DataT* y,                                                                  \
      const DataT* x_norm,                                                             \
      const DataT* y_norm,                                                             \
      OutT* out,                                                                       \
      FinOpT fin_op,                                                                   \
      cudaStream_t stream,                                                             \
      bool is_row_major)

/*
 * Hierarchy of instantiations:
 *
 * This file defines extern template instantiations of the distance kernels. The
 * instantiation of the public API is handled in raft/distance/distance-ext.cuh.
 *
 * After adding an instance here, make sure to also add the instance there.
 */

// The following two instances are used in the RBF kernel object. Note the use of int64_t for the
// index type.
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::l2_unexp_distance_op,
  float,
  float,
  float,
  raft::distance::kernels::detail::rbf_fin_op<float>,
  int64_t);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::l2_unexp_distance_op,
  double,
  double,
  double,
  raft::distance::kernels::detail::rbf_fin_op<double>,
  int64_t);

// Rest of instances
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::canberra_distance_op, float, float, float, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::canberra_distance_op,
  double,
  double,
  double,
  raft::identity_op,
  int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::correlation_distance_op,
  float,
  float,
  float,
  raft::identity_op,
  int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::correlation_distance_op,
  double,
  double,
  double,
  raft::identity_op,
  int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::cosine_distance_op, float, float, float, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::cosine_distance_op, double, double, double, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::dice_distance_op, float, float, float, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::dice_distance_op, double, double, double, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::hamming_distance_op, float, float, float, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::hamming_distance_op, double, double, double, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::hellinger_distance_op, float, float, float, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::hellinger_distance_op,
  double,
  double,
  double,
  raft::identity_op,
  int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::jensen_shannon_distance_op,
  float,
  float,
  float,
  raft::identity_op,
  int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::jensen_shannon_distance_op,
  double,
  double,
  double,
  raft::identity_op,
  int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::kl_divergence_op, float, float, float, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::kl_divergence_op, double, double, double, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::l1_distance_op, float, float, float, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::l1_distance_op, double, double, double, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::l2_exp_distance_op, float, float, float, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::l2_exp_distance_op, double, double, double, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::l2_unexp_distance_op, float, float, float, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::l2_unexp_distance_op,
  double,
  double,
  double,
  raft::identity_op,
  int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::l_inf_distance_op, float, float, float, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::l_inf_distance_op, double, double, double, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::lp_unexp_distance_op, float, float, float, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::lp_unexp_distance_op,
  double,
  double,
  double,
  raft::identity_op,
  int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::russel_rao_distance_op, float, float, float, raft::identity_op, int);
instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  raft::distance::detail::ops::russel_rao_distance_op,
  double,
  double,
  double,
  raft::identity_op,
  int);

#undef instantiate_raft_distance_detail_pairwise_matrix_dispatch
