/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <raft/distance/detail/kernels/rbf_fin_op.cuh>  // rbf_fin_op
#include <raft/distance/distance-inl.cuh>

/*
 * Hierarchy of instantiations:
 *
 * This file defines the template instantiations for the public API of
 * raft::distance. To improve compile times, the compilation of the distance
 * kernels is handled in distance/detail/pairwise_matrix/dispatch_*.cu.
 *
 */

#define instantiate_raft_distance_distance(DT, DataT, AccT, OutT, FinalLambda, IdxT) \
  template void raft::distance::distance<DT, DataT, AccT, OutT, FinalLambda, IdxT>(  \
    raft::resources const& handle,                                                   \
    const DataT* x,                                                                  \
    const DataT* y,                                                                  \
    OutT* dist,                                                                      \
    IdxT m,                                                                          \
    IdxT n,                                                                          \
    IdxT k,                                                                          \
    void* workspace,                                                                 \
    size_t worksize,                                                                 \
    FinalLambda fin_op,                                                              \
    bool isRowMajor,                                                                 \
    DataT metric_arg)

// The following two instances are used in test/distance/gram.cu. Note the use
// of int64_t for the index type.
instantiate_raft_distance_distance(raft::distance::DistanceType::L2Unexpanded,
                                   float,
                                   float,
                                   float,
                                   raft::distance::kernels::detail::rbf_fin_op<float>,
                                   int64_t);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2Unexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::distance::kernels::detail::rbf_fin_op<double>,
                                   int64_t);

instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CorrelationExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CorrelationExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::identity_op,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CosineExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CosineExpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::DiceExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::DiceExpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HammingUnexpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HammingUnexpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HellingerExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HellingerExpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L1, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L1, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtExpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtUnexpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtUnexpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Linf, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Linf, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::RusselRaoExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::RusselRaoExpanded, double, double, double, raft::identity_op, int);

#undef instantiate_raft_distance_distance

// Same, but without raft::identity_op
#define instantiate_raft_distance_distance(DT, DataT, AccT, OutT, IdxT) \
  template void raft::distance::distance<DT, DataT, AccT, OutT, IdxT>(  \
    raft::resources const& handle,                                      \
    const DataT* x,                                                     \
    const DataT* y,                                                     \
    OutT* dist,                                                         \
    IdxT m,                                                             \
    IdxT n,                                                             \
    IdxT k,                                                             \
    void* workspace,                                                    \
    size_t worksize,                                                    \
    bool isRowMajor,                                                    \
    DataT metric_arg)

instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CorrelationExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CorrelationExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CosineExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CosineExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::DiceExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::DiceExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HammingUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HammingUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HellingerExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HellingerExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, double, double, double, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L1, float, float, float, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L1, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, double, double, double, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::Linf, float, float, float, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::Linf, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::RusselRaoExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::RusselRaoExpanded, double, double, double, int);

#undef instantiate_raft_distance_distance

// Same, but without workspace
#define instantiate_raft_distance_distance(DT, DataT, AccT, OutT, IdxT) \
  template void raft::distance::distance<DT, DataT, AccT, OutT, IdxT>(  \
    raft::resources const& handle,                                      \
    const DataT* x,                                                     \
    const DataT* y,                                                     \
    OutT* dist,                                                         \
    IdxT m,                                                             \
    IdxT n,                                                             \
    IdxT k,                                                             \
    bool isRowMajor,                                                    \
    DataT metric_arg)

instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CorrelationExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CorrelationExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CosineExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CosineExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::DiceExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::DiceExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HammingUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HammingUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HellingerExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HellingerExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, double, double, double, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L1, float, float, float, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L1, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, double, double, double, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::Linf, float, float, float, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::Linf, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::RusselRaoExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::RusselRaoExpanded, double, double, double, int);

#undef instantiate_raft_distance_distance

#define instantiate_raft_distance_getWorkspaceSize(DistT, DataT, AccT, OutT, IdxT)  \
  template size_t raft::distance::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT>( \
    const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k)

instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Canberra, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Canberra, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::CorrelationExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::CorrelationExpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::CosineExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::CosineExpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::DiceExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::DiceExpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::HammingUnexpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::HammingUnexpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::HellingerExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::HellingerExpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::InnerProduct, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::InnerProduct, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::JensenShannon, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::JensenShannon, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::KLDivergence, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::KLDivergence, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L1, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L1, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2SqrtExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2SqrtExpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2SqrtUnexpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2SqrtUnexpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Unexpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Linf, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Linf, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::LpUnexpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::LpUnexpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::RusselRaoExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::RusselRaoExpanded, double, double, double, int);

#undef instantiate_raft_distance_getWorkspaceSize

#define instantiate_raft_distance_getWorkspaceSize(DistT, DataT, AccT, OutT, IdxT, layout)  \
  template size_t raft::distance::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT, layout>( \
    raft::device_matrix_view<DataT, IdxT, layout> const& x,                                 \
    raft::device_matrix_view<DataT, IdxT, layout> const& y)

// We could consider not taking template parameters for this function. The
// number of instantiations seems a bit excessive..
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Canberra, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Canberra, double, double, double, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Canberra, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Canberra, double, double, double, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CorrelationExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CorrelationExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CorrelationExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CorrelationExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CosineExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CosineExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CosineExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CosineExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::DiceExpanded, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::DiceExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::DiceExpanded, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::DiceExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HammingUnexpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HammingUnexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HammingUnexpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HammingUnexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HellingerExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HellingerExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HellingerExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HellingerExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::InnerProduct, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::InnerProduct,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::InnerProduct, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::InnerProduct,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::JensenShannon, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::JensenShannon,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::JensenShannon, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::JensenShannon,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::KLDivergence, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::KLDivergence,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::KLDivergence, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::KLDivergence,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L1, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L1, double, double, double, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L1, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L1, double, double, double, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, double, double, double, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, double, double, double, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtUnexpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtUnexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtUnexpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtUnexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2Unexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, int, raft::layout_f_contiguous);

#undef instantiate_raft_distance_getWorkspaceSize

#define instantiate_raft_distance_pairwise_distance(DataT, IdxT)                        \
  template void raft::distance::pairwise_distance(raft::resources const& handle,        \
                                                  const DataT* x,                       \
                                                  const DataT* y,                       \
                                                  DataT* dist,                          \
                                                  IdxT m,                               \
                                                  IdxT n,                               \
                                                  IdxT k,                               \
                                                  rmm::device_uvector<char>& workspace, \
                                                  raft::distance::DistanceType metric,  \
                                                  bool isRowMajor,                      \
                                                  DataT metric_arg)

instantiate_raft_distance_pairwise_distance(float, int);
instantiate_raft_distance_pairwise_distance(double, int);

#undef instantiate_raft_distance_pairwise_distance

// Same, but without workspace
#define instantiate_raft_distance_pairwise_distance(DataT, IdxT)                       \
  template void raft::distance::pairwise_distance(raft::resources const& handle,       \
                                                  const DataT* x,                      \
                                                  const DataT* y,                      \
                                                  DataT* dist,                         \
                                                  IdxT m,                              \
                                                  IdxT n,                              \
                                                  IdxT k,                              \
                                                  raft::distance::DistanceType metric, \
                                                  bool isRowMajor,                     \
                                                  DataT metric_arg)

instantiate_raft_distance_pairwise_distance(float, int);
instantiate_raft_distance_pairwise_distance(double, int);

#undef instantiate_raft_distance_pairwise_distance

// Version with mdspan
#define instantiate_raft_distance_distance(DistT, DataT, AccT, OutT, layout, IdxT) \
  template void raft::distance::distance<DistT, DataT, AccT, OutT, layout, IdxT>(  \
    raft::resources const& handle,                                                 \
    raft::device_matrix_view<DataT, IdxT, layout> const x,                         \
    raft::device_matrix_view<DataT, IdxT, layout> const y,                         \
    raft::device_matrix_view<OutT, IdxT, layout> dist,                             \
    DataT metric_arg)

// Again, we might want to consider reigning in the number of instantiations...
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, double, double, double, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, double, double, double, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CorrelationExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CorrelationExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CorrelationExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CorrelationExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CosineExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CosineExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CosineExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CosineExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::DiceExpanded, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::DiceExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::DiceExpanded, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::DiceExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HammingUnexpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HammingUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HammingUnexpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HammingUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HellingerExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HellingerExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HellingerExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HellingerExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::InnerProduct,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::InnerProduct,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::JensenShannon,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::JensenShannon,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::KLDivergence,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::KLDivergence,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L1, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L1, double, double, double, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L1, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L1, double, double, double, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, double, double, double, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, double, double, double, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtUnexpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtUnexpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2Unexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2Unexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Linf, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Linf, double, double, double, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Linf, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Linf, double, double, double, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::LpUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::LpUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::RusselRaoExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::RusselRaoExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::RusselRaoExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::RusselRaoExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);

#undef instantiate_raft_distance_distance

#define instantiate_raft_distance_pairwise_distance(DataT, layout, IdxT) \
  template void raft::distance::pairwise_distance(                       \
    raft::resources const& handle,                                       \
    raft::device_matrix_view<DataT, IdxT, layout> const x,               \
    raft::device_matrix_view<DataT, IdxT, layout> const y,               \
    raft::device_matrix_view<DataT, IdxT, layout> dist,                  \
    raft::distance::DistanceType metric,                                 \
    DataT metric_arg)

instantiate_raft_distance_pairwise_distance(float, raft::layout_c_contiguous, int);
instantiate_raft_distance_pairwise_distance(float, raft::layout_f_contiguous, int);
instantiate_raft_distance_pairwise_distance(double, raft::layout_c_contiguous, int);
instantiate_raft_distance_pairwise_distance(double, raft::layout_f_contiguous, int);

#undef instantiate_raft_distance_pairwise_distance
