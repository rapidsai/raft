/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#undef RAFT_EXPLICIT_INSTANTIATE_ONLY

#include "dist_adj_threshold.cuh"
#include <cstdint>
#include <raft/distance/distance-inl.cuh>

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

instantiate_raft_distance_distance(raft::distance::DistanceType::L2Expanded,
                                   float,
                                   float,
                                   uint8_t,
                                   raft::distance::threshold_float,
                                   int);

instantiate_raft_distance_distance(raft::distance::DistanceType::L2Expanded,
                                   double,
                                   double,
                                   uint8_t,
                                   raft::distance::threshold_double,
                                   int);

#undef instantiate_raft_distance_distance

#define instantiate_raft_distance_getWorkspaceSize(DistT, DataT, AccT, OutT, IdxT)  \
  template size_t raft::distance::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT>( \
    const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k)

instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, float, float, uint8_t, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, double, double, uint8_t, int);

#undef instantiate_raft_distance_getWorkspaceSize
