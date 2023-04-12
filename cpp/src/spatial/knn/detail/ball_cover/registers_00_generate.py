#!/usr/bin/env python3

header = """/*
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

#include <cstdint> // int64_t
#include <raft/spatial/knn/detail/ball_cover/registers-inl.cuh>

"""


macro_pass_one = """
#define instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_one(                            \\
  Mvalue_idx, Mvalue_t, Mvalue_int, Mdims, Mdist_func)                                       \\
  template void                                                                       \\
  raft::spatial::knn::detail::rbc_low_dim_pass_one<Mvalue_idx, Mvalue_t, Mvalue_int, Mdims>( \\
    raft::device_resources const& handle,                                                    \\
    const BallCoverIndex<Mvalue_idx, Mvalue_t, Mvalue_int>& index,                           \\
    const Mvalue_t* query,                                                                   \\
    const Mvalue_int n_query_rows,                                                           \\
    Mvalue_int k,                                                                            \\
    const Mvalue_idx* R_knn_inds,                                                            \\
    const Mvalue_t* R_knn_dists,                                                             \\
    Mdist_func<Mvalue_t, Mvalue_int>& dfunc,                                                 \\
    Mvalue_idx* inds,                                                                        \\
    Mvalue_t* dists,                                                                         \\
    float weight,                                                                            \\
    Mvalue_int* dists_counter)

"""

macro_pass_two = """
#define instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_two(                            \\
  Mvalue_idx, Mvalue_t, Mvalue_int, Mdims, Mdist_func)                                       \\
  template void                                                                       \\
  raft::spatial::knn::detail::rbc_low_dim_pass_two<Mvalue_idx, Mvalue_t, Mvalue_int, Mdims>( \\
    raft::device_resources const& handle,                                                    \\
    const BallCoverIndex<Mvalue_idx, Mvalue_t, Mvalue_int>& index,                           \\
    const Mvalue_t* query,                                                                   \\
    const Mvalue_int n_query_rows,                                                           \\
    Mvalue_int k,                                                                            \\
    const Mvalue_idx* R_knn_inds,                                                            \\
    const Mvalue_t* R_knn_dists,                                                             \\
    Mdist_func<Mvalue_t, Mvalue_int>& dfunc,                                                 \\
    Mvalue_idx* inds,                                                                        \\
    Mvalue_t* dists,                                                                         \\
    float weight,                                                                            \\
    Mvalue_int* dists_counter)

"""

distances = dict(
    haversine="raft::spatial::knn::detail::HaversineFunc",
    euclidean="raft::spatial::knn::detail::EuclideanFunc",
    dist="raft::spatial::knn::detail::DistFunc",
)

for k, v in distances.items():
    for dim in [2, 3]:
        path = f"registers_pass_one_{dim}d_{k}.cu"
        with open(path, "w") as f:
            f.write(header)
            f.write(macro_pass_one)
            f.write(f"instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_one(\n")
            f.write(f"  std::int64_t, float, std::uint32_t, {dim}, {v});\n")
            f.write("#undef instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_one\n")
        print(f"src/spatial/knn/detail/ball_cover/{path}")

for k, v in distances.items():
    for dim in [2, 3]:
        path = f"registers_pass_two_{dim}d_{k}.cu"
        with open(path, "w") as f:
            f.write(header)
            f.write(macro_pass_two)
            f.write(f"instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_two(\n")
            f.write(f"  std::int64_t, float, std::uint32_t, {dim}, {v});\n")
            f.write("#undef instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_two\n")
        print(f"src/spatial/knn/detail/ball_cover/{path}")
