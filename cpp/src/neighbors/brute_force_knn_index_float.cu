
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

#include <cstdint>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/brute_force-inl.cuh>

template void raft::neighbors::brute_force::search<float, int>(
  raft::resources const& res,
  const raft::neighbors::brute_force::index<float>& idx,
  raft::device_matrix_view<const float, int64_t, row_major> queries,
  raft::device_matrix_view<int, int64_t, row_major> neighbors,
  raft::device_matrix_view<float, int64_t, row_major> distances);

template void raft::neighbors::brute_force::search<float, int>(
  raft::resources const& res,
  raft::neighbors::brute_force::search_params const& params,
  const raft::neighbors::brute_force::index<float>& idx,
  raft::device_matrix_view<const float, int64_t, row_major> queries,
  raft::device_matrix_view<int, int64_t, row_major> neighbors,
  raft::device_matrix_view<float, int64_t, row_major> distances);

template void raft::neighbors::brute_force::search<float, int64_t>(
  raft::resources const& res,
  const raft::neighbors::brute_force::index<float>& idx,
  raft::device_matrix_view<const float, int64_t, row_major> queries,
  raft::device_matrix_view<int64_t, int64_t, row_major> neighbors,
  raft::device_matrix_view<float, int64_t, row_major> distances);

template void raft::neighbors::brute_force::search<float, int64_t>(
  raft::resources const& res,
  raft::neighbors::brute_force::search_params const& params,
  const raft::neighbors::brute_force::index<float>& idx,
  raft::device_matrix_view<const float, int64_t, row_major> queries,
  raft::device_matrix_view<int64_t, int64_t, row_major> neighbors,
  raft::device_matrix_view<float, int64_t, row_major> distances);

template raft::neighbors::brute_force::index<float> raft::neighbors::brute_force::
  build<float, raft::host_matrix_view<const float, int64_t, raft::row_major>::accessor_type>(
    raft::resources const& res,
    raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
    raft::distance::DistanceType metric,
    float metric_arg);

template raft::neighbors::brute_force::index<float> raft::neighbors::brute_force::
  build<float, raft::device_matrix_view<const float, int64_t, raft::row_major>::accessor_type>(
    raft::resources const& res,
    raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
    raft::distance::DistanceType metric,
    float metric_arg);

template raft::neighbors::brute_force::index<float> raft::neighbors::brute_force::
  build<float, raft::host_matrix_view<const float, int64_t, raft::row_major>::accessor_type>(
    raft::resources const& res,
    raft::neighbors::brute_force::index_params const& params,
    raft::host_matrix_view<const float, int64_t, raft::row_major> dataset);

template raft::neighbors::brute_force::index<float> raft::neighbors::brute_force::
  build<float, raft::device_matrix_view<const float, int64_t, raft::row_major>::accessor_type>(
    raft::resources const& res,
    raft::neighbors::brute_force::index_params const& params,
    raft::device_matrix_view<const float, int64_t, raft::row_major> dataset);
