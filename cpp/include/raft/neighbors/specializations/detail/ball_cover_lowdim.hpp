/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <raft/spatial/knn/detail/ball_cover/common.cuh>
#include <raft/spatial/knn/detail/ball_cover/registers.cuh>

#include <cstdint>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

extern template void rbc_low_dim_pass_one<std::int64_t, float, std::uint32_t, 2>(
  raft::resources const& handle,
  const BallCoverIndex<std::int64_t, float, std::uint32_t>& index,
  const float* query,
  const std::uint32_t n_query_rows,
  std::uint32_t k,
  const std::int64_t* R_knn_inds,
  const float* R_knn_dists,
  DistFunc<float, std::uint32_t>& dfunc,
  std::int64_t* inds,
  float* dists,
  float weight,
  std::uint32_t* dists_counter);

extern template void rbc_low_dim_pass_two<std::int64_t, float, std::uint32_t, 2>(
  raft::resources const& handle,
  const BallCoverIndex<std::int64_t, float, std::uint32_t>& index,
  const float* query,
  const std::uint32_t n_query_rows,
  std::uint32_t k,
  const std::int64_t* R_knn_inds,
  const float* R_knn_dists,
  DistFunc<float, std::uint32_t>& dfunc,
  std::int64_t* inds,
  float* dists,
  float weight,
  std::uint32_t* post_dists_counter);

extern template void rbc_low_dim_pass_one<std::int64_t, float, std::uint32_t, 3>(
  raft::resources const& handle,
  const BallCoverIndex<std::int64_t, float, std::uint32_t>& index,
  const float* query,
  const std::uint32_t n_query_rows,
  std::uint32_t k,
  const std::int64_t* R_knn_inds,
  const float* R_knn_dists,
  DistFunc<float, std::uint32_t>& dfunc,
  std::int64_t* inds,
  float* dists,
  float weight,
  std::uint32_t* dists_counter);

extern template void rbc_low_dim_pass_two<std::int64_t, float, std::uint32_t, 3>(
  raft::resources const& handle,
  const BallCoverIndex<std::int64_t, float, std::uint32_t>& index,
  const float* query,
  const std::uint32_t n_query_rows,
  std::uint32_t k,
  const std::int64_t* R_knn_inds,
  const float* R_knn_dists,
  DistFunc<float, std::uint32_t>& dfunc,
  std::int64_t* inds,
  float* dists,
  float weight,
  std::uint32_t* post_dists_counter);

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft