/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/distance/fused_distance_nn.cuh>
#include <raft/distance/fused_l2_nn.cuh>
#include <raft/linalg/norm.cuh>

#include <thrust/for_each.h>
#include <thrust/tuple.h>

namespace raft::runtime::distance {

template <typename IndexT, typename DataT>
struct KeyValueIndexOp {
  __host__ __device__ __forceinline__ IndexT
  operator()(const raft::KeyValuePair<IndexT, DataT>& a) const
  {
    return a.key;
  }
};

template <typename value_t, typename idx_t>
void compute_fused_l2_nn_min_arg(raft::resources const& handle,
                                 idx_t* min,
                                 const value_t* x,
                                 const value_t* y,
                                 idx_t m,
                                 idx_t n,
                                 idx_t k,
                                 bool sqrt)
{
  rmm::device_uvector<int> workspace(m, resource::get_cuda_stream(handle));
  auto kvp = raft::make_device_vector<raft::KeyValuePair<idx_t, value_t>>(handle, m);
  constexpr bool is_row_major = true;

  rmm::device_uvector<value_t> x_norms(m, resource::get_cuda_stream(handle));
  rmm::device_uvector<value_t> y_norms(n, resource::get_cuda_stream(handle));
  raft::linalg::rowNorm(
    x_norms.data(), x, k, m, raft::linalg::L2Norm, is_row_major, resource::get_cuda_stream(handle));
  raft::linalg::rowNorm(
    y_norms.data(), y, k, n, raft::linalg::L2Norm, is_row_major, resource::get_cuda_stream(handle));

  raft::distance::fusedL2NNMinReduce(kvp.data_handle(),
                                     x,
                                     y,
                                     x_norms.data(),
                                     y_norms.data(),
                                     m,
                                     n,
                                     k,
                                     (void*)workspace.data(),
                                     sqrt,
                                     true,
                                     resource::get_cuda_stream(handle));

  KeyValueIndexOp<idx_t, value_t> conversion_op;
  thrust::transform(resource::get_thrust_policy(handle),
                    kvp.data_handle(),
                    kvp.data_handle() + m,
                    min,
                    conversion_op);
  resource::sync_stream(handle);
}

template <typename value_t, typename idx_t>
void compute_fused_cosine_nn_min_arg(raft::resources const& handle,
                                     idx_t* min,
                                     const value_t* x,
                                     const value_t* y,
                                     idx_t m,
                                     idx_t n,
                                     idx_t k,
                                     bool sqrt)
{
  rmm::device_uvector<int> workspace(m, resource::get_cuda_stream(handle));
  auto kvp = raft::make_device_vector<raft::KeyValuePair<idx_t, value_t>>(handle, m);

  rmm::device_uvector<value_t> x_norms(m, resource::get_cuda_stream(handle));
  rmm::device_uvector<value_t> y_norms(n, resource::get_cuda_stream(handle));
  constexpr bool is_row_major = true;
  raft::linalg::rowNorm(x_norms.data(),
                        x,
                        k,
                        m,
                        raft::linalg::L2Norm,
                        is_row_major,
                        resource::get_cuda_stream(handle),
                        raft::sqrt_op{});
  raft::linalg::rowNorm(y_norms.data(),
                        y,
                        k,
                        n,
                        raft::linalg::L2Norm,
                        is_row_major,
                        resource::get_cuda_stream(handle),
                        raft::sqrt_op{});

  raft::distance::fusedDistanceNNMinReduce(kvp.data_handle(),
                                           x,
                                           y,
                                           x_norms.data(),
                                           y_norms.data(),
                                           m,
                                           n,
                                           k,
                                           (void*)workspace.data(),
                                           sqrt,
                                           true,
                                           is_row_major,
                                           raft::distance::DistanceType::CosineExpanded,
                                           0.0f,
                                           resource::get_cuda_stream(handle));

  KeyValueIndexOp<idx_t, value_t> conversion_op;
  thrust::transform(resource::get_thrust_policy(handle),
                    kvp.data_handle(),
                    kvp.data_handle() + m,
                    min,
                    conversion_op);
  resource::sync_stream(handle);
}

}  // end namespace raft::runtime::distance
