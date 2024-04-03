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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/detail/fused_distance_nn/helper_structs.cuh>

namespace raft::distance {

/**
 * \defgroup fused_l2_nn Fused 1-nearest neighbors
 * @{
 */

template <typename LabelT, typename DataT>
using KVPMinReduce = detail::KVPMinReduceImpl<LabelT, DataT>;

template <typename LabelT, typename DataT>
using MinAndDistanceReduceOp = detail::MinAndDistanceReduceOpImpl<LabelT, DataT>;

template <typename LabelT, typename DataT>
using MinReduceOp = detail::MinReduceOpImpl<LabelT, DataT>;

/** @} */

/**
 * Initialize array using init value from reduction op
 */
template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
void initialize(raft::resources const& handle, OutT* min, IdxT m, DataT maxVal, ReduceOpT redOp)
{
  detail::initialize<DataT, OutT, IdxT, ReduceOpT>(
    min, m, maxVal, redOp, resource::get_cuda_stream(handle));
}

}  // namespace raft::distance
