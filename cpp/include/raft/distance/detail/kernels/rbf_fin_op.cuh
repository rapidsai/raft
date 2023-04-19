/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

/*
 * This file defines rbf_fin_op, which is used in GramMatrixBase.
 *
 * This struct has been moved to a separate file, so that it is cheap to include
 * in distance/distance-ext.cuh, where an instance of raft::distance::distance
 * with the rbf_fin_op is instantiated.
 *
 */

#include <raft/core/math.hpp>                 // raft::exp
#include <raft/util/cuda_dev_essentials.cuh>  // HD

namespace raft::distance::kernels::detail {

/** @brief: Final op for Gram matrix with RBF kernel.
 *
 * Calculates output = e^(-gain * in)
 *
 */
template <typename OutT>
struct rbf_fin_op {
  OutT gain;

  explicit HD rbf_fin_op(OutT gain_) noexcept : gain(gain_) {}

  template <typename... Args>
  HDI OutT operator()(OutT d_val, Args... unused_args)
  {
    return raft::exp(-gain * d_val);
  }
};  // struct rbf_fin_op

}  // namespace raft::distance::kernels::detail
