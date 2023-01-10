/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include "base.cuh"
#include <raft/linalg/add.cuh>
#include <raft/solver/simple_mat.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft::solver::quasi_newton::detail::objectives {

template <typename T>
struct SquaredLoss : QNLinearBase<T, SquaredLoss<T>> {
  typedef QNLinearBase<T, SquaredLoss<T>> Super;

  const struct Lz {
    inline __device__ T operator()(const T y, const T z) const
    {
      T diff = z - y;
      return diff * diff * 0.5;
    }
  } lz;

  const struct Dlz {
    inline __device__ T operator()(const T y, const T z) const { return z - y; }
  } dlz;

  SquaredLoss(const raft::handle_t& handle, int D, bool has_bias)
    : Super(handle, D, 1, has_bias), lz{}, dlz{}
  {
  }

  inline T gradNorm(const SimpleVec<T>& grad, T* dev_scalar, cudaStream_t stream)
  {
    return squaredNorm(grad, dev_scalar, stream) * 0.5;
  }
};

template <typename T>
struct AbsLoss : QNLinearBase<T, AbsLoss<T>> {
  typedef QNLinearBase<T, AbsLoss<T>> Super;

  const struct Lz {
    inline __device__ T operator()(const T y, const T z) const { return raft::myAbs<T>(z - y); }
  } lz;

  const struct Dlz {
    inline __device__ T operator()(const T y, const T z) const
    {
      return z > y ? 1 : (z < y ? -1 : 0);
    }
  } dlz;

  AbsLoss(const raft::handle_t& handle, int D, bool has_bias)
    : Super(handle, D, 1, has_bias), lz{}, dlz{}
  {
  }

  inline T gradNorm(const SimpleVec<T>& grad, T* dev_scalar, cudaStream_t stream)
  {
    return nrm1(grad, dev_scalar, stream);
  }
};

};  // namespace  raft::solver::quasi_newton::detail::objectives
