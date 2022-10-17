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

#include "objectives/base.cuh"
#include "objectives/linear.cuh"
#include "objectives/logistic.cuh"
#include "objectives/regularizer.cuh"
#include "objectives/softmax.cuh"
#include "objectives/hinge.cuh"
#include "qn_solvers.cuh"
#include "qn_util.cuh"

#include <raft/solver/solver_types.hpp>
#include <raft/matrix/math.cuh>
#include <rmm/device_uvector.hpp>

namespace  raft::solver::quasi_newton::detail {

template <typename T>
void linear_decision_function(const raft::handle_t& handle,
                                 const qn_params& pams,
                                 SimpleMat<T>& X,
                                 int C,
                                 T* params,
                                 T* scores,
                                 cudaStream_t stream) {
  // NOTE: While gtests pass X as row-major, and python API passes X as
  // col-major, no extensive testing has been done to ensure that
  // this function works correctly for both input types
  int n_targets = qn_is_classification(pams.loss) && C == 2 ? 1 : C;
  LinearDims dims(n_targets, X.n, pams.fit_intercept);
  SimpleDenseMat<T> W(params, n_targets, dims.dims);
  SimpleDenseMat<T> Z(scores, n_targets, X.m);
  linearFwd(handle, Z, X, W, stream);
}
};  // namespace  raft::solver::quasi_newton::detail
