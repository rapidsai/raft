/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/solver/detail/qn/objectives/base.cuh>
#include <raft/solver/detail/qn/objectives/hinge.cuh>
#include <raft/solver/detail/qn/objectives/linear.cuh>
#include <raft/solver/detail/qn/objectives/logistic.cuh>
#include <raft/solver/detail/qn/objectives/regularizer.cuh>
#include <raft/solver/detail/qn/objectives/softmax.cuh>

#include <raft/solver/detail/qn/qn_solvers.cuh>
#include <raft/solver/simple_mat.cuh>
#include <raft/solver/solver_types.hpp>

namespace raft::solver::quasi_newton {

/**
 * The following loss functions are wrapped only so they will be included in the docs
 */

/**
 * Absolute difference loss function specification
 * @tparam T
 */
template <typename T>
struct AbsLoss : detail::objectives::AbsLoss<T> {
  AbsLoss(const raft::handle_t& handle, int D, bool has_bias)
    : detail::objectives::AbsLoss(handle, D, has_bias)
  {
  }
};

/**
 * Squared loss function specification
 * @tparam T
 */
template <typename T>
struct SquaredLoss : detail::objectives::SquaredLoss<T> {
  SquaredLoss(const raft::handle_t& handle, int D, bool has_bias)
    : detail::objectives::SquaredLoss(handle, D, 1, has_bias), lz{}, dlz{}
  {
  }
};

/**
 * Standard hinge loss function specification
 * @tparam T
 */
template <typename T>
struct HingeLoss : detail::objectives::HingeLoss<T> {
  HingeLoss(const raft::handle_t& handle, int D, bool has_bias)
    : detail::objectives::HingeLoss(handle, D, has_bias)
  {
  }
};

/**
 *
 * @tparam T
 */
template <typename T>
struct LogisticLoss : detail::objectives::LogisticLoss<T> {
  LogisticLoss(const raft::handle_t& handle, int D, bool has_bias)
    : detail::objectives::LogisticLoss(handle, D, has_bias)
  {
  }
};

/**
 * Squared hinge loss function specification
 * @tparam T
 */
template <typename T>
struct SqHingeLoss : detail::objectives::SqHingeLoss<T> {
  SqHingeLoss(const raft::handle_t& handle, int D, bool has_bias)
    : detail::objectives::SqHingeLoss(handle, D, has_bias)
  {
  }
};

  /**
   * Epsilon insensitive (regression) hinge loss function specification
   * @tparam T
   */
  template <typename T>
  struct EpsInsHingeLoss : detail::objectives::EpsInsHingeLoss<T> {
  EpsInsHingeLoss(const raft::handle_t& handle, int D, bool has_bias, T sensitivity)
    : detail::objectives::EpsInsHingeLoss(handle, D, 1, has_bias), lz{sensitivity}, dlz{sensitivity}
  {
  }
};

/**
 * Squared Epsilon insensitive (regression) hinge loss function specification
 * @tparam T
 */
template <typename T>
struct SqEpsInsHingeLoss : detail::objectives::SqEpsInsHingeLoss<T> {
  SqEpsInsHingeLoss(const raft::handle_t& handle, int D, bool has_bias, T sensitivity)
    : detail::objectives::SqEpsInsHingeLoss(handle, D, 1, has_bias),
      lz{sensitivity},
      dlz{sensitivity}
  {
  }
};

/**
 * Tikhonov (l2) penalty function
 * @tparam T
 */
template <typename T>
struct Tikhonov : detail::objectives::Tikhonov<T> {
  Tikhonov(T l2) : detail::objectives::Tikhonov<T>(l2) {}

  Tikhonov(const Tikhonov<T>& other) : detail::objectives::Tikhonov<T>(other.l2_penalty) {}
};

/**
 * Loss function wrapper that add a penalty to another loss function
 *
 * Example:
 *
 * raft::handle_t handle;
 * AbsLoss<float> abs_loss(handle, 5, true);
 * Tikhonov<float> l2_reg(0.3);
 * RegularizedQN(&abs_loss, &reg);
 *
 * @tparam T
 * @tparam Loss
 * @tparam Reg
 */
template <typename T, class Loss, class Reg>
class RegularizedQN : public detail::objectives::RegularizedQN<T, Loss, Reg> {
  RegularizedQN(Loss* loss, Reg* reg) : detail::objectives::RegularizedQN(loss, reg) {}
};

/**
 * Base loss function that constrains the solution to a linear system
 * @tparam T
 * @tparam Loss
 */
template <typename T, class Loss>
struct QNLinearBase : detail::objectives::QNLinearBase<T, Loss> {
  QNLinearBase(const raft::handle_t& handle, int D, int C, bool fit_intercept)
    : detail::objectives::QNLinearBase(C, D, fit_intercept)
  {
  }
};

/**
 * Softmax loss function specification
 * @tparam T
 */
template <typename T>
struct Softmax : detail::objectives::Softmax<T> {
  Softmax(const raft::handle_t& handle, int D, int C, bool has_bias)
    : detail::objectives::Softmax(handle, D, C, has_bias)
  {
  }
};

/**
 * Constructs a end-to-end quasi-newton objective function to solve the system
 * AX = b (where each row in X contains the coefficients for each target)
 *
 * Example:
 *
 * @tparam T
 * @tparam QuasiNewtonObjective
 */
template <typename T, class QuasiNewtonObjective>
struct ObjectiveWithData : detail::objectives::QNWithData<T, QuasiNewtonObjective> {
  ObjectiveWithData(QuasiNewtonObjective* obj,
                    const SimpleMat<T>& A,
                    const SimpleVec<T>& b,
                    SimpleDenseMat<T>& X)
    : detail::objectives::QNWithData(obj->C, obj->D, obj->fit_intercept)
  {
  }
};

/**
 * @brief Minimize the given `raft::solver::quasi_newton::ObjectiveWithData` using
 * the Limited-Memory Broyden-Fletcher-Goldfarb-Shanno algorithm. This algorithm
 * estimates the inverse of the Hessian matrix, minimizing the memory footprint from
 * the original BFGS algorithm by maintaining only a subset of the update history.
 *
 * @tparam T
 * @tparam Function
 * @param param
 * @param f
 * @param x
 * @param fx
 * @param k
 * @param workspace
 * @param stream
 * @param verbosity
 * @return
 */
template <typename T, typename Function>
OPT_RETCODE lbfgs_minimize(raft::handle_t& handle,
                           const LBFGSParam<T>& param,
                           Function& f,      // function to minimize
                           SimpleVec<T>& x,  // initial point, holds result
                           T& fx,            // output function value
                           int* k)
{  // output iterations
  rmm::device_uvector<T> tmp(detail::lbfgs_workspace_size(param, x.len), handle.get_stream());
  SimpleVec<T> workspace(tmp.data(), tmp.size());
  return detail::min_lbfgs(param, f, x, fx, k, workspace, handle.get_stream(), 0);
}

/**
 * @brief Minimize the given `ObjectiveWithData` using the Orthant-wise
 * Limited-Memory Quasi-Newton algorithm, an L-BFGS variant for fitting
 * models with lasso (l1) penalties, enabling it to exploit the sparsity
 * of the models.
 *
 * @tparam T
 * @tparam Function
 * @param param
 * @param f
 * @param l1_penalty
 * @param pg_limit
 * @param x
 * @param fx
 * @param k
 * @return
 */
template <typename T, typename Function>
OPT_RETCODE owl_minimize(raft::handle_t& handle,
                         const LBFGSParam<T>& param,
                         Function& f,
                         const T l1_penalty,
                         const int pg_limit,
                         SimpleVec<T>& x,
                         T& fx,
                         int* k)
{
  rmm::device_uvector<T> tmp(detail::owlqn_workspace_size(opt_param, x.len), stream);
  SimpleVec<T> workspace(tmp.data(), tmp.size());
  return detail::min_owlqn(
    param, f, l1_penalty, pg_limit, x, fx, k, workspace, handle.get_stream(), 0);
}

/**
 * @brief Simple wrapper function that chooses the quasi-newton solver to use
 * based on the presence of the L1 penalty term.
 * @tparam T
 * @tparam LossFunction
 * @param handle
 * @param x
 * @param fx
 * @param num_iters
 * @param loss
 * @param l1
 * @param opt_param
 * @return
 */
template <typename T, typename LossFunction>
inline int minimize(const raft::handle_t& handle,
                    SimpleVec<T>& x,
                    T* fx,
                    int* num_iters,
                    LossFunction& loss,
                    const T l1,
                    const LBFGSParam<T>& opt_param,
                    cudaStream_t stream,
                    const int verbosity = 0)
{
  return detail::qn_minimize(handle, x, fx, num_iters, loss, l1, opt_param, handle.get_stream(), 0);
}
}  // namespace raft::solver::quasi_newton