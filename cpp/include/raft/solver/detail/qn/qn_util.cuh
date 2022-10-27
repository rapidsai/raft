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

#include <limits>
#include <raft/core/logger.hpp>
#include <raft/util/cuda_utils.cuh>

namespace raft::solver::quasi_newton::detail {

inline bool qn_is_classification(qn_loss_type t)
{
  switch (t) {
    case QN_LOSS_LOGISTIC:
    case QN_LOSS_SOFTMAX:
    case QN_LOSS_HINGE:
    case QN_LOSS_SQ_HINGE: return true;
    default: return false;
  }
}

template <typename T>
HDI T project_orth(T x, T y)
{
  return x * y <= T(0) ? T(0) : x;
}

template <typename T>
inline bool check_convergence(
  const LBFGSParam<T>& param, const int k, const T fx, const T gnorm, std::vector<T>& fx_hist)
{
  // Positive scale factor for the stop condition
  T fmag = std::max(fx, param.epsilon);

  RAFT_LOG_DEBUG(
    "%04d: f(x)=%.8f conv.crit=%.8f (gnorm=%.8f, fmag=%.8f)", k, fx, gnorm / fmag, gnorm, fmag);
  // Convergence test -- gradient
  if (gnorm <= param.epsilon * fmag) {
    RAFT_LOG_DEBUG("Converged after %d iterations: f(x)=%.6f", k, fx);
    return true;
  }
  // Convergence test -- objective function value
  if (param.past > 0) {
    if (k >= param.past && std::abs(fx_hist[k % param.past] - fx) <= param.delta * fmag) {
      RAFT_LOG_DEBUG("Insufficient change in objective value");
      return true;
    }

    fx_hist[k % param.past] = fx;
  }
  return false;
}

/*
 * Multiplies a vector g with the inverse hessian approximation, i.e.
 * drt = - H * g,
 * e.g. to compute the new search direction for g = \nabla f(x)
 */
template <typename T>
inline int lbfgs_search_dir(const LBFGSParam<T>& param,
                            int* n_vec,
                            const int end_prev,
                            const SimpleDenseMat<T>& S,
                            const SimpleDenseMat<T>& Y,
                            const SimpleVec<T>& g,
                            const SimpleVec<T>& svec,
                            const SimpleVec<T>& yvec,
                            SimpleVec<T>& drt,
                            std::vector<T>& yhist,
                            std::vector<T>& alpha,
                            T* dev_scalar,
                            cudaStream_t stream)
{
  SimpleVec<T> sj, yj;  // mask vectors
  int end = end_prev;
  // note: update_state assigned svec, yvec to m_s[:,end], m_y[:,end]
  T ys = dot(svec, yvec, dev_scalar, stream);
  T yy = dot(yvec, yvec, dev_scalar, stream);
  RAFT_LOG_TRACE("ys=%e, yy=%e", ys, yy);
  // Skipping test:
  if (ys <= std::numeric_limits<T>::epsilon() * yy) {
    // We can land here for example if yvec == 0 (no change in the gradient,
    // g_k == g_k+1). That means the Hessian is approximately zero. We cannot
    // use the QN model to update the search dir, we just continue along the
    // previous direction.
    //
    // See eq (3.9) and Section 6 in "A limited memory algorithm for bound
    // constrained optimization" Richard H. Byrd, Peihuang Lu, Jorge Nocedal and
    // Ciyou Zhu Technical Report NAM-08 (1994) NORTHWESTERN UNIVERSITY.
    //
    // Alternative condition to skip update is: ys / (-gs) <= epsmch,
    // (where epsmch = std::numeric_limits<T>::epsilon) given in Section 5 of
    // "L-BFGS-B Fortran subroutines for large-scale bound constrained
    // optimization" Ciyou Zhu, Richard H. Byrd, Peihuang Lu and Jorge Nocedal
    // (1994).
    RAFT_LOG_DEBUG("L-BFGS WARNING: skipping update step ys=%f, yy=%f", ys, yy);
    return end;
  }
  (*n_vec)++;
  yhist[end] = ys;

  // Recursive formula to compute d = -H * g
  drt.ax(-1.0, g, stream);
  int bound = std::min(param.m, *n_vec);
  end       = (end + 1) % param.m;
  int j     = end;
  for (int i = 0; i < bound; i++) {
    j = (j + param.m - 1) % param.m;
    col_ref(S, sj, j);
    col_ref(Y, yj, j);
    alpha[j] = dot(sj, drt, dev_scalar, stream) / yhist[j];
    drt.axpy(-alpha[j], yj, drt, stream);
  }

  drt.ax(ys / yy, drt, stream);

  for (int i = 0; i < bound; i++) {
    col_ref(S, sj, j);
    col_ref(Y, yj, j);
    T beta = dot(yj, drt, dev_scalar, stream) / yhist[j];
    drt.axpy((alpha[j] - beta), sj, drt, stream);
    j = (j + 1) % param.m;
  }

  return end;
}

template <typename T>
HDI T get_pseudo_grad(T x, T dlossx, T C)
{
  if (x != 0) { return dlossx + raft::sgn(x) * C; }
  T dplus = dlossx + C;
  T dmins = dlossx - C;
  if (dmins > T(0)) return dmins;
  if (dplus < T(0)) return dplus;
  return T(0);
}

template <typename T>
struct op_project {
  T scal;
  op_project(T s) : scal(s) {}

  HDI T operator()(const T x, const T y) const { return project_orth(x, scal * y); }
};

template <typename T>
struct op_pseudo_grad {
  T l1;
  op_pseudo_grad(const T lam) : l1(lam) {}

  HDI T operator()(const T x, const T dlossx) const { return get_pseudo_grad(x, dlossx, l1); }
};

};  // namespace  raft::solver::quasi_newton::detail
