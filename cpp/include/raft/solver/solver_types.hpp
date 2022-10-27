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

namespace raft::solver {

enum STORAGE_ORDER { COL_MAJOR = 0, ROW_MAJOR = 1 };

    enum lr_type {
  OPTIMAL,
  CONSTANT,
  INVSCALING,
  ADAPTIVE,
};

enum loss_funct {
  SQUARED,
  HINGE,
  LOG,
};

enum penalty { NONE, L1, L2, ELASTICNET };

namespace gradient_descent {
template <typename math_t>
struct sgd_params {
  int batch_size;
  int epochs;
  lr_type lr_type;
  math_t eta0;
  math_t power_t;
  loss_funct loss;
  penalty penalty;
  math_t alpha;
  math_t l1_ratio;
  bool shuffle;
  math_t tol;
  int n_iter_no_change;

  sgd_params()
    : batch_size(100),
      epochs(100),
      lr_type(lr_type::OPTIMAL),
      eta0(0.5),
      power_t(0.5),
      loss(loss_funct::SQUARED),
      penalty(penalty::L1),
      alpha(0.5),
      l1_ratio(0.2),
      shuffle(true),
      tol(1e-8),
      n_iter_no_change(5)
  {
  }
};
}  // namespace gradient_descent
namespace coordinate_descent {
template <typename math_t>
struct cd_params {
  bool normalize;   // whether to normalize the data to zero-mean and unit std
  int epochs;       // number of iterations
  loss_funct loss;  // loss function to minimize
  math_t alpha;     // l1 penalty parameter
  math_t l1_ratio;  // ratio of alpha that will be used for l1 penalty. (1 - l1_ratio) * alpha will
                    // be used for l2 penalty
  bool shuffle;     // randomly pick coordinates
  math_t tol;       // early-stopping convergence tolerance

  cd_params()
    : normalize(true),
      epochs(100),
      alpha(0.3),
      l1_ratio(0.5),
      shuffle(true),
      tol(1e-8),
      loss(loss_funct::SQRD_LOSS)
  {
  }
};
}  // namespace coordinate_descent

namespace least_angle_regression {
template <typename math_t>
struct lars_params {
  int max_iter;
  math_t eps;

  lars_params() : max_iter(500), eps(-1) {}
};
}  // namespace least_angle_regression

enum class LarsFitStatus { kOk, kCollinear, kError, kStop };

namespace quasi_newton {

/** Loss function types supported by the Quasi-Newton solvers. */
enum qn_loss_type {
    /** Logistic classification.
     *  Expected target: {0, 1}.
     */
    QN_LOSS_LOGISTIC = 0,
    /** L2 regression.
     *  Expected target: R.
     */
    QN_LOSS_SQUARED = 1,
    /** Softmax classification..
     *  Expected target: {0, 1, ...}.
     */
    QN_LOSS_SOFTMAX = 2,
    /** Hinge.
     *  Expected target: {0, 1}.
     */
    QN_LOSS_HINGE = 3,
    /** Squared-hinge.
     *  Expected target: {0, 1}.
     */
    QN_LOSS_SQ_HINGE = 4,
    /** Epsilon-insensitive.
     *  Expected target: R.
     */
    QN_LOSS_HINGE_EPS_INS = 5,
    /** Epsilon-insensitive-squared.
     *  Expected target: R.
     */
    QN_LOSS_HINGE_SQ_EPS_INS = 6,
    /** L1 regression.
     *  Expected target: R.
     */
    QN_LOSS_ABS = 7,
    /** Someone forgot to set the loss type! */
    QN_LOSS_UNKNOWN = 99
};


    struct qn_params {
  /** Loss type. */
  qn_loss_type loss;
  /** Regularization: L1 component. */
  double penalty_l1;
  /** Regularization: L2 component. */
  double penalty_l2;
  /** Convergence criteria: the threshold on the gradient. */
  double grad_tol;
  /** Convergence criteria: the threshold on the function change. */
  double change_tol;
  /** Maximum number of iterations. */
  int max_iter;
  /** Maximum number of linesearch (inner loop) iterations. */
  int linesearch_max_iter;
  /** Number of vectors approximating the hessian (l-bfgs). */
  int lbfgs_memory;
  /** Triggers extra output when greater than zero. */
  int verbose;
  /** Whether to fit the bias term. */
  bool fit_intercept;
  /**
   * Whether to divide the L1 and L2 regularization parameters by the sample size.
   *
   * Note, the defined QN loss functions normally are scaled for the sample size,
   * e.g. the average across the data rows is calculated.
   * Enabling `penalty_normalized` makes this solver's behavior compatible to those solvers,
   * which do not scale the loss functions (like sklearn.LogisticRegression()).
   */
  bool penalty_normalized;

  qn_params()
    : loss(QN_LOSS_UNKNOWN),
      penalty_l1(0),
      penalty_l2(0),
      grad_tol(1e-4),
      change_tol(1e-5),
      max_iter(1000),
      linesearch_max_iter(50),
      lbfgs_memory(5),
      verbose(0),
      fit_intercept(true),
      penalty_normalized(true)
  {
  }
};

enum LINE_SEARCH_ALGORITHM {
  LBFGS_LS_BT_ARMIJO       = 1,
  LBFGS_LS_BT              = 2,  // Default. Alias for Wolfe
  LBFGS_LS_BT_WOLFE        = 2,
  LBFGS_LS_BT_STRONG_WOLFE = 3
};

enum LINE_SEARCH_RETCODE {
  LS_SUCCESS           = 0,
  LS_INVALID_STEP_MIN  = 1,
  LS_INVALID_STEP_MAX  = 2,
  LS_MAX_ITERS_REACHED = 3,
  LS_INVALID_DIR       = 4,
  LS_INVALID_STEP      = 5
};

enum OPT_RETCODE {
  OPT_SUCCESS           = 0,
  OPT_NUMERIC_ERROR     = 1,
  OPT_LS_FAILED         = 2,
  OPT_MAX_ITERS_REACHED = 3,
  OPT_INVALID_ARGS      = 4
};

template <typename T = double>
class LBFGSParam {
 public:
  int m;      // lbfgs memory limit
  T epsilon;  // controls convergence
  int past;   // lookback for function value based convergence test
  T delta;    // controls fun val based conv test
  int max_iterations;
  int linesearch;  // see enum above
  int max_linesearch;
  T min_step;  // min. allowed step length
  T max_step;  // max. allowed step length
  T ftol;      // line  search tolerance
  T wolfe;     // wolfe parameter
  T ls_dec;    // line search decrease factor
  T ls_inc;    // line search increase factor

 public:
  LBFGSParam()
  {
    m              = 6;
    epsilon        = T(1e-5);
    past           = 0;
    delta          = T(0);
    max_iterations = 0;
    linesearch     = LBFGS_LS_BT_ARMIJO;
    max_linesearch = 20;
    min_step       = T(1e-20);
    max_step       = T(1e+20);
    ftol           = T(1e-4);
    wolfe          = T(0.9);
    ls_dec         = T(0.5);
    ls_inc         = T(2.1);
  }

  explicit LBFGSParam(const qn_params& pams) : LBFGSParam()
  {
    m       = pams.lbfgs_memory;
    epsilon = T(pams.grad_tol);
    // sometimes even number works better - to detect zig-zags;
    past           = pams.change_tol > 0 ? 10 : 0;
    delta          = T(pams.change_tol);
    max_iterations = pams.max_iter;
    max_linesearch = pams.linesearch_max_iter;
    ftol           = pams.change_tol > 0 ? T(pams.change_tol * 0.1) : T(1e-4);
  }

  inline int check_param() const
  {  // TODO exceptions
    int ret = 1;
    if (m <= 0) return ret;
    ret++;
    if (epsilon <= 0) return ret;
    ret++;
    if (past < 0) return ret;
    ret++;
    if (delta < 0) return ret;
    ret++;
    if (max_iterations < 0) return ret;
    ret++;
    if (linesearch < LBFGS_LS_BT_ARMIJO || linesearch > LBFGS_LS_BT_STRONG_WOLFE) return ret;
    ret++;
    if (max_linesearch <= 0) return ret;
    ret++;
    if (min_step < 0) return ret;
    ret++;
    if (max_step < min_step) return ret;
    ret++;
    if (ftol <= 0 || ftol >= 0.5) return ret;
    ret++;
    if (wolfe <= ftol || wolfe >= 1) return ret;
    ret++;
    return 0;
  }
};

struct LinearDims {
  bool fit_intercept;
  int C, D, dims, n_param;
  LinearDims(int C, int D, bool fit_intercept) : C(C), D(D), fit_intercept(fit_intercept)
  {
    dims    = D + fit_intercept;
    n_param = dims * C;
  }
};
}  // namespace quasi_newton

}  // namespace raft::solver
