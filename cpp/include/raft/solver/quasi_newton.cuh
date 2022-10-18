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

#include <raft/solver/solver_types.hpp>
#include <raft/solver/detail/qn/qn.cuh>

namespace raft::solver::quasi_newton {

    /**
     * The follow loss functions are wrapped so they will be included in the docs
     * @tparam T
     */

    /**
     *
     * @tparam T
     */
    template <typename T>
    struct AbsLoss : detail::objectives::AbsLoss<T> {
        AbsLoss(const raft::handle_t &handle, int D, bool has_bias)
                : detail::objectives::AbsLoss(handle, D, has_bias) {}
    }

    /**
     *
     * @tparam T
     */
    template <typename T>
    struct HingeLoss : detail::objectives::HingeLoss<T> {
        HingeLoss(const raft::handle_t &handle, int D, bool has_bias)
                : detail::objectives::HingeLoss(handle, D, has_bias) {}
    }

    /**
     *
     * @tparam T
     */
    template <typename T>
    struct LogisticLoss : detail::objectives::LogisticLoss<T> {
        LogisticLoss(const raft::handle_t &handle, int D, bool has_bias)
                : detail::objectives::LogisticLoss(handle, D, has_bias) {}
    }

    /**
     *
     * @tparam T
     */
    template <typename T>
    struct SqHingeLoss : detail::objectives::SqHingeLoss<T> {
        SqHingeLoss(const raft::handle_t &handle, int D, bool has_bias)
                : detail::objectives::SqHingeLoss(handle, D, has_bias) {}
    }


    template <typename T>
    struct EpsInsHingeLoss : QNLinearBase<T, EpsInsHingeLoss<T>> {
        typedef QNLinearBase<T, EpsInsHingeLoss<T>> Super;
        EpsInsHingeLoss(const raft::handle_t& handle, int D, bool has_bias, T sensitivity)
                : Super(handle, D, 1, has_bias), lz{sensitivity}, dlz{sensitivity}
        {
        }

    using raft::solver::quasi_newton::detail::objectives::SqEpsInsHingeLoss;
    using raft::solver::quasi_newton::detail::objectives::EpsInsHingeLoss;
    using raft::solver::quasi_newton::detail::LBFGSParam;

    /**
     *
     * @tparam T
     * @tparam Loss
     * @tparam Reg
     */
    template <typename T, class Loss, class Reg>
    class RegularizedQN : public detail::objectives::RegularizedQN<T, Loss, Reg> {
        RegularizedQN(Loss* loss, Reg* reg): detail::objectives::RegularizedQN(loss, reg) {}
    };

    /**
     *
     * @tparam T
     * @tparam Loss
     */
    template <typename T, class Loss>
    struct QNLinearBase : detail::objectives::QNLinearBase<T, Loss> {
        QNLinearBase(const raft::handle_t &handle, int D, int C, bool fit_intercept)
                : detail::objectives::QNLinearBase(C, D, fit_intercept) {}
    }

    using raft::solver::quasi_newton::detail::objectives::Softmax;
    using raft::solver::quasi_newton::detail::objectives::QNWithData;
    using raft::solver::quasi_newton::detail::objectives::QNLinearBase;

    template <typename T, typename Function>
    OPT_RETCODE lbfgs_minimize(const LBFGSParam<T>& param,
                                 Function& f,              // function to minimize
                                 SimpleVec<T>& x,          // initial point, holds result
                                 T& fx,                    // output function value
                                 int* k,                   // output iterations
                                 SimpleVec<T>& workspace,  // scratch space
                                 cudaStream_t stream,
                                 int verbosity = 0)

    template <typename T, typename Function>
    OPT_RETCODE owl_minimize(const LBFGSParam<T>& param,
                                    Function& f,
                                    const T l1_penalty,
                                    const int pg_limit,
                                    SimpleVec<T>& x,
                                    T& fx,
                                    int* k) {

    }

    template <typename T, typename LossFunction>
    int qn_minimize(const raft::handle_t& handle,
                           T *x,
                           T* fx,
                           int* num_iters,
                           LossFunction& loss,
                           const T l1,
                           const detail::LBFGSParam<T>& opt_param) {

    }

}