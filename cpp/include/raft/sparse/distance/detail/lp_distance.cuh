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

#pragma once

#include "common.hpp"

#include <raft/core/operators.cuh>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/csr.hpp>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <limits.h>

#include <algorithm>
#include <nvfunctional>

namespace raft {
namespace sparse {
namespace distance {
namespace detail {

template <typename value_idx = int,
          typename value_t   = float,
          typename product_f,
          typename accum_f,
          typename write_f>
void unexpanded_lp_distances(value_t* out_dists,
                             const distances_config_t<value_idx, value_t>* config_,
                             product_f product_func,
                             accum_f accum_func,
                             write_f write_func)
{
  rmm::device_uvector<value_idx> coo_rows(std::max(config_->b_nnz, config_->a_nnz),
                                          resource::get_cuda_stream(config_->handle));

  raft::sparse::convert::csr_to_coo(config_->b_indptr,
                                    config_->b_nrows,
                                    coo_rows.data(),
                                    config_->b_nnz,
                                    resource::get_cuda_stream(config_->handle));

  balanced_coo_pairwise_generalized_spmv<value_idx, value_t>(
    out_dists, *config_, coo_rows.data(), product_func, accum_func, write_func);

  raft::sparse::convert::csr_to_coo(config_->a_indptr,
                                    config_->a_nrows,
                                    coo_rows.data(),
                                    config_->a_nnz,
                                    resource::get_cuda_stream(config_->handle));

  balanced_coo_pairwise_generalized_spmv_rev<value_idx, value_t>(
    out_dists, *config_, coo_rows.data(), product_func, accum_func, write_func);
}

/**
 * Computes L1 distances for sparse input. This does not have
 * an equivalent expanded form, so it is only executed in
 * an unexpanded form.
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx = int, typename value_t = float>
class l1_unexpanded_distances_t : public distances_t<value_t> {
 public:
  l1_unexpanded_distances_t(const distances_config_t<value_idx, value_t>& config) : config_(&config)
  {
  }

  void compute(value_t* out_dists)
  {
    unexpanded_lp_distances<value_idx, value_t>(
      out_dists, config_, raft::absdiff_op(), raft::add_op(), raft::atomic_add_op());
  }

 private:
  const distances_config_t<value_idx, value_t>* config_;
};

template <typename value_idx = int, typename value_t = float>
class l2_unexpanded_distances_t : public distances_t<value_t> {
 public:
  l2_unexpanded_distances_t(const distances_config_t<value_idx, value_t>& config) : config_(&config)
  {
  }

  void compute(value_t* out_dists)
  {
    unexpanded_lp_distances<value_idx, value_t>(
      out_dists, config_, raft::sqdiff_op(), raft::add_op(), raft::atomic_add_op());
  }

 protected:
  const distances_config_t<value_idx, value_t>* config_;
};

template <typename value_idx = int, typename value_t = float>
class l2_sqrt_unexpanded_distances_t : public l2_unexpanded_distances_t<value_idx, value_t> {
 public:
  l2_sqrt_unexpanded_distances_t(const distances_config_t<value_idx, value_t>& config)
    : l2_unexpanded_distances_t<value_idx, value_t>(config)
  {
  }

  void compute(value_t* out_dists)
  {
    l2_unexpanded_distances_t<value_idx, value_t>::compute(out_dists);

    uint64_t n = (uint64_t)this->config_->a_nrows * (uint64_t)this->config_->b_nrows;
    // Sqrt Post-processing
    raft::linalg::unaryOp<value_t>(
      out_dists,
      out_dists,
      n,
      [] __device__(value_t input) {
        int neg = input < 0 ? -1 : 1;
        return raft::sqrt(abs(input) * neg);
      },
      resource::get_cuda_stream(this->config_->handle));
  }
};

template <typename value_idx = int, typename value_t = float>
class linf_unexpanded_distances_t : public distances_t<value_t> {
 public:
  explicit linf_unexpanded_distances_t(const distances_config_t<value_idx, value_t>& config)
    : config_(&config)
  {
  }

  void compute(value_t* out_dists)
  {
    unexpanded_lp_distances<value_idx, value_t>(
      out_dists, config_, raft::absdiff_op(), raft::max_op(), raft::atomic_max_op());
  }

 private:
  const distances_config_t<value_idx, value_t>* config_;
};

template <typename value_idx = int, typename value_t = float>
class canberra_unexpanded_distances_t : public distances_t<value_t> {
 public:
  explicit canberra_unexpanded_distances_t(const distances_config_t<value_idx, value_t>& config)
    : config_(&config)
  {
  }

  void compute(value_t* out_dists)
  {
    unexpanded_lp_distances<value_idx, value_t>(
      out_dists,
      config_,
      [] __device__(value_t a, value_t b) {
        value_t d = fabs(a) + fabs(b);

        // deal with potential for 0 in denominator by
        // forcing 1/0 instead
        return ((d != 0) * fabs(a - b)) / (d + (d == 0));
      },
      raft::add_op(),
      raft::atomic_add_op());
  }

 private:
  const distances_config_t<value_idx, value_t>* config_;
};

template <typename value_idx = int, typename value_t = float>
class lp_unexpanded_distances_t : public distances_t<value_t> {
 public:
  explicit lp_unexpanded_distances_t(const distances_config_t<value_idx, value_t>& config,
                                     value_t p_)
    : config_(&config), p(p_)
  {
  }

  void compute(value_t* out_dists)
  {
    unexpanded_lp_distances<value_idx, value_t>(
      out_dists,
      config_,
      raft::compose_op(raft::pow_const_op<value_t>(p), raft::sub_op()),
      raft::add_op(),
      raft::atomic_add_op());

    uint64_t n         = (uint64_t)this->config_->a_nrows * (uint64_t)this->config_->b_nrows;
    value_t one_over_p = value_t{1} / p;
    raft::linalg::unaryOp<value_t>(out_dists,
                                   out_dists,
                                   n,
                                   raft::pow_const_op<value_t>(one_over_p),
                                   resource::get_cuda_stream(config_->handle));
  }

 private:
  const distances_config_t<value_idx, value_t>* config_;
  value_t p;
};

template <typename value_idx = int, typename value_t = float>
class hamming_unexpanded_distances_t : public distances_t<value_t> {
 public:
  explicit hamming_unexpanded_distances_t(const distances_config_t<value_idx, value_t>& config)
    : config_(&config)
  {
  }

  void compute(value_t* out_dists)
  {
    unexpanded_lp_distances<value_idx, value_t>(
      out_dists, config_, raft::notequal_op(), raft::add_op(), raft::atomic_add_op());

    uint64_t n     = (uint64_t)config_->a_nrows * (uint64_t)config_->b_nrows;
    value_t n_cols = 1.0 / config_->a_ncols;
    raft::linalg::unaryOp<value_t>(out_dists,
                                   out_dists,
                                   n,
                                   raft::mul_const_op<value_t>(n_cols),
                                   resource::get_cuda_stream(config_->handle));
  }

 private:
  const distances_config_t<value_idx, value_t>* config_;
};

template <typename value_idx = int, typename value_t = float>
class jensen_shannon_unexpanded_distances_t : public distances_t<value_t> {
 public:
  explicit jensen_shannon_unexpanded_distances_t(
    const distances_config_t<value_idx, value_t>& config)
    : config_(&config)
  {
  }

  void compute(value_t* out_dists)
  {
    unexpanded_lp_distances<value_idx, value_t>(
      out_dists,
      config_,
      [] __device__(value_t a, value_t b) {
        value_t m   = 0.5f * (a + b);
        bool a_zero = a == 0;
        bool b_zero = b == 0;

        value_t x = (!a_zero * m) / (a_zero + a);
        value_t y = (!b_zero * m) / (b_zero + b);

        bool x_zero = x == 0;
        bool y_zero = y == 0;

        return (-a * (!x_zero * log(x + x_zero))) + (-b * (!y_zero * log(y + y_zero)));
      },
      raft::add_op(),
      raft::atomic_add_op());

    uint64_t n = (uint64_t)this->config_->a_nrows * (uint64_t)this->config_->b_nrows;
    raft::linalg::unaryOp<value_t>(
      out_dists,
      out_dists,
      n,
      [=] __device__(value_t input) { return raft::sqrt(0.5 * input); },
      resource::get_cuda_stream(config_->handle));
  }

 private:
  const distances_config_t<value_idx, value_t>* config_;
};

template <typename value_idx = int, typename value_t = float>
class kl_divergence_unexpanded_distances_t : public distances_t<value_t> {
 public:
  explicit kl_divergence_unexpanded_distances_t(
    const distances_config_t<value_idx, value_t>& config)
    : config_(&config)
  {
  }

  void compute(value_t* out_dists)
  {
    rmm::device_uvector<value_idx> coo_rows(std::max(config_->b_nnz, config_->a_nnz),
                                            resource::get_cuda_stream(config_->handle));

    raft::sparse::convert::csr_to_coo(config_->b_indptr,
                                      config_->b_nrows,
                                      coo_rows.data(),
                                      config_->b_nnz,
                                      resource::get_cuda_stream(config_->handle));

    balanced_coo_pairwise_generalized_spmv<value_idx, value_t>(
      out_dists,
      *config_,
      coo_rows.data(),
      [] __device__(value_t a, value_t b) { return a * log(a / b); },
      raft::add_op(),
      raft::atomic_add_op());

    uint64_t n = (uint64_t)this->config_->a_nrows * (uint64_t)this->config_->b_nrows;
    raft::linalg::unaryOp<value_t>(out_dists,
                                   out_dists,
                                   n,
                                   raft::mul_const_op<value_t>(0.5),
                                   resource::get_cuda_stream(config_->handle));
  }

 private:
  const distances_config_t<value_idx, value_t>* config_;
};

};  // END namespace detail
};  // END namespace distance
};  // END namespace sparse
};  // END namespace raft
