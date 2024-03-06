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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/sparse/csr.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/distance/detail/ip_distance.cuh>
#include <raft/spatial/knn/knn.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <nvfunctional>

namespace raft {
namespace sparse {
namespace distance {
namespace detail {

// @TODO: Move this into sparse prims (coo_norm)
template <typename value_idx, typename value_t>
RAFT_KERNEL compute_row_norm_kernel(value_t* out,
                                    const value_idx* __restrict__ coo_rows,
                                    const value_t* __restrict__ data,
                                    value_idx nnz)
{
  value_idx i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nnz) { atomicAdd(&out[coo_rows[i]], data[i] * data[i]); }
}

template <typename value_idx, typename value_t>
RAFT_KERNEL compute_row_sum_kernel(value_t* out,
                                   const value_idx* __restrict__ coo_rows,
                                   const value_t* __restrict__ data,
                                   value_idx nnz)
{
  value_idx i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nnz) { atomicAdd(&out[coo_rows[i]], data[i]); }
}

template <typename value_idx, typename value_t, typename expansion_f>
RAFT_KERNEL compute_euclidean_warp_kernel(value_t* __restrict__ C,
                                          const value_t* __restrict__ Q_sq_norms,
                                          const value_t* __restrict__ R_sq_norms,
                                          value_idx n_rows,
                                          value_idx n_cols,
                                          expansion_f expansion_func)
{
  std::size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  value_idx i     = tid / n_cols;
  value_idx j     = tid % n_cols;

  if (i >= n_rows || j >= n_cols) return;

  value_t dot = C[(size_t)i * n_cols + j];

  // e.g. Euclidean expansion func = -2.0 * dot + q_norm + r_norm
  value_t val = expansion_func(dot, Q_sq_norms[i], R_sq_norms[j]);

  // correct for small instabilities
  C[(size_t)i * n_cols + j] = val * (fabs(val) >= 0.0001);
}

template <typename value_idx, typename value_t>
RAFT_KERNEL compute_correlation_warp_kernel(value_t* __restrict__ C,
                                            const value_t* __restrict__ Q_sq_norms,
                                            const value_t* __restrict__ R_sq_norms,
                                            const value_t* __restrict__ Q_norms,
                                            const value_t* __restrict__ R_norms,
                                            value_idx n_rows,
                                            value_idx n_cols,
                                            value_idx n)
{
  std::size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  value_idx i     = tid / n_cols;
  value_idx j     = tid % n_cols;

  if (i >= n_rows || j >= n_cols) return;

  value_t dot  = C[(size_t)i * n_cols + j];
  value_t Q_l1 = Q_norms[i];
  value_t R_l1 = R_norms[j];

  value_t Q_l2 = Q_sq_norms[i];
  value_t R_l2 = R_sq_norms[j];

  value_t numer   = n * dot - (Q_l1 * R_l1);
  value_t Q_denom = n * Q_l2 - (Q_l1 * Q_l1);
  value_t R_denom = n * R_l2 - (R_l1 * R_l1);

  value_t val = 1 - (numer / raft::sqrt(Q_denom * R_denom));

  // correct for small instabilities
  C[(size_t)i * n_cols + j] = val * (fabs(val) >= 0.0001);
}

template <typename value_idx, typename value_t, int tpb = 256, typename expansion_f>
void compute_euclidean(value_t* C,
                       const value_t* Q_sq_norms,
                       const value_t* R_sq_norms,
                       value_idx n_rows,
                       value_idx n_cols,
                       cudaStream_t stream,
                       expansion_f expansion_func)
{
  int blocks = raft::ceildiv<size_t>((size_t)n_rows * n_cols, tpb);
  compute_euclidean_warp_kernel<<<blocks, tpb, 0, stream>>>(
    C, Q_sq_norms, R_sq_norms, n_rows, n_cols, expansion_func);
}

template <typename value_idx, typename value_t, int tpb = 256, typename expansion_f>
void compute_l2(value_t* out,
                const value_idx* Q_coo_rows,
                const value_t* Q_data,
                value_idx Q_nnz,
                const value_idx* R_coo_rows,
                const value_t* R_data,
                value_idx R_nnz,
                value_idx m,
                value_idx n,
                cudaStream_t stream,
                expansion_f expansion_func)
{
  rmm::device_uvector<value_t> Q_sq_norms(m, stream);
  rmm::device_uvector<value_t> R_sq_norms(n, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(Q_sq_norms.data(), 0, Q_sq_norms.size() * sizeof(value_t)));
  RAFT_CUDA_TRY(cudaMemsetAsync(R_sq_norms.data(), 0, R_sq_norms.size() * sizeof(value_t)));

  compute_row_norm_kernel<<<raft::ceildiv(Q_nnz, tpb), tpb, 0, stream>>>(
    Q_sq_norms.data(), Q_coo_rows, Q_data, Q_nnz);
  compute_row_norm_kernel<<<raft::ceildiv(R_nnz, tpb), tpb, 0, stream>>>(
    R_sq_norms.data(), R_coo_rows, R_data, R_nnz);

  compute_euclidean(out, Q_sq_norms.data(), R_sq_norms.data(), m, n, stream, expansion_func);
}

template <typename value_idx, typename value_t, int tpb = 256>
void compute_correlation(value_t* C,
                         const value_t* Q_sq_norms,
                         const value_t* R_sq_norms,
                         const value_t* Q_norms,
                         const value_t* R_norms,
                         value_idx n_rows,
                         value_idx n_cols,
                         value_idx n,
                         cudaStream_t stream)
{
  int blocks = raft::ceildiv<size_t>((size_t)n_rows * n_cols, tpb);
  compute_correlation_warp_kernel<<<blocks, tpb, 0, stream>>>(
    C, Q_sq_norms, R_sq_norms, Q_norms, R_norms, n_rows, n_cols, n);
}

template <typename value_idx, typename value_t, int tpb = 256>
void compute_corr(value_t* out,
                  const value_idx* Q_coo_rows,
                  const value_t* Q_data,
                  value_idx Q_nnz,
                  const value_idx* R_coo_rows,
                  const value_t* R_data,
                  value_idx R_nnz,
                  value_idx m,
                  value_idx n,
                  value_idx n_cols,
                  cudaStream_t stream)
{
  // sum_sq for std dev
  rmm::device_uvector<value_t> Q_sq_norms(m, stream);
  rmm::device_uvector<value_t> R_sq_norms(n, stream);

  // sum for mean
  rmm::device_uvector<value_t> Q_norms(m, stream);
  rmm::device_uvector<value_t> R_norms(n, stream);

  RAFT_CUDA_TRY(cudaMemsetAsync(Q_sq_norms.data(), 0, Q_sq_norms.size() * sizeof(value_t)));
  RAFT_CUDA_TRY(cudaMemsetAsync(R_sq_norms.data(), 0, R_sq_norms.size() * sizeof(value_t)));

  RAFT_CUDA_TRY(cudaMemsetAsync(Q_norms.data(), 0, Q_norms.size() * sizeof(value_t)));
  RAFT_CUDA_TRY(cudaMemsetAsync(R_norms.data(), 0, R_norms.size() * sizeof(value_t)));

  compute_row_norm_kernel<<<raft::ceildiv(Q_nnz, tpb), tpb, 0, stream>>>(
    Q_sq_norms.data(), Q_coo_rows, Q_data, Q_nnz);
  compute_row_norm_kernel<<<raft::ceildiv(R_nnz, tpb), tpb, 0, stream>>>(
    R_sq_norms.data(), R_coo_rows, R_data, R_nnz);

  compute_row_sum_kernel<<<raft::ceildiv(Q_nnz, tpb), tpb, 0, stream>>>(
    Q_norms.data(), Q_coo_rows, Q_data, Q_nnz);
  compute_row_sum_kernel<<<raft::ceildiv(R_nnz, tpb), tpb, 0, stream>>>(
    R_norms.data(), R_coo_rows, R_data, R_nnz);

  compute_correlation(out,
                      Q_sq_norms.data(),
                      R_sq_norms.data(),
                      Q_norms.data(),
                      R_norms.data(),
                      m,
                      n,
                      n_cols,
                      stream);
}

/**
 * L2 distance using the expanded form: sum(x_k)^2 + sum(y_k)^2 - 2 * sum(x_k * y_k)
 * The expanded form is more efficient for sparse data.
 */
template <typename value_idx = int, typename value_t = float>
class l2_expanded_distances_t : public distances_t<value_t> {
 public:
  explicit l2_expanded_distances_t(const distances_config_t<value_idx, value_t>& config)
    : config_(&config), ip_dists(config)
  {
  }

  void compute(value_t* out_dists)
  {
    ip_dists.compute(out_dists);

    value_idx* b_indices = ip_dists.b_rows_coo();
    value_t* b_data      = ip_dists.b_data_coo();

    rmm::device_uvector<value_idx> search_coo_rows(config_->a_nnz,
                                                   resource::get_cuda_stream(config_->handle));
    raft::sparse::convert::csr_to_coo(config_->a_indptr,
                                      config_->a_nrows,
                                      search_coo_rows.data(),
                                      config_->a_nnz,
                                      resource::get_cuda_stream(config_->handle));

    compute_l2(out_dists,
               search_coo_rows.data(),
               config_->a_data,
               config_->a_nnz,
               b_indices,
               b_data,
               config_->b_nnz,
               config_->a_nrows,
               config_->b_nrows,
               resource::get_cuda_stream(config_->handle),
               [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) {
                 return -2 * dot + q_norm + r_norm;
               });
  }

  ~l2_expanded_distances_t() = default;

 protected:
  const distances_config_t<value_idx, value_t>* config_;
  ip_distances_t<value_idx, value_t> ip_dists;
};

/**
 * L2 sqrt distance performing the sqrt operation after the distance computation
 * The expanded form is more efficient for sparse data.
 */
template <typename value_idx = int, typename value_t = float>
class l2_sqrt_expanded_distances_t : public l2_expanded_distances_t<value_idx, value_t> {
 public:
  explicit l2_sqrt_expanded_distances_t(const distances_config_t<value_idx, value_t>& config)
    : l2_expanded_distances_t<value_idx, value_t>(config)
  {
  }

  void compute(value_t* out_dists) override
  {
    l2_expanded_distances_t<value_idx, value_t>::compute(out_dists);
    // Sqrt Post-processing
    raft::linalg::unaryOp<value_t>(
      out_dists,
      out_dists,
      this->config_->a_nrows * this->config_->b_nrows,
      [] __device__(value_t input) {
        int neg = input < 0 ? -1 : 1;
        return raft::sqrt(abs(input) * neg);
      },
      resource::get_cuda_stream(this->config_->handle));
  }

  ~l2_sqrt_expanded_distances_t() = default;
};

template <typename value_idx, typename value_t>
class correlation_expanded_distances_t : public distances_t<value_t> {
 public:
  explicit correlation_expanded_distances_t(const distances_config_t<value_idx, value_t>& config)
    : config_(&config), ip_dists(config)
  {
  }

  void compute(value_t* out_dists)
  {
    ip_dists.compute(out_dists);

    value_idx* b_indices = ip_dists.b_rows_coo();
    value_t* b_data      = ip_dists.b_data_coo();

    rmm::device_uvector<value_idx> search_coo_rows(config_->a_nnz,
                                                   resource::get_cuda_stream(config_->handle));
    raft::sparse::convert::csr_to_coo(config_->a_indptr,
                                      config_->a_nrows,
                                      search_coo_rows.data(),
                                      config_->a_nnz,
                                      resource::get_cuda_stream(config_->handle));

    compute_corr(out_dists,
                 search_coo_rows.data(),
                 config_->a_data,
                 config_->a_nnz,
                 b_indices,
                 b_data,
                 config_->b_nnz,
                 config_->a_nrows,
                 config_->b_nrows,
                 config_->b_ncols,
                 resource::get_cuda_stream(config_->handle));
  }

  ~correlation_expanded_distances_t() = default;

 protected:
  const distances_config_t<value_idx, value_t>* config_;
  ip_distances_t<value_idx, value_t> ip_dists;
};

/**
 * Cosine distance using the expanded form: 1 - ( sum(x_k * y_k) / (sqrt(sum(x_k)^2) *
 * sqrt(sum(y_k)^2))) The expanded form is more efficient for sparse data.
 */
template <typename value_idx = int, typename value_t = float>
class cosine_expanded_distances_t : public distances_t<value_t> {
 public:
  explicit cosine_expanded_distances_t(const distances_config_t<value_idx, value_t>& config)
    : config_(&config), workspace(0, resource::get_cuda_stream(config.handle)), ip_dists(config)
  {
  }

  void compute(value_t* out_dists)
  {
    ip_dists.compute(out_dists);

    value_idx* b_indices = ip_dists.b_rows_coo();
    value_t* b_data      = ip_dists.b_data_coo();

    rmm::device_uvector<value_idx> search_coo_rows(config_->a_nnz,
                                                   resource::get_cuda_stream(config_->handle));
    raft::sparse::convert::csr_to_coo(config_->a_indptr,
                                      config_->a_nrows,
                                      search_coo_rows.data(),
                                      config_->a_nnz,
                                      resource::get_cuda_stream(config_->handle));

    compute_l2(out_dists,
               search_coo_rows.data(),
               config_->a_data,
               config_->a_nnz,
               b_indices,
               b_data,
               config_->b_nnz,
               config_->a_nrows,
               config_->b_nrows,
               resource::get_cuda_stream(config_->handle),
               [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) {
                 value_t norms = raft::sqrt(q_norm) * raft::sqrt(r_norm);
                 // deal with potential for 0 in denominator by forcing 0/1 instead
                 value_t cos = ((norms != 0) * dot) / ((norms == 0) + norms);

                 // flip the similarity when both rows are 0
                 bool both_empty = (q_norm == 0) && (r_norm == 0);
                 return 1 - ((!both_empty * cos) + both_empty);
               });
  }

  ~cosine_expanded_distances_t() = default;

 private:
  const distances_config_t<value_idx, value_t>* config_;
  rmm::device_uvector<char> workspace;
  ip_distances_t<value_idx, value_t> ip_dists;
};

/**
 * Hellinger distance using the expanded form: sqrt(1 - sum(sqrt(x_k) * sqrt(y_k)))
 * The expanded form is more efficient for sparse data.
 *
 * This distance computation modifies A and B by computing a sqrt
 * and then performing a `pow(x, 2)` to convert it back. Because of this,
 * it is possible that the values in A and B might differ slightly
 * after this is invoked.
 */
template <typename value_idx = int, typename value_t = float>
class hellinger_expanded_distances_t : public distances_t<value_t> {
 public:
  explicit hellinger_expanded_distances_t(const distances_config_t<value_idx, value_t>& config)
    : config_(&config), workspace(0, resource::get_cuda_stream(config.handle))
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
      [] __device__(value_t a, value_t b) { return raft::sqrt(a) * raft::sqrt(b); },
      raft::add_op(),
      raft::atomic_add_op());

    raft::linalg::unaryOp<value_t>(
      out_dists,
      out_dists,
      config_->a_nrows * config_->b_nrows,
      [=] __device__(value_t input) {
        // Adjust to replace NaN in sqrt with 0 if input to sqrt is negative
        bool rectifier = (1 - input) > 0;
        return raft::sqrt(rectifier * (1 - input));
      },
      resource::get_cuda_stream(config_->handle));
  }

  ~hellinger_expanded_distances_t() = default;

 private:
  const distances_config_t<value_idx, value_t>* config_;
  rmm::device_uvector<char> workspace;
};

template <typename value_idx = int, typename value_t = float>
class russelrao_expanded_distances_t : public distances_t<value_t> {
 public:
  explicit russelrao_expanded_distances_t(const distances_config_t<value_idx, value_t>& config)
    : config_(&config), workspace(0, resource::get_cuda_stream(config.handle)), ip_dists(config)
  {
  }

  void compute(value_t* out_dists)
  {
    ip_dists.compute(out_dists);

    value_t n_cols     = config_->a_ncols;
    value_t n_cols_inv = 1.0 / n_cols;
    raft::linalg::unaryOp<value_t>(
      out_dists,
      out_dists,
      config_->a_nrows * config_->b_nrows,
      [=] __device__(value_t input) { return (n_cols - input) * n_cols_inv; },
      resource::get_cuda_stream(config_->handle));

    auto exec_policy  = rmm::exec_policy(resource::get_cuda_stream(config_->handle));
    auto diags        = thrust::counting_iterator<value_idx>(0);
    value_idx b_nrows = config_->b_nrows;
    thrust::for_each(exec_policy, diags, diags + config_->a_nrows, [=] __device__(value_idx input) {
      out_dists[input * b_nrows + input] = 0.0;
    });
  }

  ~russelrao_expanded_distances_t() = default;

 private:
  const distances_config_t<value_idx, value_t>* config_;
  rmm::device_uvector<char> workspace;
  ip_distances_t<value_idx, value_t> ip_dists;
};

};  // END namespace detail
};  // END namespace distance
};  // END namespace sparse
};  // END namespace raft
