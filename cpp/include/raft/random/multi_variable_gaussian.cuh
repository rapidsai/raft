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

#ifndef __MVG_H
#define __MVG_H

#pragma once

#include "detail/multi_variable_gaussian.cuh"

namespace raft::random {

template <typename T>
class multi_variable_gaussian : public detail::multi_variable_gaussian_impl<T> {
 public:
  // using Decomposer = typename detail::multi_variable_gaussian_impl<T>::Decomposer;
  // using detail::multi_variable_gaussian_impl<T>::Decomposer::chol_decomp;
  // using detail::multi_variable_gaussian_impl<T>::Decomposer::jacobi;
  // using detail::multi_variable_gaussian_impl<T>::Decomposer::qr;

  multi_variable_gaussian() = delete;
  multi_variable_gaussian(const raft::handle_t& handle,
                          const int dim,
                          typename detail::multi_variable_gaussian_impl<T>::Decomposer method)
    : detail::multi_variable_gaussian_impl<T>{handle, dim, method}
  {
  }

  std::size_t get_workspace_size()
  {
    return detail::multi_variable_gaussian_impl<T>::get_workspace_size();
  }

  void set_workspace(T* workarea)
  {
    detail::multi_variable_gaussian_impl<T>::set_workspace(workarea);
  }

  void give_gaussian(const int nPoints, T* P, T* X, const T* x = 0)
  {
    detail::multi_variable_gaussian_impl<T>::give_gaussian(nPoints, P, X, x);
  }

  void deinit() { detail::multi_variable_gaussian_impl<T>::deinit(); }

  ~multi_variable_gaussian() { deinit(); }
};  // end of multi_variable_gaussian

/**
 * @brief Matrix decomposition method for `compute_multi_variable_gaussian` to use.
 *
 * `compute_multi_variable_gaussian` can use any of the following methods.
 *
 * - `CHOLESKY`: Uses Cholesky decomposition on the normal equations.
 *   This may be faster than the other two methods, but less accurate.
 *
 * - `JACOBI`: Uses the singular value decomposition (SVD) computed with
 *   cuSOLVER's gesvdj algorithm, which is based on the Jacobi method
 *   (sweeps of plane rotations).  This exposes more parallelism
 *   for small and medium size matrices than the QR option below.
 *
 * - `QR`: Uses the SVD computed with cuSOLVER's gesvd algorithm,
 *   which is based on the QR algorithm.
 */
using detail::multi_variable_gaussian_decomposition_method;

template <typename ValueType>
void compute_multi_variable_gaussian(
  const raft::handle_t& handle,
  rmm::mr::device_memory_resource& mem_resource,
  std::optional<raft::device_vector_view<const ValueType, int>> x,
  raft::device_matrix_view<ValueType, int, raft::col_major> P,
  raft::device_matrix_view<ValueType, int, raft::col_major> X,
  const multi_variable_gaussian_decomposition_method method)
{
  detail::compute_multi_variable_gaussian_impl(handle, mem_resource, x, P, X, method);
}

template <typename ValueType>
void compute_multi_variable_gaussian(
  const raft::handle_t& handle,
  std::optional<raft::device_vector_view<const ValueType, int>> x,
  raft::device_matrix_view<ValueType, int, raft::col_major> P,
  raft::device_matrix_view<ValueType, int, raft::col_major> X,
  const multi_variable_gaussian_decomposition_method method)
{
  rmm::mr::device_memory_resource* mem_resource_ptr = rmm::mr::get_current_device_resource();
  RAFT_EXPECTS(mem_resource_ptr != nullptr,
               "compute_multi_variable_gaussian: "
               "rmm::mr::get_current_device_resource() returned null; "
               "please report this bug to the RAPIDS RAFT developers.");
  detail::compute_multi_variable_gaussian_impl(handle, *mem_resource_ptr, x, P, X, method);
}

};  // end of namespace raft::random

#endif
