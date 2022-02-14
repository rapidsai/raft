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

};  // end of namespace raft::random