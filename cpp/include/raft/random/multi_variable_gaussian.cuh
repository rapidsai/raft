/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <raft/core/resources.hpp>
#include <raft/random/random_types.hpp>

#include <rmm/resource_ref.hpp>

namespace raft::random {

/**
 * \defgroup multi_variable_gaussian Compute multi-variable Gaussian
 * @{
 */

template <typename ValueType>
void multi_variable_gaussian(raft::resources const& handle,
                             rmm::device_async_resource_ref mem_resource,
                             std::optional<raft::device_vector_view<const ValueType, int>> x,
                             raft::device_matrix_view<ValueType, int, raft::col_major> P,
                             raft::device_matrix_view<ValueType, int, raft::col_major> X,
                             const multi_variable_gaussian_decomposition_method method)
{
  detail::compute_multi_variable_gaussian_impl(handle, mem_resource, x, P, X, method);
}

template <typename ValueType>
void multi_variable_gaussian(raft::resources const& handle,
                             std::optional<raft::device_vector_view<const ValueType, int>> x,
                             raft::device_matrix_view<ValueType, int, raft::col_major> P,
                             raft::device_matrix_view<ValueType, int, raft::col_major> X,
                             const multi_variable_gaussian_decomposition_method method)
{
  detail::compute_multi_variable_gaussian_impl(
    handle, rmm::mr::get_current_device_resource(), x, P, X, method);
}

/** @} */

};  // end of namespace raft::random

#endif
