/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#ifndef __MAP_H
#define __MAP_H

#pragma once

#include "detail/map.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>

namespace raft::linalg {

/**
 * @defgroup map Mapping ops
 * @{
 */

/**
 * @brief Map a function over zero or more input mdspans of the same size.
 *
 * The algorithm applied on `k` inputs can be described in a following pseudo-code:
 * @code
 *  for (auto i: [0 ... out.size()]) {
 *    out[i] = f(in_0[i], in_1[i], ..., in_k[i])
 *  }
 * @endcode
 *
 * _Performance note_: when possible, this function loads the argument arrays and stores the output
 * array using vectorized cuda load/store instructions. The size of the vectorization depends on the
 * size of the largest input/output element type and on the alignment of all pointers.
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/device_mdarray.hpp>
 *  #include <raft/core/device_resources.hpp>
 *  #include <raft/core/operators.hpp>
 *  #include <raft/linalg/map.cuh>
 *
 *  auto input = raft::make_device_vector<int>(res, n);
 *  ... fill input ..
 *  auto squares = raft::make_device_vector<int>(res, n);
 *  raft::linalg::map_offset(res, squares.view(), raft::sq_op{}, input.view());
 * @endcode
 *
 * @tparam OutType data-type of the result (device_mdspan)
 * @tparam Func the device-lambda performing the actual operation
 * @tparam InTypes data-types of the inputs (device_mdspan)
 *
 * @param[in] res raft::device_resources
 * @param[out] out the output of the map operation (device_mdspan)
 * @param[in] f device lambda
 *                 (InTypes::value_type xs...) -> OutType::value_type
 * @param[in] ins the inputs (each of the same size as the output) (device_mdspan)
 */
template <typename OutType,
          typename Func,
          typename... InTypes,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InTypes...>>
void map(const raft::device_resources& res, OutType out, Func f, InTypes... ins)
{
  return detail::map<false>(res, out, f, ins...);
}

/**
 *  @brief Map a function over zero-based flat index (element offset) and zero or more inputs.
 *
 * The algorithm applied on `k` inputs can be described in a following pseudo-code:
 * @code
 *  for (auto i: [0 ... out.size()]) {
 *    out[i] = f(i, in_0[i], in_1[i], ..., in_k[i])
 *  }
 * @endcode
 *
 * _Performance note_: when possible, this function loads the argument arrays and stores the output
 * array using vectorized cuda load/store instructions. The size of the vectorization depends on the
 * size of the largest input/output element type and on the alignment of all pointers.
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/device_mdarray.hpp>
 *  #include <raft/core/device_resources.hpp>
 *  #include <raft/core/operators.hpp>
 *  #include <raft/linalg/map.cuh>
 *
 *  auto squares = raft::make_device_vector<int>(handle, n);
 *  raft::linalg::map_offset(res, squares.view(), raft::sq_op{});
 * @endcode
 *
 * @tparam OutType data-type of the result (device_mdspan)
 * @tparam Func the device-lambda performing the actual operation
 * @tparam InTypes data-types of the inputs (device_mdspan)
 *
 * @param[in] res raft::device_resources
 * @param[out] out the output of the map operation (device_mdspan)
 * @param[in] f device lambda
 *                 (auto offset, InTypes::value_type xs...) -> OutType::value_type
 * @param[in] ins the inputs (each of the same size as the output) (device_mdspan)
 */
template <typename OutType,
          typename Func,
          typename... InTypes,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InTypes...>>
void map_offset(const raft::device_resources& res, OutType out, Func f, InTypes... ins)
{
  return detail::map<true>(res, out, f, ins...);
}

/** @} */  // end of map

}  // namespace raft::linalg

#endif
