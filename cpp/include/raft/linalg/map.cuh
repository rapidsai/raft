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
 * @brief Map a function over zero or more input mdspans.
 *
 * Usage example:
 * @code{.cpp}
 *  auto input = raft::make_device_vector<int>(res, n);
 *  ... fill input ..
 *  auto squares = raft::make_device_vector<int>(res, n);
 *  raft::linalg::map_offset(res, squares.view(), raft::sq_op(), input.view());
 * @endcode
 *
 * @tparam OutType data-type of the result (device_mdspan)
 * @tparam Func the device-lambda performing the actual operation
 * @tparam InTypes data-types of the inputs (device_mdspan)
 *
 * @param[in] res raft::device_resources
 * @param[out] out the output of the map operation (device_mdspan)
 * @param[in] f device lambda
 * @param[in] ins the inputs (each of the same size as the output) (device_mdspan)
 */
template <typename OutType,
          typename Func,
          typename... InTypes,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InTypes...>>
void map(const raft::device_resources& res, OutType out, Func op, InTypes... ins)
{
  return detail::map<false>(res, out, op, ins...);
}

/**
 *  @brief Map a function over zero-based flat index (element offset) and zero or more inputs.
 *
 * Usage example:
 * @code{.cpp}
 *  auto squares = raft::make_device_vector<int>(handle, n);
 *  raft::linalg::map_offset(handle, squares.view(), raft::sq_op());
 * @endcode
 *
 * @tparam OutType data-type of the result (device_mdspan)
 * @tparam Func the device-lambda performing the actual operation
 * @tparam InTypes data-types of the inputs (device_mdspan)
 *
 * @param[in] res raft::device_resources
 * @param[out] out the output of the map operation (device_mdspan)
 * @param[in] f device lambda
 * @param[in] ins the inputs (each of the same size as the output) (device_mdspan)
 */
template <typename OutType,
          typename Func,
          typename... InTypes,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InTypes...>>
void map_offset(const raft::device_resources& res, OutType out, Func op, InTypes... ins)
{
  return detail::map<true>(res, out, op, ins...);
}

/** @} */  // end of map

}  // namespace raft::linalg

#endif
