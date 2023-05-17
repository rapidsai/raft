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
#include <raft/core/resources.hpp>

namespace raft::linalg {

/**
 * @brief CUDA version of map
 *
 * Note: This call is deprecated, please use `map` from the same file.
 *
 * @tparam InType data-type upon which the math operation will be performed
 * @tparam MapOp the device-lambda performing the actual operation
 * @tparam TPB threads-per-block in the final kernel launched
 * @tparam Args additional parameters
 * @tparam OutType data-type in which the result will be stored
 * @param out the output of the map operation (assumed to be a device pointer)
 * @param len number of elements in the input array
 * @param map the device-lambda
 * @param stream cuda-stream where to launch this kernel
 * @param in the input array
 * @param args additional input arrays
 */
template <typename InType,
          typename MapOp,
          typename IdxType = std::uint32_t,
          int TPB          = 256,
          typename OutType = InType,
          typename... Args>
[[deprecated("Use function `map` from the same file")]] void map_k(
  OutType* out, IdxType len, MapOp map, cudaStream_t stream, const InType* in, Args... args)
{
  return detail::map<false>(stream, out, len, map, in, args...);
}

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
 *  #include <raft/core/resources.hpp>
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
 * @param[in] res raft::resources
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
void map(const raft::resources& res, OutType out, Func f, InTypes... ins)
{
  return detail::map<false>(res, out, f, ins...);
}

/**
 * @brief Map a function over one mdspan.
 *
 * @tparam InType1 data-type of the input (device_mdspan)
 * @tparam OutType data-type of the result (device_mdspan)
 * @tparam Func the device-lambda performing the actual operation
 *
 * @param[in] res raft::resources
 * @param[in] in1 the input (the same size as the output) (device_mdspan)
 * @param[out] out the output of the map operation (device_mdspan)
 * @param[in] f device lambda
 *                 (InType1::value_type x) -> OutType::value_type
 */
template <typename InType1,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1>>
void map(const raft::resources& res, InType1 in1, OutType out, Func f)
{
  return detail::map<false>(res, out, f, in1);
}

/**
 * @brief Map a function over two mdspans.
 *
 * @tparam InType1 data-type of the input (device_mdspan)
 * @tparam InType2 data-type of the input (device_mdspan)
 * @tparam OutType data-type of the result (device_mdspan)
 * @tparam Func the device-lambda performing the actual operation
 *
 * @param[in] res raft::resources
 * @param[in] in1 the input (the same size as the output) (device_mdspan)
 * @param[in] in2 the input (the same size as the output) (device_mdspan)
 * @param[out] out the output of the map operation (device_mdspan)
 * @param[in] f device lambda
 *                 (InType1::value_type x1, InType2::value_type x2) -> OutType::value_type
 */
template <typename InType1,
          typename InType2,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1, InType2>>
void map(const raft::resources& res, InType1 in1, InType2 in2, OutType out, Func f)
{
  return detail::map<false>(res, out, f, in1, in2);
}

/**
 * @brief Map a function over three mdspans.
 *
 * @tparam InType1 data-type of the input 1 (device_mdspan)
 * @tparam InType2 data-type of the input 2 (device_mdspan)
 * @tparam InType3 data-type of the input 3 (device_mdspan)
 * @tparam OutType data-type of the result (device_mdspan)
 * @tparam Func the device-lambda performing the actual operation
 *
 * @param[in] res raft::resources
 * @param[in] in1 the input 1 (the same size as the output) (device_mdspan)
 * @param[in] in2 the input 2 (the same size as the output) (device_mdspan)
 * @param[in] in3 the input 3 (the same size as the output) (device_mdspan)
 * @param[out] out the output of the map operation (device_mdspan)
 * @param[in] f device lambda
 *   (InType1::value_type x1, InType2::value_type x2, InType3::value_type x3) -> OutType::value_type
 */
template <typename InType1,
          typename InType2,
          typename InType3,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1, InType2, InType3>>
void map(const raft::resources& res, InType1 in1, InType2 in2, InType3 in3, OutType out, Func f)
{
  return detail::map<false>(res, out, f, in1, in2, in3);
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
 *  #include <raft/core/resources.hpp>
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
 * @param[in] res raft::resources
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
void map_offset(const raft::resources& res, OutType out, Func f, InTypes... ins)
{
  return detail::map<true>(res, out, f, ins...);
}

/**
 * @brief Map a function over zero-based flat index (element offset) and one mdspan.
 *
 * @tparam InType1 data-type of the input (device_mdspan)
 * @tparam OutType data-type of the result (device_mdspan)
 * @tparam Func the device-lambda performing the actual operation
 *
 * @param[in] res raft::resources
 * @param[in] in1 the input (the same size as the output) (device_mdspan)
 * @param[out] out the output of the map operation (device_mdspan)
 * @param[in] f device lambda
 *                 (auto offset, InType1::value_type x) -> OutType::value_type
 */
template <typename InType1,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1>>
void map_offset(const raft::resources& res, InType1 in1, OutType out, Func f)
{
  return detail::map<true>(res, out, f, in1);
}

/**
 * @brief Map a function over zero-based flat index (element offset) and two mdspans.
 *
 * @tparam InType1 data-type of the input (device_mdspan)
 * @tparam InType2 data-type of the input (device_mdspan)
 * @tparam OutType data-type of the result (device_mdspan)
 * @tparam Func the device-lambda performing the actual operation
 *
 * @param[in] res raft::resources
 * @param[in] in1 the input (the same size as the output) (device_mdspan)
 * @param[in] in2 the input (the same size as the output) (device_mdspan)
 * @param[out] out the output of the map operation (device_mdspan)
 * @param[in] f device lambda
 *    (auto offset, InType1::value_type x1, InType2::value_type x2) -> OutType::value_type
 */
template <typename InType1,
          typename InType2,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1, InType2>>
void map_offset(const raft::resources& res, InType1 in1, InType2 in2, OutType out, Func f)
{
  return detail::map<true>(res, out, f, in1, in2);
}

/**
 * @brief Map a function over zero-based flat index (element offset) and three mdspans.
 *
 * @tparam InType1 data-type of the input 1 (device_mdspan)
 * @tparam InType2 data-type of the input 2 (device_mdspan)
 * @tparam InType3 data-type of the input 3 (device_mdspan)
 * @tparam OutType data-type of the result (device_mdspan)
 * @tparam Func the device-lambda performing the actual operation
 *
 * @param[in] res raft::resources
 * @param[in] in1 the input 1 (the same size as the output) (device_mdspan)
 * @param[in] in2 the input 2 (the same size as the output) (device_mdspan)
 * @param[in] in3 the input 3 (the same size as the output) (device_mdspan)
 * @param[out] out the output of the map operation (device_mdspan)
 * @param[in] f device lambda
 *   (auto offset, InType1::value_type x1, InType2::value_type x2, InType3::value_type x3)
 *      -> OutType::value_type
 */
template <typename InType1,
          typename InType2,
          typename InType3,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1, InType2, InType3>>
void map_offset(
  const raft::resources& res, InType1 in1, InType2 in2, InType3 in3, OutType out, Func f)
{
  return detail::map<true>(res, out, f, in1, in2, in3);
}

/** @} */  // end of map

}  // namespace raft::linalg

#endif
