/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <raft/core/operators.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/detail/scatter.cuh>

namespace raft {

/**
 * @brief Performs scatter operation based on the input indexing array
 * @tparam DataT data type whose array gets scattered
 * @tparam IdxT indexing type
 * @tparam TPB threads-per-block in the final kernel launched
 * @tparam Lambda the device-lambda performing a unary operation on the loaded
 * data before it gets scattered
 * @param out the output array
 * @param in the input array
 * @param idx the indexing array
 * @param len number of elements in the input array
 * @param stream cuda stream where to launch work
 * @param op the device-lambda with signature `DataT func(DataT, IdxT);`. This
 * will be applied to every element before scattering it to the right location.
 * The second param in this method will be the destination index.
 */
template <typename DataT, typename IdxT, typename Lambda = raft::identity_op, int TPB = 256>
void scatter(DataT* out,
             const DataT* in,
             const IdxT* idx,
             IdxT len,
             cudaStream_t stream,
             Lambda op = raft::identity_op())
{
  if (len <= 0) return;
  constexpr size_t DataSize   = sizeof(DataT);
  constexpr size_t IdxSize    = sizeof(IdxT);
  constexpr size_t MaxPerElem = DataSize > IdxSize ? DataSize : IdxSize;
  size_t bytes                = len * MaxPerElem;
  if (16 / MaxPerElem && bytes % 16 == 0) {
    detail::scatterImpl<DataT, 16 / MaxPerElem, Lambda, IdxT, TPB>(out, in, idx, len, op, stream);
  } else if (8 / MaxPerElem && bytes % 8 == 0) {
    detail::scatterImpl<DataT, 8 / MaxPerElem, Lambda, IdxT, TPB>(out, in, idx, len, op, stream);
  } else if (4 / MaxPerElem && bytes % 4 == 0) {
    detail::scatterImpl<DataT, 4 / MaxPerElem, Lambda, IdxT, TPB>(out, in, idx, len, op, stream);
  } else if (2 / MaxPerElem && bytes % 2 == 0) {
    detail::scatterImpl<DataT, 2 / MaxPerElem, Lambda, IdxT, TPB>(out, in, idx, len, op, stream);
  } else if (1 / MaxPerElem) {
    detail::scatterImpl<DataT, 1 / MaxPerElem, Lambda, IdxT, TPB>(out, in, idx, len, op, stream);
  } else {
    detail::scatterImpl<DataT, 1, Lambda, IdxT, TPB>(out, in, idx, len, op, stream);
  }
}

}  // namespace raft
