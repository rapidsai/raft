/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <raft/util/detail/popc.cuh>
namespace raft {

/**
 * @brief Count the number of bits that are set to 1 in a vector.
 *
 * @tparam value_t the value type of the vector.
 * @tparam index_t the index type of vector and scalar.
 *
 * @param[in] res RAFT handle for managing expensive resources
 * @param[in] values Device vector view containing the values to be processed.
 * @param[in] max_len Host scalar view to store the Maximum number of bits to count.
 * @param[out] counter Device scalar view to store the number of bits that are set to 1.
 */
template <typename value_t, typename index_t>
void popc(const raft::resources& res,
          device_vector_view<value_t, index_t> values,
          raft::host_scalar_view<index_t> max_len,
          raft::device_scalar_view<index_t> counter)
{
  detail::popc(res, values, max_len, counter);
}

}  // namespace raft
