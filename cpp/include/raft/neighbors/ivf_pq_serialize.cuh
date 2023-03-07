/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "detail/ivf_pq_serialize.cuh"

namespace raft::neighbors::ivf_pq {

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index IVF-PQ index
 *
 */
template <typename IdxT>
void serialize(raft::device_resources const& handle_, std::ostream& os, const index<IdxT>& index)
{
  detail::serialize(handle_, os, index);
}

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-PQ index
 *
 */
template <typename IdxT>
void serialize(raft::device_resources const& handle_,
               const std::string& filename,
               const index<IdxT>& index)
{
  detail::serialize(handle_, filename, index);
}

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[in] index IVF-PQ index
 *
 */
template <typename IdxT>
index<IdxT> deserialize(raft::device_resources const& handle_, std::istream& is)
{
  return detail::deserialize<IdxT>(handle_, is);
}

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index IVF-PQ index
 *
 */
template <typename IdxT>
index<IdxT> deserialize(raft::device_resources const& handle_, const std::string& filename)
{
  return detail::deserialize<IdxT>(handle_, filename);
}

}  // namespace raft::neighbors::ivf_pq
