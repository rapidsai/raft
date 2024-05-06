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

#include <raft/neighbors/ann_mg_types.hpp>
#include <raft/neighbors/detail/ann_mg.cuh>

namespace raft::neighbors::mg {

template <typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const detail::ann_mg_index<cagra::index<T, IdxT>, T, IdxT>& index,
               const std::string& filename)
{
  mg::detail::serialize(handle, index, filename);
}

template <typename T, typename IdxT>
detail::ann_mg_index<cagra::index<T, IdxT>, T, IdxT> deserialize_cagra(const raft::resources& handle,
                                                                       const std::string& filename)
{
  return mg::detail::deserialize_cagra<T, IdxT>(handle, filename);
}

template <typename T, typename IdxT>
detail::ann_mg_index<cagra::index<T, IdxT>, T, IdxT> distribute_cagra(const raft::resources& handle,
                                                                      const std::vector<int>& dev_list,
                                                                      const std::string& filename)
{
  return mg::detail::distribute_cagra<T, IdxT>(handle, dev_list, filename);
}

}  // namespace raft::neighbors::mg