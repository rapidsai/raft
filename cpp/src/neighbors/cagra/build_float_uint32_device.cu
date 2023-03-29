/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/specializations/ivf_flat.cuh>
#include <raft/neighbors/specializations/ivf_pq.cuh>
namespace raft::neighbors::experimental::cagra {

template auto
build<float,
      uint32_t,
      host_device_accessor<std::experimental::default_accessor<float>, memory_type::device>>(
  raft::device_resources const& handle,
  const index_params& params,
  mdspan<const float,
         matrix_extent<uint32_t>,
         row_major,
         host_device_accessor<std::experimental::default_accessor<float>, memory_type::device>>
    dataset) -> index<float, uint32_t>;
}  // namespace raft::neighbors::experimental::cagra
