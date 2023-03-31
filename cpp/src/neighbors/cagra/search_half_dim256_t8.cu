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

// File generated with make_search_cores.sh

#include "raft/neighbors/detail/cagra/search_multi_cta.cuh"
#include "raft/neighbors/detail/cagra/search_multi_kernel.cuh"
#include "raft/neighbors/detail/cagra/search_single_cta.cuh"

namespace raft::neighbors::experimental::cagra::detail::single_cta_search {
template struct search<8, 256, half, uint32_t, float>;
}
namespace raft::neighbors::experimental::cagra::detail::multi_cta_search {
template struct search<8, 256, half, uint32_t, float>;
}
namespace raft::neighbors::experimental::cagra::detail::multi_kernel_search {
template struct search<8, 256, half, uint32_t, float>;
}
