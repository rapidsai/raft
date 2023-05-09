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

#include <raft/matrix/detail/select_warpsort.cuh>  // matrix::detail::select::warpsort::warp_sort_distributed

/*
 * This header file is a bit of an ugly duckling. The type dummy_block_sort is
 * needed by both ivf_pq_search.cuh and ivf_pq_compute_similarity.cuh.
 *
 * I have decided to move it to it's own header file, which is overkill. Perhaps
 * there is a nicer solution.
 *
 */

namespace raft::neighbors::ivf_pq::detail {

template <typename T, typename IdxT>
struct dummy_block_sort_t {
  using queue_t = matrix::detail::select::warpsort::warp_sort_distributed<WarpSize, true, T, IdxT>;
  template <typename... Args>
  __device__ dummy_block_sort_t(int k, Args...){};
};

}  // namespace raft::neighbors::ivf_pq::detail
