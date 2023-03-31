#!/bin/bash

# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

for max_dataset_dim in 128 256 512 1024 ; do
    for dtype in float half int8_t uint8_t ; do
	for team_size in 4 8 16 32 ; do
	    if [ $max_dataset_dim -gt 128 ] && [ $team_size -lt 8 ]; then
		continue
	    fi
	    if [ $max_dataset_dim -gt 256 ] && [ $team_size -lt 16 ]; then
		continue
	    fi
	    if [ $max_dataset_dim -gt 512 ] && [ $team_size -lt 32 ]; then
		continue
	    fi
	    echo "/*
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
 
// File generated with make_search_cores.sh

#include \"raft/neighbors/detail/cagra/search_single_cta.cuh\"
#include \"raft/neighbors/detail/cagra/search_multi_cta.cuh\"
#include \"raft/neighbors/detail/cagra/search_multi_kernel.cuh\"

namespace raft::neighbors::experimental::cagra::detail::single_cta_search {
  template struct search<${team_size}, ${max_dataset_dim}, ${dtype}, uint32_t, float>;
}
namespace raft::neighbors::experimental::cagra::detail::multi_cta_search {
  template struct search<${team_size}, ${max_dataset_dim}, ${dtype}, uint32_t, float>;
}
namespace raft::neighbors::experimental::cagra::detail::multi_kernel_search {
  template struct search<${team_size}, ${max_dataset_dim}, ${dtype}, uint32_t, float>;
}
" > search_${dtype}_dim${max_dataset_dim}_t${team_size}.cu
    done
    done
done
