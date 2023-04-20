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

copyright_notice = """
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

/*
 * NOTE: this file is generated by 00_generate.py
 *
 * Make changes there and run in this directory:
 *
 * > python 00_generate.py
 *
 */

"""

ext_headers = [
    "raft/neighbors/brute_force-ext.cuh",
    "raft/distance/distance-ext.cuh",
    "raft/distance/detail/pairwise_matrix/dispatch-ext.cuh",
    "raft/matrix/detail/select_k-ext.cuh",
    "raft/neighbors/ball_cover-ext.cuh",
    "raft/spatial/knn/detail/fused_l2_knn-ext.cuh",
    "raft/distance/fused_l2_nn-ext.cuh",
    "raft/neighbors/ivf_pq-ext.cuh",
    "raft/neighbors/ivf_flat-ext.cuh",
    "raft/neighbors/refine-ext.cuh",
    "raft/neighbors/detail/ivf_flat_search-ext.cuh",
    "raft/neighbors/detail/selection_faiss-ext.cuh",
    "raft/linalg/detail/coalesced_reduction-ext.cuh",
    "raft/spatial/knn/detail/ball_cover/registers-ext.cuh",
]

for ext_header in ext_headers:
    header = ext_header.replace("-ext", "")

    path = (
        header
        .replace("/", "_")
        .replace(".cuh", ".cu")
        .replace(".hpp", ".cpp")
    )

    with open(path, "w") as f:
        f.write(copyright_notice)
        f.write(f"#include <{header}>\n")

    # For in CMakeLists.txt
    print(f"test/ext_headers/{path}")
