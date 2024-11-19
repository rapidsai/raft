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

header = """/*
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

/*
 * NOTE: this file is generated by ivf_pq_compute_similarity_00_generate.py
 * Make changes there and run in this directory:
 * > python ivf_pq_compute_similarity_00_generate.py
 */
          
#include <raft/neighbors/detail/ivf_pq_compute_similarity_template.cuh>
"""

none_filter_int64 = "raft::neighbors::filtering::ivf_to_sample_filter" \
                    "<int64_t COMMA raft::neighbors::filtering::none_ivf_sample_filter>"
none_filter_int32 = "raft::neighbors::filtering::ivf_to_sample_filter" \
                    "<uint32_t COMMA raft::neighbors::filtering::none_ivf_sample_filter>"
bitset_filter32 = "raft::neighbors::filtering::ivf_to_sample_filter" \
                  "<uint32_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA uint32_t>>"
bitset_filter64 = "raft::neighbors::filtering::ivf_to_sample_filter" \
                  "<int64_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>"

types = dict(
    half_fp8_false=("half", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>", none_filter_int64),
    half_fp8_true=("half", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>", none_filter_int64),
    half_half=("half", "half", none_filter_int64),
    float_half=("float", "half", none_filter_int64),
    float_float= ("float", "float", none_filter_int64),
    float_fp8_false=("float", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>", none_filter_int64),
    float_fp8_true=("float", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>", none_filter_int64),
    half_fp8_false_filt32=("half", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>", none_filter_int32),
    half_fp8_true_filt32=("half", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>", none_filter_int32),
    half_half_filt32=("half", "half", none_filter_int32),
    float_half_filt32=("float", "half", none_filter_int32),
    float_float_filt32= ("float", "float", none_filter_int32),
    float_fp8_false_filt32=("float", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>", none_filter_int32),
    float_fp8_true_filt32=("float", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>", none_filter_int32),
    half_fp8_false_bitset32=("half", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>", bitset_filter32),
    half_fp8_true_bitset32=("half", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>", bitset_filter32),
    half_half_bitset32=("half", "half", bitset_filter32),
    float_half_bitset32=("float", "half", bitset_filter32),
    float_float_bitset32= ("float", "float", bitset_filter32),
    float_fp8_false_bitset32=("float", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>", bitset_filter32),
    float_fp8_true_bitset32=("float", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>", bitset_filter32),
    half_fp8_false_bitset64=("half", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>", bitset_filter64),
    half_fp8_true_bitset64=("half", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>", bitset_filter64),
    half_half_bitset64=("half", "half", bitset_filter64),
    float_half_bitset64=("float", "half", bitset_filter64),
    float_float_bitset64= ("float", "float", bitset_filter64),
    float_fp8_false_bitset64=("float", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>", bitset_filter64),
    float_fp8_true_bitset64=("float", "raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>", bitset_filter64)
)

for path_key, (OutT, LutT, FilterT) in types.items():
    path = f"ivf_pq_compute_similarity_{path_key}.cu"
    with open(path, "w") as f:
        f.write(header)
        f.write(f"instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select({OutT}, {LutT}, {FilterT});\n")
    print(f"src/neighbors/detail/{path}")