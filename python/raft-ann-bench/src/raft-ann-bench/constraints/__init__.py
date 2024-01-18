#
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

DTYPE_SIZES = {"float": 4, "half": 2, "fp8": 1}


def raft_cagra_build_constraints(params, dims):
    if "graph_degree" in params and "intermediate_graph_degree" in params:
        return params["graph_degree"] <= params["intermediate_graph_degree"]
    return True


def raft_ivf_pq_build_constraints(params, dims):
    if "pq_dim" in params:
        return params["pq_dim"] <= dims
    return True


def raft_ivf_pq_search_constraints(params, build_params, k, batch_size):
    ret = True
    if "internalDistanceDtype" in params and "smemLutDtype" in params:
        ret = (
            DTYPE_SIZES[params["smemLutDtype"]]
            <= DTYPE_SIZES[params["internalDistanceDtype"]]
        )

    if "nlist" in build_params and "nprobe" in params:
        ret = ret and build_params["nlist"] >= params["nprobe"]
    return ret


def raft_cagra_search_constraints(params, build_params, k, batch_size):
    ret = True
    if "itopk" in params:
        ret = ret and params["itopk"] >= k
    return ret


def hnswlib_search_constraints(params, build_params, k, batch_size):
    if "ef" in params:
        return params["ef"] >= k


def faiss_gpu_ivf_pq_build_constraints(params, dims):
    ret = True
    # M must be defined
    ret = params["M"] <= dims and dims % params["M"] == 0
    if "use_raft" in params and params["use_raft"]:
        return ret
    pq_bits = 8
    if "bitsPerCode" in params:
        pq_bits = params["bitsPerCode"]
    lookup_table_size = 4
    if "useFloat16" in params and params["useFloat16"]:
        lookup_table_size = 2
    # FAISS constraint to check if lookup table fits in shared memory
    # for now hard code maximum shared memory per block to 49 kB (the value for A100 and V100)
    return ret and lookup_table_size * params["M"] * (2 ** pq_bits) <= 49152


def faiss_gpu_ivf_pq_search_constraints(params, build_params, k, batch_size):
    ret = True
    if "nlist" in build_params and "nprobe" in params:
        ret = ret and build_params["nlist"] >= params["nprobe"]
    return ret
