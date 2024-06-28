#
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

DTYPE_SIZES = {"float": 4, "half": 2, "fp8": 1}


def raft_cagra_build_constraints(params, dims):
    if "graph_degree" in params and "intermediate_graph_degree" in params:
        return params["graph_degree"] <= params["intermediate_graph_degree"]
    return True


def raft_ivf_pq_build_constraints(params, dims):
    if "pq_dim" in params:
        return params["pq_dim"] <= dims
    return True

def diskann_build_constraints(params, dims):
    ret = True
    if "cagra_graph_degree" in params:
        ret = params["R"] <= params["cagra_graph_degree"] and params["cagra_graph_degree"] <= params["cagra_intermediate_graph_degree"]
    return ret

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


def diskann_search_constraints(params, build_params, k, batch_size):
    return params["L_search"] >= k