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


def raft_ivf_pq_search_validator(params, k, batch_size):
    if "internalDistanceDtype" in params and "smemLutDtype" in params:
        return (
            DTYPE_SIZES[params["smemLutDtype"]]
            >= DTYPE_SIZES[params["internalDistanceDtype"]]
        )


def raft_cagra_search_validator(params, k, batch_size):
    if "itopk" in params:
        return params["itopk"] >= k
