# Copyright (c) 2022, NVIDIA CORPORATION.
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
#

from c_ivf_pq cimport cudaDataType_t.CUDA_R_32F as CUDA_R_32F
from c_ivf_pq cimport cudaDataType_t.CUDA_R_16F as CUDA_R_16F
from c_ivf_pq cimport cudaDataType_t.CUDA_R_8U as CUDA_R_8U

from c_ivf_pq cimport codebook_gen.PER_SUBSPACE as PER_SUBSPACE
from c_ivf_pq cimport codebook_gen.PER_CLUSTER as PER_CLUSTER

from c_ivf_pq cimport index_params
from c_ivf_pq cimport search_params