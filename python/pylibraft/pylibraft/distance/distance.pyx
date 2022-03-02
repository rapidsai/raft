#
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

from pylibraft.common.handle cimport handle_t
from .distance_type import DistanceType
from pylibraft.common.mdarray cimport make_device_matrix, device_matrix

cdef pairwise_distance():

    cdef handle_t handle
    cdef device_matrix[int] hellp = make_device_matrix[int](5, 10, handle.get_stream())
