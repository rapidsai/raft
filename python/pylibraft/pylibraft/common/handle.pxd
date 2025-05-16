#
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.vector cimport vector

from rmm.librmm.cuda_stream_pool cimport cuda_stream_pool
from rmm.librmm.cuda_stream_view cimport cuda_stream_view


# Keeping `handle_t` around for backwards compatibility at the
# cython layer but users are encourage to switch to device_resources
cdef extern from "raft/core/handle.hpp" namespace "raft" nogil:
    cdef cppclass handle_t:
        handle_t() except +
        handle_t(cuda_stream_view stream_view) except +
        handle_t(cuda_stream_view stream_view,
                 shared_ptr[cuda_stream_pool] stream_pool) except +
        cuda_stream_view get_stream() except +
        void sync_stream() except +


cdef extern from "raft/core/device_resources.hpp" namespace "raft" nogil:
    cdef cppclass device_resources:
        device_resources() except +
        device_resources(cuda_stream_view stream_view) except +
        device_resources(cuda_stream_view stream_view,
                         shared_ptr[cuda_stream_pool] stream_pool) except +
        cuda_stream_view get_stream() except +
        void sync_stream() except +

cdef class DeviceResources:
    cdef unique_ptr[device_resources] c_obj
    cdef shared_ptr[cuda_stream_pool] stream_pool
    cdef int n_streams

cdef extern from "raft/core/device_resources_snmg.hpp" namespace "raft":
    cdef cppclass device_resources_snmg(device_resources):
        device_resources_snmg() except +
        device_resources_snmg(const vector[int]& device_ids) except +
        device_resources_snmg(const device_resources_snmg&) except +

cdef class DeviceResourcesSNMG:
    cdef unique_ptr[device_resources_snmg] c_obj
    cdef object device_ids
