#
# Copyright (c) 2020, NVIDIA CORPORATION.
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


from libcpp.memory cimport shared_ptr
cimport raft.common.cuda


cdef extern from "raft/mr/device/allocator.hpp" \
        namespace "raft::mr::device" nogil:
    cdef cppclass allocator:
        pass

cdef extern from "raft/handle.hpp" namespace "raft" nogil:
    cdef cppclass handle_t:
        handle_t() except +
        handle_t(int ns) except +
        void set_stream(raft.common.cuda._Stream s) except +
        void set_device_allocator(shared_ptr[allocator] a) except +
        raft.common.cuda._Stream get_stream() except +
        int get_num_internal_streams() except +
