#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

from rmm._lib.cuda_stream_view cimport cuda_stream_view


cdef extern from "thread":
    cdef cppclass cpp_thread_id "std::thread::id":
        pass

cdef extern from "thread" namespace "std::this_thread" nogil:
    cdef cpp_thread_id get_id()


cdef extern from "raft/interruptible.hpp" \
        namespace "raft::interruptible" nogil:
    cdef void synchronize(cuda_stream_view stream) except+
    cdef void cancel(cpp_thread_id tid) except+
