#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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


import sys

from libcpp cimport bool
from libcpp.string cimport string


cdef extern from "raft/core/logger.hpp" namespace "raft" nogil:
    cdef cppclass logger:
        @staticmethod
        logger& get(string& name)
        void set_level(int level)
        void set_pattern(const string& pattern)
        void set_callback(void(*callback)(int, char*))
        void set_flush(void(*flush)())
        bool should_log_for(int level) const
        int get_level() const
        string get_pattern() const
        void flush()


cdef extern from "raft/core/logger.hpp" nogil:
    string RAFT_NAME
    void RAFT_LOG_TRACE(const char* fmt, ...)
    void RAFT_LOG_DEBUG(const char* fmt, ...)
    void RAFT_LOG_INFO(const char* fmt, ...)
    void RAFT_LOG_WARN(const char* fmt, ...)
    void RAFT_LOG_ERROR(const char* fmt, ...)
    void RAFT_LOG_CRITICAL(const char* fmt, ...)

    cdef string RAFT_NAME
    cdef int RAFT_LEVEL_TRACE
    cdef int RAFT_LEVEL_DEBUG
    cdef int RAFT_LEVEL_INFO
    cdef int RAFT_LEVEL_WARN
    cdef int RAFT_LEVEL_ERROR
    cdef int RAFT_LEVEL_CRITICAL
    cdef int RAFT_LEVEL_OFF
