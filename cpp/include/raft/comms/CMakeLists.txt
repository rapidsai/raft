#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(comms LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT NCCL_PATH)
    find_package(NCCL REQUIRED)
else()
    message("-- Manually set NCCL PATH to ${NCCL_PATH}")
    set(NCCL_INCLUDE_DIRS ${NCCL_PATH}/include)
    set(NCCL_LIBRARIES ${NCCL_PATH}/lib/libnccl.so)
endif(NOT NCCL_PATH)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(UCX)
include_directories(${UCX_INCLUDE_DIRS})

include_directories( ${NCCL_INCLUDE_DIRS} )
list(APPEND RAFT_LINK_LIBRARIES ${NCCL_LIBRARIES})