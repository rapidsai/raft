# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

set(raft_dask_version 24.02.00)

include(../../fetch_rapids.cmake)
include(rapids-cpm)
include(rapids-cuda)
rapids_cuda_init_architectures(raft-dask-python)

project(
  raft-dask-python
  VERSION ${raft_dask_version}
  LANGUAGES CXX CUDA
)

option(FIND_RAFT_CPP "Search for existing RAFT C++ installations before defaulting to local files"
       OFF
)

# If the user requested it we attempt to find RAFT.
if(FIND_RAFT_CPP)
  find_package(raft ${raft_dask_version} REQUIRED COMPONENTS distributed)
else()
  set(raft_FOUND OFF)
endif()

if(NOT raft_FOUND)
  rapids_cpm_init()
  include(cmake/thirdparty/get_ucxx.cmake)

  find_package(ucx REQUIRED)

  # raft-dask doesn't actually use raft libraries, it just needs the headers, so we can turn off all
  # library compilation and we don't need to install anything here.
  set(BUILD_TESTS OFF)
  set(BUILD_ANN_BENCH OFF)
  set(BUILD_PRIMS_BENCH OFF)
  set(RAFT_COMPILE_LIBRARIES OFF)
  set(RAFT_COMPILE_DIST_LIBRARY OFF)
  set(RAFT_COMPILE_NN_LIBRARY OFF)
  set(CUDA_STATIC_RUNTIME ON)
  set(RAFT_DASK_UCXX_STATIC ON)

  add_subdirectory(../../cpp raft-cpp EXCLUDE_FROM_ALL)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/cmake/find_modules)
  find_package(NCCL REQUIRED)
endif()

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(raft_dask/common)
add_subdirectory(raft_dask/include_test)
