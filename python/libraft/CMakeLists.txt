# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
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

include(../../rapids_config.cmake)

project(
  libraft-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX
)

# Check if raft is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python raft package.
find_package(raft "${RAPIDS_VERSION}")

if(raft_FOUND)
  return()
endif()

unset(raft_FOUND)

# --- CUDA --- #
set(CUDA_STATIC_RUNTIME ON)
set(CUDA_STATIC_MATH_LIBRARIES OFF)

# --- RAFT ---#
set(BUILD_TESTS OFF)
set(BUILD_PRIMS_BENCH OFF)
set(RAFT_COMPILE_DYNAMIC_ONLY ON)
set(RAFT_COMPILE_LIBRARY ON)

add_subdirectory(../../cpp raft-cpp)

# assumes libraft.so is installed 2 levels deep, e.g. site-packages/libraft/lib64/libraft.so
set_property(
  TARGET raft_lib
  PROPERTY INSTALL_RPATH
            "$ORIGIN/../../nvidia/cublas/lib"
            "$ORIGIN/../../nvidia/curand/lib"
            "$ORIGIN/../../nvidia/cusolver/lib"
            "$ORIGIN/../../nvidia/cusparse/lib"
            "$ORIGIN/../../nvidia/nvjitlink/lib"
)
