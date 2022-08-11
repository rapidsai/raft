# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

set(pylibraft_version 22.08.00)

include(../../fetch_rapids.cmake)

project(
  pylibraft
  VERSION ${pylibraft_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C
            CXX)

option(FIND_RAFT_CPP "Search for existing RAFT C++ installations before defaulting to local files"
       OFF)

# If the user requested it we attempt to find RAFT.
if(FIND_RAFT_CPP)
    find_package(raft ${pylibraft_version} REQUIRED COMPONENTS distance)
    if(NOT TARGET raft::raft_distance_lib)
        message(FATAL_ERROR "Building against a preexisting libraft library requires the distance component of that library to have been built!")
    endif()
else()
  set(raft_FOUND OFF)
endif()

if(NOT raft_FOUND)
  # TODO: This will not be necessary once we upgrade to CMake 3.22, which will
  # pull in the required languages for the C++ project even if this project
  # does not require those languges.
  include(rapids-cuda)
  rapids_cuda_init_architectures(pylibraft)
  enable_language(CUDA)
  # Since pylibraft only enables CUDA optionally we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` including.
  include("${CMAKE_PROJECT_pylibraft_INCLUDE}")

  set(BUILD_TESTS OFF)
  set(BUILD_BENCH OFF)
  set(RAFT_COMPILE_LIBRARIES OFF)
  set(RAFT_COMPILE_DIST_LIBRARY ON)
  add_subdirectory(../../cpp raft-cpp)

  # When building the C++ libraries from source we must copy
  # libraft_distance.so alongside the pairwise_distance Cython library.
  install(TARGETS raft_distance_lib DESTINATION pylibraft/distance)
endif()

include(rapids-cython)
rapids_cython_init()

add_subdirectory(pylibraft/distance)
