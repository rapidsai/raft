# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

set(pylibraft_version 23.06.00)

include(../../fetch_rapids.cmake)

project(
  pylibraft
  VERSION ${pylibraft_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX
)

option(FIND_RAFT_CPP "Search for existing RAFT C++ installations before defaulting to local files"
       ON
)

option(RAFT_BUILD_WHEELS "Whether this build is generating a Python wheel." OFF)

# If the user requested it we attempt to find RAFT.
if(FIND_RAFT_CPP)
  find_package(raft ${pylibraft_version} REQUIRED COMPONENTS compiled)
  if(NOT TARGET raft::raft_lib)
    message(
      FATAL_ERROR
        "Building against a preexisting libraft library requires the compiled libraft to have been built!"
    )

  endif()
else()
  set(raft_FOUND OFF)
endif()

include(rapids-cython)

if(NOT raft_FOUND)
  # TODO: This will not be necessary once we upgrade to CMake 3.22, which will pull in the required
  # languages for the C++ project even if this project does not require those languages.
  include(rapids-cuda)
  rapids_cuda_init_architectures(pylibraft)
  enable_language(CUDA)
  # Since pylibraft only enables CUDA optionally we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` including.
  include("${CMAKE_PROJECT_pylibraft_INCLUDE}")

  set(BUILD_TESTS OFF)
  set(BUILD_PRIMS_BENCH OFF)
  set(BUILD_ANN_BENCH OFF)
  set(RAFT_COMPILE_LIBRARY ON)

  set(_exclude_from_all "")
  if(RAFT_BUILD_WHEELS)
    # Statically link dependencies if building wheels
    set(CUDA_STATIC_RUNTIME ON)
    # Don't install the raft C++ targets into wheels
    set(_exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  add_subdirectory(../../cpp raft-cpp ${_exclude_from_all})

  # When building the C++ libraries from source we must copy libraft.so alongside the
  # pairwise_distance and random Cython libraries TODO: when we have a single 'compiled' raft
  # library, we shouldn't need this
  set(cython_lib_dir pylibraft)
  install(TARGETS raft_lib DESTINATION ${cython_lib_dir})
endif()

rapids_cython_init()

add_subdirectory(pylibraft/common)
add_subdirectory(pylibraft/distance)
add_subdirectory(pylibraft/matrix)
add_subdirectory(pylibraft/neighbors)
add_subdirectory(pylibraft/random)
add_subdirectory(pylibraft/cluster)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET raft PATHS "${cython_lib_dir}")
endif()
