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

include(../../rapids_config.cmake)

# We always need CUDA for pylibraft because the raft dependency brings in a header-only cuco
# dependency that enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(pylibraft)

project(
  pylibraft
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

option(FIND_RAFT_CPP "Search for existing RAFT C++ installations before defaulting to local files"
       ON
)
option(USE_CUDA_MATH_WHEELS "Use the CUDA math wheels instead of the system libraries" OFF)

# If the user requested it we attempt to find RAFT.
if(FIND_RAFT_CPP)
  find_package(raft "${RAPIDS_VERSION}" REQUIRED COMPONENTS compiled)
  if(NOT TARGET raft::raft_lib)
    message(
      FATAL_ERROR
        "Building against a preexisting libraft library requires the compiled libraft to have been built!"
    )

  endif()
else()
  set(raft_FOUND OFF)
endif()

include(rapids-cython-core)

if(NOT raft_FOUND)
  find_package(CUDAToolkit REQUIRED)

  set(BUILD_TESTS OFF)
  set(BUILD_PRIMS_BENCH OFF)
  set(RAFT_COMPILE_LIBRARY ON)
  set(CUDA_STATIC_RUNTIME ON)
  set(CUDA_STATIC_MATH_LIBRARIES ON)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.0)
    set(CUDA_STATIC_MATH_LIBRARIES OFF)
  elseif(USE_CUDA_MATH_WHEELS)
    message(FATAL_ERROR "Cannot use CUDA math wheels with CUDA < 12.0")
  endif()

  add_subdirectory(../../cpp raft-cpp EXCLUDE_FROM_ALL)

  if(NOT CUDA_STATIC_MATH_LIBRARIES AND USE_CUDA_MATH_WHEELS)
    set_property(
      TARGET raft_lib
      PROPERTY INSTALL_RPATH
               "$ORIGIN/../nvidia/cublas/lib"
               "$ORIGIN/../nvidia/curand/lib"
               "$ORIGIN/../nvidia/cusolver/lib"
               "$ORIGIN/../nvidia/cusparse/lib"
               "$ORIGIN/../nvidia/nvjitlink/lib"
    )
  endif()

  # When building the C++ libraries from source we must copy libraft.so alongside the
  # pairwise_distance and random Cython libraries TODO: when we have a single 'compiled' raft
  # library, we shouldn't need this
  set(cython_lib_dir pylibraft)
  install(TARGETS raft_lib DESTINATION ${cython_lib_dir})
endif()

rapids_cython_init()

add_subdirectory(pylibraft/common)
add_subdirectory(pylibraft/random)
add_subdirectory(pylibraft/sparse)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET raft PATHS "${cython_lib_dir}")
endif()
