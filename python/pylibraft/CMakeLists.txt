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

rapids_cython_init()

add_subdirectory(pylibraft/common)
add_subdirectory(pylibraft/random)
add_subdirectory(pylibraft/sparse)
