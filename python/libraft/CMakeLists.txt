# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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

include(rapids-cuda)
rapids_cuda_init_architectures(libraft-python)

project(
  libraft-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

option(USE_CUDA_MATH_WHEELS "Use the CUDA math wheels instead of the system libraries" OFF)

find_package(CUDAToolkit REQUIRED)

# Check if raft is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python raft package.
find_package(raft "${RAPIDS_VERSION}")

if(raft_FOUND)
  return()
endif()

unset(raft_FOUND)

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

# ensure installs are inside the libraft/ folder in site-packages/
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# TODO(jameslamb): figure out how to keep the static library (libraft.a) out of wheels.
#
# things tried:
#
# - 
add_subdirectory(../../cpp raft-cpp)

# install(
#   TARGETS
#     cuco
#     CUTLASS
#   COMPONENT libraft_wheel_interface_dependencies
# )

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
