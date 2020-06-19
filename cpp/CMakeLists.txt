#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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
#=============================================================================


cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(RAFT VERSION 0.14.0 LANGUAGES CXX CUDA)

##############################################################################
# - build type ---------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release")
endif()

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - User Options  ------------------------------------------------------------

option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)

option(EMPTY_MARKER_KERNEL "Enable empty marker kernel after nvtxRangePop" ON)

option(KERNEL_INFO "Enable kernel resource usage info" OFF)

option(LINE_INFO "Enable lineinfo in nvcc" OFF)

option(NVTX "Enable nvtx markers" OFF)

option(BUILD_RAFT_TESTS "Build raft unit-tests" ON)

set(PARALLEL_LEVEL "" CACHE STRING
    "Sub-projects parallel level for compilation. Currently only affects FAISS" )

set(GPU_ARCHS "" CACHE STRING
    "List of GPU architectures (semicolon-separated) to be compiled for.
    Pass 'ALL' if you want to compile for all supported GPU architectures.
    Empty string means to auto-detect the GPUs on the current system")

##############################################################################
# - Requirements -------------------------------------------------------------

find_package(CUDA 10.0 REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

##############################################################################
# - Compiler Options  --------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CUDA_HOST_COMPILER)
  # If CMAKE_CUDA_HOST_COMPILER is set to a nonempty string cmake was called with the environment variable CUDAHOSTCXX set or -DCMAKE_CUDA_HOST_COMPILER
  if(NOT CMAKE_CUDA_HOST_COMPILER STREQUAL CMAKE_CXX_COMPILER)
    message(WARNING "CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER} and CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} are inconsistent!")
  endif(NOT CMAKE_CUDA_HOST_COMPILER STREQUAL CMAKE_CXX_COMPILER)
else()
  # No attempt to set CMAKE_CUDA_HOST_COMPILER has been made. Make sure CMAKE_CXX_COMPILER is used as CUDA host compiler.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")
endif(CMAKE_CUDA_HOST_COMPILER)

if(OPENMP_FOUND)
  message(STATUS "Building with OpenMP support")
  find_package(Threads REQUIRED)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

if(${CMAKE_VERSION} VERSION_LESS "3.17.0")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++14")
endif(${CMAKE_VERSION} VERSION_LESS "3.17.0")

if(LINE_INFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif(LINE_INFO)

if(KERNEL_INFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")
endif(KERNEL_INFO)

if(NVTX)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNVTX_ENABLED")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNVTX_ENABLED")
  if(EMPTY_MARKER_KERNEL)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DENABLE_EMPTY_MARKER_KERNEL")
  endif(EMPTY_MARKER_KERNEL)
endif(NVTX)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

if("${GPU_ARCHS}" STREQUAL "")
  include(cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()

if("${GPU_ARCHS}" STREQUAL "ALL")
  set(GPU_ARCHS "60")
  if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
    set(GPU_ARCHS "${GPU_ARCHS};70")
  endif()
  if((CUDA_VERSION_MAJOR EQUAL 10) OR (CUDA_VERSION_MAJOR GREATER 10))
    set(GPU_ARCHS "${GPU_ARCHS};75")
  endif()
endif()

message("-- Building for GPU_ARCHS = ${GPU_ARCHS}")

foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
  set(FAISS_GPU_ARCHS "${FAISS_GPU_ARCHS} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

list(GET GPU_ARCHS -1 ptx)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${ptx},code=compute_${ptx}")
set(FAISS_GPU_ARCHS "${FAISS_GPU_ARCHS} -gencode arch=compute_${ptx},code=compute_${ptx}")

if(CMAKE_COMPILER_IS_GNUCXX)
  if(NOT CMAKE_CXX11_ABI)
    message(STATUS "Disabling the GLIBCXX11 ABI")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
  elseif(CMAKE_CXX11_ABI)
    message(STATUS "Enabling the GLIBCXX11 ABI")
  endif(NOT CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_CUDA_FLAGS
  "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=unrecognized_gcc_pragma")

##############################################################################
# - dependencies -------------------------------------------------------------

include(cmake/Dependencies.cmake)

##############################################################################
# - include paths ------------------------------------------------------------

set(RAFT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING
  "Path to RAFT include directories")

set(RAFT_INCLUDE_DIRECTORIES
  ${RAFT_INCLUDE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(DEFINED ENV{CONDA_PREFIX})
  message(STATUS "Using RMM installation froM $ENV{CONDA_PREFIX}")
  list(APPEND RAFT_INCLUDE_DIRECTORIES $ENV{CONDA_PREFIX}/include)
endif(DEFINED ENV{CONDA_PREFIX})

##############################################################################
# - libraries ----------------------------------------------------------------

set(RAFT_LINK_LIBRARIES
  ${CUDA_cublas_LIBRARY}
  ${CUDA_cusolver_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
  ${CUDA_curand_LIBRARY}
  rmm)

set(RAFT_LINK_DIRECTORIES "")

if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND RAFT_LINK_DIRECTORIES $ENV{CONDA_PREFIX}/lib)
endif(DEFINED ENV{CONDA_PREFIX})

##############################################################################
# - build test executable ----------------------------------------------------

if(BUILD_RAFT_TESTS)
  find_package(OpenMP REQUIRED)

  # keep the files in alphabetical order!
  add_executable(test_raft
    test/cudart_utils.cpp
    test/handle.cpp
    test/mr/device/buffer.cpp
    test/mr/host/buffer.cpp
    test/test.cpp
    test/spectral_matrix.cu
    test/eigen_solvers.cu
    test/cluster_solvers.cu)

  target_include_directories(test_raft
    PRIVATE
      ${RAFT_INCLUDE_DIRECTORIES}
      ${GTEST_DIR}/include)

  target_link_directories(test_raft
    PRIVATE
      ${RAFT_LINK_DIRECTORIES})

  target_link_libraries(test_raft
    PRIVATE
      ${RAFT_LINK_LIBRARIES}
      gtestlib
      gtest_mainlib
      OpenMP::OpenMP_CXX
      Threads::Threads)
endif(BUILD_RAFT_TESTS)

##############################################################################
# - doxygen targets ----------------------------------------------------------

include(cmake/doxygen.cmake)
add_doxygen_target(IN_DOXYFILE Doxyfile.in
  OUT_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  CWD ${CMAKE_CURRENT_BINARY_DIR})
