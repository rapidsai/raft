#=============================================================================
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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


cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# This needs to be run before enabling the CUDA language due to the default initialization behavior
# of `CMAKE_CUDA_ARCHITECTURES`, https://gitlab.kitware.com/cmake/cmake/-/issues/21302
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "ALL")
  set(RAFT_BUILD_FOR_ALL_ARCHS TRUE)
  set(CMAKE_CUDA_ARCHITECTURES "")
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(RAFT_BUILD_FOR_DETECTED_ARCHS TRUE)
endif()

# language CUDA is enabled in ConfigureCUDA.cmake
project(RAFT VERSION 0.20.0 LANGUAGES CXX)

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

option(BUILD_TESTS "Build raft unit-tests" ON)
option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(NVTX "Enable nvtx markers" OFF)

message(VERBOSE "RAFT: Build RAFT unit-tests: ${BUILD_TESTS}")
message(VERBOSE "RAFT: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "RAFT: Disable OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "RAFT: Enable kernel resource usage info: ${CUDA_ENABLE_KERNELINFO}")
message(VERBOSE "RAFT: Enable lineinfo in nvcc: ${CUDA_ENABLE_LINEINFO}")
message(VERBOSE "RAFT: Enable nvtx markers: ${NVTX}")
message(VERBOSE "RAFT: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")

# Set RMM logging level
set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(VERBOSE "RAFT: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

if (NOT RAFT_GENERATED_INCLUDE_DIR)
    set(RAFT_GENERATED_INCLUDE_DIR ${RAFT_BINARY_DIR})
endif()

##############################################################################
# - Conda environment detection ----------------------------------------------

# conda-build conda package building detection
if("$ENV{CONDA_BUILD}" STREQUAL "1")
    message(STATUS "RAFT: Detected conda-build call, CMAKE_PREFIX_PATH unchanged.")

# otherwise detecting conda environment
else("$ENV{CONDA_BUILD}" STREQUAL "1")
  if(DETECT_CONDA_ENV)

    if(DEFINED ENV{CONDA_PREFIX})
      set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
      message(STATUS "RAFT: Detected Conda environment, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")

      if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message(STATUS "RAFT: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
        set (CMAKE_INSTALL_PREFIX $ENV{CONDA_PREFIX})
      endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

    else(DEFINED ENV{CONDA_PREFIX})
      message(STATUS "RAFT: No Conda environment detected, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
    endif(DEFINED ENV{CONDA_PREFIX})

  else(DETECT_CONDA_ENV)
    message(STATUS "RAFT: No Conda detection being used")
  endif(DETECT_CONDA_ENV)

endif("$ENV{CONDA_BUILD}" STREQUAL "1")

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
include(cmake/modules/ConfigureCUDA.cmake)

##############################################################################
# - Requirements -------------------------------------------------------------

# our GENERATE_FIND_MODULE function works perfectly to allow CPMFindPackage work with packages
# that don't have a find*.cmake (like it did for find_package), like nccl :)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR}/cmake)

if (NOT DISABLE_OPENMP OR NOT ${DISABLE_OPENMP})
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(VERBOSE "RAFT: OpenMP found in ${OpenMP_CXX_INCLUDE_DIRS}")
  endif(OPENMP_FOUND)
endif(NOT DISABLE_OPENMP OR NOT ${DISABLE_OPENMP})

# add third party dependencies using CPM
include(cmake/thirdparty/get_cpm.cmake)

# add third party dependencies using CPM
include(cmake/thirdparty/get_cpm.cmake)

include(cmake/thirdparty/get_faiss.cmake)

include(cmake/thirdparty/get_rmm.cmake)

if(NOT CUB_IS_PART_OF_CTK)
  include(cmake/thirdparty/get_cub.cmake)
endif(NOT CUB_IS_PART_OF_CTK)

if(BUILD_TESTS)
  include(cmake/thirdparty/get_nccl.cmake)

  include(cmake/thirdparty/get_ucx.cmake)

  include(cmake/thirdparty/get_gtest.cmake)
endif()

##############################################################################
# - install targets-----------------------------------------------------------

# Currently not used, but added to ease the transition to a conda package, see:
# https://github.com/rapidsai/raft/issues/83

# add_library(raft INTERFACE)
# add_library(raft::raft ALIAS raft)
# target_include_directories(raft INTERFACE "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/include>"
#                                           "$<INSTALL_INTERFACE:include>")


# target_compile_features(raft INTERFACE cxx_std_14 $<BUILD_INTERFACE:cuda_std_14>)

##############################################################################
# - build test executable ----------------------------------------------------

if(BUILD_TESTS)
  include(test/CMakeLists.txt)
endif()

##############################################################################
# - doxygen targets ----------------------------------------------------------

include(cmake/doxygen.cmake)
add_doxygen_target(IN_DOXYFILE Doxyfile.in
  OUT_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  CWD ${CMAKE_CURRENT_BINARY_DIR})

macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

# print_all_variables()
