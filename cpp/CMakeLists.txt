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

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)
file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-21.12/RAPIDS.cmake
    ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(RAFT)

project(RAFT VERSION 21.12.00 LANGUAGES CXX CUDA)

##############################################################################
# - build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - User Options  ------------------------------------------------------------

option(BUILD_TESTS "Build raft unit-tests" ON)
option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler)" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable depreaction warnings " ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(NVTX "Enable nvtx markers" OFF)

option(RAFT_USE_FAISS_STATIC "Build and statically link the FAISS library for nearest neighbors search on GPU" ON)

message(VERBOSE "RAFT: Build RAFT unit-tests: ${BUILD_TESTS}")
message(VERBOSE "RAFT: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "RAFT: Disable depreaction warnings " ${DISABLE_DEPRECATION_WARNINGS})
message(VERBOSE "RAFT: Disable OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "RAFT: Enable kernel resource usage info: ${CUDA_ENABLE_KERNELINFO}")
message(VERBOSE "RAFT: Enable lineinfo in nvcc: ${CUDA_ENABLE_LINEINFO}")
message(VERBOSE "RAFT: Enable nvtx markers: ${NVTX}")
message(VERBOSE "RAFT: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")

# Set RMM logging level
set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(VERBOSE "RAFT: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

##############################################################################
# - Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
      message(STATUS "RAFT: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
      set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

##############################################################################
# - compiler options ---------------------------------------------------------

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET raft-exports
    INSTALL_EXPORT_SET raft-exports
    )
include(cmake/modules/ConfigureCUDA.cmake)

##############################################################################
# - Requirements -------------------------------------------------------------

if (NOT DISABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(VERBOSE "RAFT: OpenMP found in ${OpenMP_CXX_INCLUDE_DIRS}")
  endif()
endif()

# add third party dependencies using CPM
rapids_cpm_init()

include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_cuco.cmake)
# CPMFindPackage(faiss) so it's available for raft consumers to use
include(cmake/thirdparty/get_faiss.cmake)
# CPMFindPackage(GTest) so it's available for raft consumers to use
include(cmake/thirdparty/get_gtest.cmake)

if(BUILD_TESTS)
  include(cmake/thirdparty/get_nccl.cmake)
  include(cmake/thirdparty/get_ucx.cmake)
endif()

##############################################################################
# - install targets-----------------------------------------------------------
rapids_cmake_install_lib_dir( lib_dir )

include(CPack)

add_library(raft INTERFACE)
add_library(raft::raft ALIAS raft)
target_include_directories(raft INTERFACE "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/include>"
                                          "$<INSTALL_INTERFACE:include>")

target_link_libraries(raft
INTERFACE
  CUDA::cublas
  CUDA::curand
  CUDA::cusolver
  CUDA::cudart
  CUDA::cusparse
  rmm::rmm
  cuco::cuco
  )

target_compile_features(raft INTERFACE cxx_std_17 $<BUILD_INTERFACE:cuda_std_17>)

install(TARGETS raft
        DESTINATION ${lib_dir}
        EXPORT raft-exports
        )

include(GNUInstallDirs)
install(DIRECTORY include/raft/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/raft
        )

# Temporary install of raft.hpp while the file is removed
install(FILES include/raft.hpp
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/raft
	)

##############################################################################
# - install export -----------------------------------------------------------
set(doc_string
[=[
Provide targets for the RAFT: RAPIDS Analytics Frameworks Toolset.

RAFT is a repository containining shared utilities, mathematical operations
and common functions for the analytics components of RAPIDS.

]=])

set(raft_code_string
[=[
if(TARGET faiss AND (NOT TARGET faiss::faiss))
    add_library(faiss::faiss ALIAS faiss)
elseif(TARGET faiss::faiss AND (NOT TARGET faiss))
    add_library(faiss ALIAS faiss::faiss)
endif()
]=]
)

 rapids_export(INSTALL raft
    EXPORT_SET raft-exports
    GLOBAL_TARGETS raft # since we can't hook into EXPORT SETS
    NAMESPACE raft::
    DOCUMENTATION doc_string
    FINAL_CODE_BLOCK raft_code_string
    )

##############################################################################
# - build export -------------------------------------------------------------

rapids_export(BUILD raft
    EXPORT_SET raft-exports
    GLOBAL_TARGETS raft # since we can't hook into EXPORT SETS
    LANGUAGES CUDA
    DOCUMENTATION doc_string
    NAMESPACE raft::
    FINAL_CODE_BLOCK raft_code_string
    )

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
