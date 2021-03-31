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


cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(RAFT VERSION 0.20.0 LANGUAGES CXX CUDA)

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

option(BUILD_GTEST "Build the GTEST library for running raft test executables" OFF)

option(BUILD_STATIC_FAISS "Build the FAISS library for nearest neighbors search on GPU" OFF)

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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

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
  if((CUDA_VERSION_MAJOR EQUAL 11) OR (CUDA_VERSION_MAJOR GREATER 11))
    set(GPU_ARCHS "${GPU_ARCHS};80")
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
  "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=unrecognized_gcc_pragma --expt-relaxed-constexpr")

##############################################################################
# - dependencies -------------------------------------------------------------

# CUDA 11 onwards cub ships with CTK
if((CUDA_VERSION_MAJOR EQUAL 11) OR (CUDA_VERSION_MAJOR GREATER 11))
  set(CUB_IS_PART_OF_CTK ON)
else()
  set(CUB_IS_PART_OF_CTK OFF)
endif()

include(cmake/Dependencies.cmake)
include(cmake/comms.cmake)

###################################################################################################
# - FAISS -------------------------------------------------------------------------------------------

if(NOT BUILD_STATIC_FAISS)
 find_path(FAISS_INCLUDE_DIRS "faiss"
    HINTS
    "$ENV{FAISS_ROOT}/include"
    "$ENV{CONDA_PREFIX}/include/faiss"
    "$ENV{CONDA_PREFIX}/include")
endif(NOT BUILD_STATIC_FAISS)
message(STATUS "FAISS: FAISS_INCLUDE_DIRS set to ${FAISS_INCLUDE_DIRS}")

###################################################################################################
# - RMM -------------------------------------------------------------------------------------------

find_path(RMM_INCLUDE_DIRS "rmm"
    HINTS
    "$ENV{RMM_ROOT}/include"
    "$ENV{CONDA_PREFIX}/include/rmm"
    "$ENV{CONDA_PREFIX}/include")

message(STATUS "RMM: RMM_INCLUDE_DIRS set to ${RMM_INCLUDE_DIRS}")


##############################################################################
# - include paths ------------------------------------------------------------

set(RAFT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING
  "Path to RAFT include directories")

set(RAFT_INCLUDE_DIRECTORIES
  ${RAFT_INCLUDE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${FAISS_INCLUDE_DIRS}
  ${RMM_INCLUDE_DIRS})

if(NOT CUB_IS_PART_OF_CTK)
  list(APPEND RAFT_INCLUDE_DIRECTORIES ${CUB_DIR}/src/cub)
endif(NOT CUB_IS_PART_OF_CTK)

if(DEFINED ENV{CONDA_PREFIX})
  message(STATUS "Using RMM installation from $ENV{CONDA_PREFIX}")
  list(APPEND RAFT_INCLUDE_DIRECTORIES $ENV{CONDA_PREFIX}/include)
endif(DEFINED ENV{CONDA_PREFIX})

##############################################################################
# - libraries ----------------------------------------------------------------

set(RAFT_LINK_LIBRARIES
  ${CUDA_cublas_LIBRARY}
  ${CUDA_cusolver_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
  ${CUDA_curand_LIBRARY})

set(RAFT_LINK_DIRECTORIES
  ${FAISS_INCLUDE_DIRS}
  ${RMM_INCLUDE_DIRS})

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
    test/distance/fused_l2_nn.cu
    test/handle.cpp
    test/integer_utils.cpp
    test/label/label.cu
    test/label/merge_labels.cu
    test/lap/lap.cu
    test/linalg/add.cu
    test/linalg/binary_op.cu
    test/linalg/cholesky_r1.cu
    test/linalg/coalesced_reduction.cu
    test/linalg/divide.cu
    test/linalg/eig.cu
    test/linalg/eig_sel.cu
    test/linalg/gemm_layout.cu
    test/linalg/map.cu
    test/linalg/map_then_reduce.cu
    test/linalg/matrix_vector_op.cu
    test/linalg/multiply.cu
    test/linalg/norm.cu
    test/linalg/reduce.cu
    test/linalg/strided_reduction.cu
    test/linalg/subtract.cu
    test/linalg/svd.cu
    test/linalg/transpose.cu
    test/linalg/unary_op.cu
    test/matrix/math.cu
    test/matrix/matrix.cu
    test/mr/device/buffer.cpp
    test/mr/host/buffer.cpp
    test/random/rng.cu
    test/random/rng_int.cu
    test/random/sample_without_replacement.cu
    test/sparse/add.cu
    test/sparse/convert_coo.cu
    test/sparse/convert_csr.cu
    test/sparse/connect_components.cu
    test/sparse/csr_row_slice.cu
    test/sparse/csr_to_dense.cu
    test/sparse/csr_transpose.cu
    test/sparse/degree.cu
    test/sparse/dist_coo_spmv.cu
    test/sparse/dist_csr_spmv.cu
    test/sparse/distance.cu
    test/sparse/filter.cu
    test/sparse/knn.cu
    test/sparse/knn_graph.cu
    test/sparse/linkage.cu
    test/sparse/norm.cu
    test/sparse/reduce.cu
    test/sparse/row_op.cu
    test/sparse/selection.cu
    test/sparse/sort.cu
    test/sparse/symmetrize.cu
    test/spatial/knn.cu
    test/spatial/haversine.cu
    test/stats/mean.cu
    test/stats/mean_center.cu
    test/stats/stddev.cu
    test/stats/sum.cu
    test/test.cpp
    test/spectral_matrix.cu
    test/eigen_solvers.cu
    test/cluster_solvers.cu
    test/mst.cu)

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
      FAISS::FAISS
      GTest::GTest
      GTest::Main
      OpenMP::OpenMP_CXX
      Threads::Threads)

endif(BUILD_RAFT_TESTS)

##############################################################################
# - doxygen targets ----------------------------------------------------------

include(cmake/doxygen.cmake)
add_doxygen_target(IN_DOXYFILE Doxyfile.in
  OUT_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  CWD ${CMAKE_CURRENT_BINARY_DIR})
