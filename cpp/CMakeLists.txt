#=============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
set(RAPIDS_VERSION "22.06")
set(RAFT_VERSION "${RAPIDS_VERSION}.00")

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)
include(../fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(RAFT)

project(RAFT VERSION 22.08.00 LANGUAGES CXX CUDA)

# Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to
# have different values for the `Threads::Threads` target. Setting this flag ensures
# `Threads::Threads` is the same value in first run and subsequent runs.
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Write the version header
rapids_cmake_write_version_file(include/raft/version_config.hpp)

##############################################################################
# - build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build raft shared libraries" ON)
option(BUILD_TESTS "Build raft unit-tests" ON)
option(BUILD_BENCH "Build raft C++ benchmark tests" OFF)
option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler)" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable deprecaction warnings " ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(RAFT_NVTX "Enable nvtx markers" OFF)

option(RAFT_COMPILE_LIBRARIES "Enable building raft shared library instantiations" ${BUILD_TESTS})
option(RAFT_COMPILE_NN_LIBRARY "Enable building raft nearest neighbors shared library instantiations" ${RAFT_COMPILE_LIBRARIES})
option(RAFT_COMPILE_DIST_LIBRARY "Enable building raft distant shared library instantiations" ${RAFT_COMPILE_LIBRARIES})
option(RAFT_ENABLE_NN_DEPENDENCIES "Search for raft::nn dependencies like faiss" ${RAFT_COMPILE_LIBRARIES})

option(RAFT_ENABLE_thrust_DEPENDENCY "Enable Thrust dependency" ON)

if(BUILD_TESTS AND NOT RAFT_ENABLE_thrust_DEPENDENCY)
  message(VERBOSE "RAFT: BUILD_TESTS is enabled, overriding RAFT_ENABLE_thrust_DEPENDENCY")
  set(RAFT_ENABLE_thrust_DEPENDENCY ON)
endif()

option(RAFT_EXCLUDE_FAISS_FROM_ALL "Exclude FAISS targets from RAFT's 'all' target" ON)

include(CMakeDependentOption)
cmake_dependent_option(RAFT_USE_FAISS_STATIC "Build and statically link the FAISS library for nearest neighbors search on GPU" ON RAFT_COMPILE_LIBRARIES OFF)

message(VERBOSE "RAFT: Building optional components: ${raft_FIND_COMPONENTS}")
message(VERBOSE "RAFT: Build RAFT unit-tests: ${BUILD_TESTS}")
message(VERBOSE "RAFT: Building raft C++ benchmarks: ${BUILD_BENCH}")
message(VERBOSE "RAFT: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "RAFT: Disable depreaction warnings " ${DISABLE_DEPRECATION_WARNINGS})
message(VERBOSE "RAFT: Disable OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "RAFT: Enable kernel resource usage info: ${CUDA_ENABLE_KERNELINFO}")
message(VERBOSE "RAFT: Enable lineinfo in nvcc: ${CUDA_ENABLE_LINEINFO}")
message(VERBOSE "RAFT: Enable nvtx markers: ${RAFT_NVTX}")
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

if (NOT DISABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(VERBOSE "RAFT: OpenMP found in ${OpenMP_CXX_INCLUDE_DIRS}")
  endif()
endif()

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET raft-exports
    INSTALL_EXPORT_SET raft-exports)
include(cmake/modules/ConfigureCUDA.cmake)

##############################################################################
# - Requirements -------------------------------------------------------------

if(RAFT_COMPILE_LIBRARIES)
  set(RAFT_COMPILE_DIST_LIBRARY ON)
  set(RAFT_COMPILE_NN_LIBRARY ON)
endif()

if(RAFT_COMPILE_DIST_LIBRARY OR distance IN_LIST raft_FIND_COMPONENTS)
  set(RAFT_ENABLE_cuco_DEPENDENCY ON)
endif()

# add third party dependencies using CPM
rapids_cpm_init()

# thrust before rmm/cuco so we get the right version of thrust/cub
include(cmake/thirdparty/get_thrust.cmake)
include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_faiss.cmake)

if(RAFT_ENABLE_cuco_DEPENDENCY)
  include(${rapids-cmake-dir}/cpm/cuco.cmake)
  rapids_cpm_cuco(BUILD_EXPORT_SET raft-distance-lib-exports INSTALL_EXPORT_SET  raft-distance-lib-exports)
endif()

if(BUILD_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

if(BUILD_BENCH)
  include(cmake/thirdparty/get_gbench.cmake)
endif()

##############################################################################
# - raft ---------------------------------------------------------------------
add_library(raft INTERFACE)
add_library(raft::raft ALIAS raft)

target_include_directories(raft INTERFACE
        "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>")

# Keep RAFT as lightweight as possible.
# Only CUDA libs and rmm should
# be used in global target.
target_link_libraries(raft INTERFACE
        CUDA::cublas
        CUDA::curand
        CUDA::cusolver
        CUDA::cudart
        CUDA::cusparse
        rmm::rmm
        $<$<BOOL:${RAFT_ENABLE_thrust_DEPENDENCY}>:raft::Thrust>
)

target_compile_features(raft INTERFACE cxx_std_17 $<BUILD_INTERFACE:cuda_std_17>)

if(RAFT_COMPILE_DIST_LIBRARY OR RAFT_COMPILE_NN_LIBRARY)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
          [=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=])
endif()

##############################################################################
# - NVTX support in raft -----------------------------------------------------

if (RAFT_NVTX)
  # This enables NVTX within the project with no option to disable it downstream.
  target_link_libraries(raft INTERFACE CUDA::nvToolsExt)
  target_compile_definitions(raft INTERFACE NVTX_ENABLED)
else()
  # Allow enable NVTX downstream if not set here.
  # This creates a new option at build/install time, which is set by default to OFF,
  # but can be enabled in the dependent project.
  get_property(nvtx_option_help_string CACHE RAFT_NVTX PROPERTY HELPSTRING)
  string(CONCAT nvtx_export_string
  "option(RAFT_NVTX \"" ${nvtx_option_help_string} "\" OFF)"
  [=[

target_link_libraries(raft::raft INTERFACE $<$<BOOL:${RAFT_NVTX}>:CUDA::nvToolsExt>)
target_compile_definitions(raft::raft INTERFACE $<$<BOOL:${RAFT_NVTX}>:NVTX_ENABLED>)

  ]=])
endif()

##############################################################################
# - raft_distance ------------------------------------------------------------
add_library(raft_distance INTERFACE)

if(TARGET raft_distance AND (NOT TARGET raft::distance))
  add_library(raft::distance ALIAS raft_distance)
endif()

set_target_properties(raft_distance PROPERTIES EXPORT_NAME distance)

if(RAFT_COMPILE_DIST_LIBRARY)
  add_library(raft_distance_lib
    src/distance/pairwise_distance.cu
    src/distance/specializations/detail/canberra.cu
    src/distance/specializations/detail/chebyshev.cu
    src/distance/specializations/detail/correlation.cu
    src/distance/specializations/detail/cosine.cu
    src/distance/specializations/detail/hamming_unexpanded.cu
    src/distance/specializations/detail/hellinger_expanded.cu
    src/distance/specializations/detail/jensen_shannon_float_float_float_int.cu
    src/distance/specializations/detail/jensen_shannon_float_float_float_uint32.cu
    src/distance/specializations/detail/jensen_shannon_double_double_double_int.cu
    src/distance/specializations/detail/kl_divergence_float_float_float_int.cu
    src/distance/specializations/detail/kl_divergence_float_float_float_uint32.cu
    src/distance/specializations/detail/kl_divergence_double_double_double_int.cu
    src/distance/specializations/detail/l1_float_float_float_int.cu
    src/distance/specializations/detail/l1_float_float_float_uint32.cu
    src/distance/specializations/detail/l1_double_double_double_int.cu
    src/distance/specializations/detail/l2_expanded_float_float_float_int.cu
    src/distance/specializations/detail/l2_expanded_float_float_float_uint32.cu
    src/distance/specializations/detail/l2_expanded_double_double_double_int.cu
    src/distance/specializations/detail/l2_sqrt_expanded_float_float_float_int.cu
    src/distance/specializations/detail/l2_sqrt_expanded_float_float_float_uint32.cu
    src/distance/specializations/detail/l2_sqrt_expanded_double_double_double_int.cu
    src/distance/specializations/detail/l2_sqrt_unexpanded_float_float_float_int.cu
    src/distance/specializations/detail/l2_sqrt_unexpanded_float_float_float_uint32.cu
    src/distance/specializations/detail/l2_sqrt_unexpanded_double_double_double_int.cu
    src/distance/specializations/detail/l2_unexpanded_double_double_double_int.cu
    src/distance/specializations/detail/l2_unexpanded_float_float_float_uint32.cu
    src/distance/specializations/detail/l2_unexpanded_float_float_float_int.cu
    src/distance/specializations/detail/lp_unexpanded_double_double_double_int.cu
    src/distance/specializations/detail/lp_unexpanded_float_float_float_uint32.cu
    src/distance/specializations/detail/lp_unexpanded_float_float_float_int.cu
  )
  set_target_properties(
    raft_distance_lib
    PROPERTIES OUTPUT_NAME                         raft_distance
               BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON)

  target_link_libraries(raft_distance_lib
    PUBLIC raft::raft
           cuco::cuco
    )
  target_compile_options(raft_distance_lib
          PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAFT_CXX_FLAGS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
          )
  target_compile_definitions(raft_distance_lib
          INTERFACE "RAFT_DISTANCE_COMPILED")

  # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
  target_link_options(raft_distance_lib PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")

endif()

if(TARGET raft_distance_lib AND (NOT TARGET raft::raft_distance_lib))
  add_library(raft::raft_distance_lib ALIAS raft_distance_lib)
endif()

target_link_libraries(raft_distance INTERFACE
    raft::raft
    $<TARGET_NAME_IF_EXISTS:raft::raft_distance_lib>
)

##############################################################################
# - raft_nn ------------------------------------------------------------------
add_library(raft_nn INTERFACE)

if(TARGET raft_nn AND (NOT TARGET raft::nn))
  add_library(raft::nn ALIAS raft_nn)
endif()

set_target_properties(raft_nn PROPERTIES EXPORT_NAME nn)

if(RAFT_COMPILE_NN_LIBRARY)
  add_library(raft_nn_lib
          src/nn/specializations/ball_cover.cu
          src/nn/specializations/detail/ball_cover_lowdim_pass_one_2d.cu
          src/nn/specializations/detail/ball_cover_lowdim_pass_two_2d.cu
          src/nn/specializations/detail/ball_cover_lowdim_pass_one_3d.cu
          src/nn/specializations/detail/ball_cover_lowdim_pass_two_3d.cu
          src/nn/specializations/fused_l2_knn_long_float_true.cu
          src/nn/specializations/fused_l2_knn_long_float_false.cu
          src/nn/specializations/fused_l2_knn_int_float_true.cu
          src/nn/specializations/fused_l2_knn_int_float_false.cu
          src/nn/specializations/knn.cu
          )
  set_target_properties(
    raft_nn_lib
    PROPERTIES OUTPUT_NAME                         raft_nn
               BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON)

  target_link_libraries(raft_nn_lib
    PUBLIC faiss::faiss
           raft::raft)
  target_compile_options(raft_nn_lib
          PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAFT_CXX_FLAGS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
          )
  # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
  target_link_options(raft_nn_lib PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")

  target_compile_definitions(raft_nn_lib
          INTERFACE "RAFT_NN_COMPILED")
endif()

if(TARGET raft_nn_lib AND (NOT TARGET raft::raft_nn_lib))
  add_library(raft::raft_nn_lib ALIAS raft_nn_lib)
endif()

target_link_libraries(raft_nn INTERFACE
    raft::raft
    $<TARGET_NAME_IF_EXISTS:raft::raft_nn_lib>)

##############################################################################
# - install targets-----------------------------------------------------------
rapids_cmake_install_lib_dir( lib_dir )
include(GNUInstallDirs)
include(CPack)

install(TARGETS raft
        DESTINATION ${lib_dir}
        COMPONENT raft
        EXPORT raft-exports)

install(TARGETS raft_distance
        DESTINATION ${lib_dir}
        COMPONENT raft
        EXPORT raft-distance-exports)

install(TARGETS raft_nn
        DESTINATION ${lib_dir}
        COMPONENT raft
        EXPORT raft-nn-exports)

if(TARGET raft_distance_lib)
  install(TARGETS raft_distance_lib
          DESTINATION ${lib_dir}
          COMPONENT distance
          EXPORT raft-distance-lib-exports)
  install(DIRECTORY include/raft_distance
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          COMPONENT distance)
endif()

if(TARGET raft_nn_lib)
  install(TARGETS raft_nn_lib
          DESTINATION ${lib_dir}
          COMPONENT nn
          EXPORT raft-nn-lib-exports)
endif()


install(DIRECTORY include/raft
        COMPONENT raft
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Temporary install of raft.hpp while the file is removed
install(FILES include/raft.hpp
        COMPONENT raft
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/raft)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/raft/version_config.hpp
        COMPONENT raft
        DESTINATION include/raft)

##############################################################################
# - export/install optional components  --------------------------------------

include("${rapids-cmake-dir}/export/write_dependencies.cmake")

set(raft_components distance nn)
set(raft_install_comp raft raft)
if(TARGET raft_distance_lib)
  list(APPEND raft_components distance-lib)
  list(APPEND raft_install_comp distance)
endif()
if(TARGET raft_nn_lib)
  list(APPEND raft_components nn-lib)
  list(APPEND raft_install_comp nn)
endif()

foreach(comp install_comp IN ZIP_LISTS raft_components raft_install_comp)
  install(
          EXPORT raft-${comp}-exports
          FILE raft-${comp}-targets.cmake
          NAMESPACE raft::
          DESTINATION "${lib_dir}/cmake/raft"
          COMPONENT ${install_comp}
  )
  export(
          EXPORT raft-${comp}-exports
          FILE ${RAFT_BINARY_DIR}/raft-${comp}-targets.cmake
          NAMESPACE raft::
  )
  rapids_export_write_dependencies(
          BUILD raft-${comp}-exports "${PROJECT_BINARY_DIR}/raft-${comp}-dependencies.cmake"
  )
  rapids_export_write_dependencies(
          INSTALL raft-${comp}-exports "${PROJECT_BINARY_DIR}/rapids-cmake/raft/export/${install_comp}/raft-${comp}-dependencies.cmake"
  )

endforeach()

##############################################################################
# - install export -----------------------------------------------------------
set(doc_string
        [=[
Provide targets for the RAFT: Reusable Accelerated Functions and Tools

RAFT contains fundamental widely-used algorithms and primitives
for data science and machine learning.

Optional Components:
  - nn
  - distance

Imported Targets:
  - raft::raft
  - raft::nn brought in by the `nn` optional component
  - raft::distance brought in by the `distance` optional component

]=])

set(code_string ${nvtx_export_string})

if(RAFT_ENABLE_thrust_DEPENDENCY)
  string(APPEND code_string
  [=[
  if(NOT TARGET raft::Thrust)
    thrust_create_target(raft::Thrust FROM_OPTIONS)
  endif()
  ]=])
endif()

string(APPEND code_string
[=[
if(distance IN_LIST raft_FIND_COMPONENTS)
  enable_language(CUDA)
endif()

if(nn IN_LIST raft_FIND_COMPONENTS)
  enable_language(CUDA)

  if(TARGET faiss AND (NOT TARGET faiss::faiss))
      add_library(faiss::faiss ALIAS faiss)
  elseif(TARGET faiss::faiss AND (NOT TARGET faiss))
      add_library(faiss ALIAS faiss::faiss)
  endif()
endif()
]=])

# Use `rapids_export` for 22.04 as it will have COMPONENT support
include(cmake/modules/raft_export.cmake)
raft_export(INSTALL raft
        COMPONENTS nn distance
        EXPORT_SET raft-exports
        GLOBAL_TARGETS raft nn distance
        NAMESPACE raft::
        DOCUMENTATION doc_string
        FINAL_CODE_BLOCK code_string)

##############################################################################
# - build export -------------------------------------------------------------
raft_export(BUILD raft
        EXPORT_SET raft-exports
        COMPONENTS nn distance
        GLOBAL_TARGETS raft raft_distance raft_nn
        DOCUMENTATION doc_string
        NAMESPACE raft::
        FINAL_CODE_BLOCK code_string)

##############################################################################
# - build test executable ----------------------------------------------------

if(BUILD_TESTS)
  include(test/CMakeLists.txt)
endif()

##############################################################################
# - build benchmark executable -----------------------------------------------

if(BUILD_BENCH)
  include(bench/CMakeLists.txt)
endif()

##############################################################################
# - doxygen targets ----------------------------------------------------------

include(cmake/doxygen.cmake)
add_doxygen_target(IN_DOXYFILE doxygen/Doxyfile.in
        OUT_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        CWD ${CMAKE_CURRENT_BINARY_DIR})
