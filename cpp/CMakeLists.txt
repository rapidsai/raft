# =============================================================================
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
set(RAPIDS_VERSION "23.06")
set(RAFT_VERSION "23.06.00")

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)
include(../fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(RAFT)

project(
  RAFT
  VERSION ${RAFT_VERSION}
  LANGUAGES CXX CUDA
)

# Write the version header
rapids_cmake_write_version_file(include/raft/version_config.hpp)

# ##################################################################################################
# * build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ##################################################################################################
# * User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build raft shared libraries" ON)
option(BUILD_TESTS "Build raft unit-tests" ON)
option(BUILD_PRIMS_BENCH "Build raft C++ benchmark tests" OFF)
option(BUILD_ANN_BENCH "Build raft ann benchmarks" OFF)
option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINEINFO
       "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler)" OFF
)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA toolkit runtime and libraries" OFF)
option(CUDA_LOG_COMPILE_TIME "Write a log of compilation times to nvcc_compile_log.csv" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable deprecaction warnings " ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(RAFT_NVTX "Enable nvtx markers" OFF)

set(RAFT_COMPILE_LIBRARY_DEFAULT OFF)
if(BUILD_TESTS
   OR BUILD_PRIMS_BENCH
   OR BUILD_ANN_BENCH
)
  set(RAFT_COMPILE_LIBRARY_DEFAULT ON)
endif()
option(RAFT_COMPILE_LIBRARY "Enable building raft shared library instantiations"
       ${RAFT_COMPILE_LIBRARY_DEFAULT}
)

# Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to
# have different values for the `Threads::Threads` target. Setting this flag ensures
# `Threads::Threads` is the same value across all builds so that cache hits occur
set(THREADS_PREFER_PTHREAD_FLAG ON)

include(CMakeDependentOption)
# cmake_dependent_option( RAFT_USE_FAISS_STATIC "Build and statically link the FAISS library for
# nearest neighbors search on GPU" ON RAFT_COMPILE_LIBRARY OFF )

message(VERBOSE "RAFT: Building optional components: ${raft_FIND_COMPONENTS}")
message(VERBOSE "RAFT: Build RAFT unit-tests: ${BUILD_TESTS}")
message(VERBOSE "RAFT: Building raft C++ benchmarks: ${BUILD_PRIMS_BENCH}")
message(VERBOSE "RAFT: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "RAFT: Disable depreaction warnings " ${DISABLE_DEPRECATION_WARNINGS})
message(VERBOSE "RAFT: Disable OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "RAFT: Enable kernel resource usage info: ${CUDA_ENABLE_KERNELINFO}")
message(VERBOSE "RAFT: Enable lineinfo in nvcc: ${CUDA_ENABLE_LINEINFO}")
message(VERBOSE "RAFT: Enable nvtx markers: ${RAFT_NVTX}")
message(VERBOSE
        "RAFT: Statically link the CUDA toolkit runtime and libraries: ${CUDA_STATIC_RUNTIME}"
)

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
)
message(VERBOSE "RAFT: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

# ##################################################################################################
# * Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
    message(
      STATUS "RAFT: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}"
    )
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

# ##################################################################################################
# * compiler options ---------------------------------------------------------

set(_ctk_static_suffix "")
if(CUDA_STATIC_RUNTIME)
  set(_ctk_static_suffix "_static")
endif()

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})

if(NOT DISABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(VERBOSE "RAFT: OpenMP found in ${OpenMP_CXX_INCLUDE_DIRS}")
  endif()
endif()

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET raft-exports
  INSTALL_EXPORT_SET raft-exports
)
include(cmake/modules/ConfigureCUDA.cmake)

# ##################################################################################################
# * Requirements -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

# thrust before rmm/cuco so we get the right version of thrust/cub
include(cmake/thirdparty/get_thrust.cmake)
include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_cutlass.cmake)

include(${rapids-cmake-dir}/cpm/cuco.cmake)
rapids_cpm_cuco(BUILD_EXPORT_SET raft-exports INSTALL_EXPORT_SET raft-exports)

if(BUILD_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

if(BUILD_PRIMS_BENCH)
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench()
endif()

# ##################################################################################################
# * raft ---------------------------------------------------------------------
add_library(raft INTERFACE)
add_library(raft::raft ALIAS raft)

target_include_directories(
  raft INTERFACE "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/include>" "$<INSTALL_INTERFACE:include>"
)

# Keep RAFT as lightweight as possible. Only CUDA libs and rmm should be used in global target.
target_link_libraries(raft INTERFACE rmm::rmm cuco::cuco nvidia::cutlass::cutlass raft::Thrust)

target_compile_features(raft INTERFACE cxx_std_17 $<BUILD_INTERFACE:cuda_std_17>)
target_compile_options(
  raft INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda
                 --expt-relaxed-constexpr>
)

set(RAFT_CUSOLVER_DEPENDENCY CUDA::cusolver${_ctk_static_suffix})
set(RAFT_CUBLAS_DEPENDENCY CUDA::cublas${_ctk_static_suffix})
set(RAFT_CURAND_DEPENDENCY CUDA::curand${_ctk_static_suffix})
set(RAFT_CUSPARSE_DEPENDENCY CUDA::cusparse${_ctk_static_suffix})

set(RAFT_CTK_MATH_DEPENDENCIES ${RAFT_CUBLAS_DEPENDENCY} ${RAFT_CUSOLVER_DEPENDENCY}
                               ${RAFT_CUSPARSE_DEPENDENCY} ${RAFT_CURAND_DEPENDENCY}
)

# Endian detection
include(TestBigEndian)
test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
  target_compile_definitions(raft INTERFACE RAFT_SYSTEM_LITTLE_ENDIAN=0)
else()
  target_compile_definitions(raft INTERFACE RAFT_SYSTEM_LITTLE_ENDIAN=1)
endif()

if(RAFT_COMPILE_LIBRARY)
  file(
    WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
    [=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=]
  )
endif()

# ##################################################################################################
# * NVTX support in raft -----------------------------------------------------

if(RAFT_NVTX)
  # This enables NVTX within the project with no option to disable it downstream.
  target_link_libraries(raft INTERFACE CUDA::nvToolsExt)
  target_compile_definitions(raft INTERFACE NVTX_ENABLED)
else()
  # Allow enable NVTX downstream if not set here. This creates a new option at build/install time,
  # which is set by default to OFF, but can be enabled in the dependent project.
  get_property(
    nvtx_option_help_string
    CACHE RAFT_NVTX
    PROPERTY HELPSTRING
  )
  string(
    CONCAT
      nvtx_export_string
      "option(RAFT_NVTX \""
      ${nvtx_option_help_string}
      "\" OFF)"
      [=[

target_link_libraries(raft::raft INTERFACE $<$<BOOL:${RAFT_NVTX}>:CUDA::nvToolsExt>)
target_compile_definitions(raft::raft INTERFACE $<$<BOOL:${RAFT_NVTX}>:NVTX_ENABLED>)

  ]=]
  )
endif()

# ##################################################################################################
# * raft_compiled ------------------------------------------------------------ TODO: Currently, this
#   package also contains the 'random' namespace (for rmat logic) We couldn't get this to work
#   properly due to strange CI failures as noticed in the PR#778. In the long term, we should rename
#   this package to `raft_compiled` in order to have a single pre-compiled raft package for those
#   who need it.
add_library(raft_compiled INTERFACE)

if(TARGET raft_compiled AND (NOT TARGET raft::compiled))
  add_library(raft::compiled ALIAS raft_compiled)
endif()

set_target_properties(raft_compiled PROPERTIES EXPORT_NAME compiled)

if(RAFT_COMPILE_LIBRARY)
  add_library(
    raft_lib
    src/distance/pairwise_distance.cu
    src/distance/fused_l2_min_arg.cu
    src/cluster/update_centroids_float.cu
    src/cluster/update_centroids_double.cu
    src/cluster/cluster_cost_float.cu
    src/cluster/cluster_cost_double.cu
    src/neighbors/refine_d_int64_t_float.cu
    src/neighbors/refine_d_int64_t_int8_t.cu
    src/neighbors/refine_d_int64_t_uint8_t.cu
    src/neighbors/refine_h_int64_t_float.cu
    src/neighbors/refine_h_int64_t_int8_t.cu
    src/neighbors/refine_h_int64_t_uint8_t.cu
    src/neighbors/specializations/refine_d_int64_t_float.cu
    src/neighbors/specializations/refine_d_int64_t_int8_t.cu
    src/neighbors/specializations/refine_d_int64_t_uint8_t.cu
    src/neighbors/specializations/refine_h_int64_t_float.cu
    src/neighbors/specializations/refine_h_int64_t_int8_t.cu
    src/neighbors/specializations/refine_h_int64_t_uint8_t.cu
    src/cluster/kmeans_fit_float.cu
    src/cluster/kmeans_fit_double.cu
    src/cluster/kmeans_init_plus_plus_double.cu
    src/cluster/kmeans_init_plus_plus_float.cu
    src/distance/specializations/detail/canberra_double_double_double_int.cu
    src/distance/specializations/detail/canberra_float_float_float_int.cu
    src/distance/specializations/detail/correlation_double_double_double_int.cu
    src/distance/specializations/detail/correlation_float_float_float_int.cu
    src/distance/specializations/detail/cosine_double_double_double_int.cu
    src/distance/specializations/detail/cosine_float_float_float_int.cu
    src/distance/specializations/detail/hamming_unexpanded_double_double_double_int.cu
    src/distance/specializations/detail/hamming_unexpanded_float_float_float_int.cu
    src/distance/specializations/detail/hellinger_expanded_float_float_float_int.cu
    src/distance/specializations/detail/hellinger_expanded_double_double_double_int.cu
    src/distance/specializations/detail/inner_product_float_float_float_int.cu
    src/distance/specializations/detail/inner_product_double_double_double_int.cu
    src/distance/specializations/detail/jensen_shannon_float_float_float_int.cu
    src/distance/specializations/detail/jensen_shannon_double_double_double_int.cu
    src/distance/specializations/detail/kernels/gram_matrix_base_double.cu
    src/distance/specializations/detail/kernels/gram_matrix_base_float.cu
    src/distance/specializations/detail/kernels/polynomial_kernel_double_int.cu
    src/distance/specializations/detail/kernels/polynomial_kernel_float_int.cu
    # These are somehow missing a kernel definition which is causing a compile error.
    # src/distance/specializations/detail/kernels/rbf_kernel_double.cu
    # src/distance/specializations/detail/kernels/rbf_kernel_float.cu
    src/neighbors/brute_force_knn_int64_t_float.cu
    src/distance/specializations/detail/kernels/tanh_kernel_double.cu
    src/distance/specializations/detail/kernels/tanh_kernel_float.cu
    src/distance/specializations/detail/kl_divergence_float_float_float_int.cu
    src/distance/specializations/detail/kl_divergence_double_double_double_int.cu
    src/distance/specializations/detail/l1_float_float_float_int.cu
    src/distance/specializations/detail/l1_double_double_double_int.cu
    src/distance/specializations/detail/l2_expanded_float_float_float_int.cu
    src/distance/specializations/detail/l2_expanded_double_double_double_int.cu
    src/distance/specializations/detail/l2_unexpanded_double_double_double_int.cu
    src/distance/specializations/detail/l2_unexpanded_float_float_float_int.cu
    src/distance/specializations/detail/l_inf_double_double_double_int.cu
    src/distance/specializations/detail/l_inf_float_float_float_int.cu
    src/distance/specializations/detail/lp_unexpanded_double_double_double_int.cu
    src/distance/specializations/detail/lp_unexpanded_float_float_float_int.cu
    src/distance/specializations/detail/russel_rao_double_double_double_int.cu
    src/distance/specializations/detail/russel_rao_float_float_float_int.cu
    src/distance/specializations/fused_l2_nn_double_int.cu
    src/distance/specializations/fused_l2_nn_double_int64.cu
    src/distance/specializations/fused_l2_nn_float_int.cu
    src/distance/specializations/fused_l2_nn_float_int64.cu
    src/matrix/select_k_float_int64_t.cu
    src/matrix/specializations/detail/select_k_float_uint32_t.cu
    src/matrix/specializations/detail/select_k_float_int64_t.cu
    src/matrix/specializations/detail/select_k_half_uint32_t.cu
    src/matrix/specializations/detail/select_k_half_int64_t.cu
    src/neighbors/ivfpq_build.cu
    src/neighbors/ivfpq_deserialize.cu
    src/neighbors/ivfpq_serialize.cu
    src/neighbors/ivfpq_search_float_int64_t.cu
    src/neighbors/ivfpq_search_int8_t_int64_t.cu
    src/neighbors/ivfpq_search_uint8_t_int64_t.cu
    src/neighbors/specializations/ivfpq_build_float_int64_t.cu
    src/neighbors/specializations/ivfpq_build_int8_t_int64_t.cu
    src/neighbors/specializations/ivfpq_build_uint8_t_int64_t.cu
    src/neighbors/specializations/ivfpq_extend_float_int64_t.cu
    src/neighbors/specializations/ivfpq_extend_int8_t_int64_t.cu
    src/neighbors/specializations/ivfpq_extend_uint8_t_int64_t.cu
    src/neighbors/specializations/ivfpq_search_float_int64_t.cu
    src/neighbors/specializations/ivfpq_search_int8_t_int64_t.cu
    src/neighbors/specializations/ivfpq_search_uint8_t_int64_t.cu
    src/neighbors/specializations/detail/brute_force_knn_impl_long_float_int.cu
    src/neighbors/specializations/detail/brute_force_knn_impl_long_float_uint.cu
    src/neighbors/specializations/detail/brute_force_knn_impl_uint_float_int.cu
    src/neighbors/specializations/detail/brute_force_knn_impl_uint_float_uint.cu
    src/neighbors/specializations/detail/compute_similarity_float_float_fast.cu
    src/neighbors/specializations/detail/compute_similarity_float_float_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_float_float_no_smem_lut.cu
    src/neighbors/specializations/detail/compute_similarity_float_fp8s_fast.cu
    src/neighbors/specializations/detail/compute_similarity_float_fp8s_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_float_fp8s_no_smem_lut.cu
    src/neighbors/specializations/detail/compute_similarity_float_fp8u_fast.cu
    src/neighbors/specializations/detail/compute_similarity_float_fp8u_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_float_fp8u_no_smem_lut.cu
    src/neighbors/specializations/detail/compute_similarity_float_half_fast.cu
    src/neighbors/specializations/detail/compute_similarity_float_half_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_float_half_no_smem_lut.cu
    src/neighbors/specializations/detail/compute_similarity_half_fp8s_fast.cu
    src/neighbors/specializations/detail/compute_similarity_half_fp8s_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_half_fp8s_no_smem_lut.cu
    src/neighbors/specializations/detail/compute_similarity_half_fp8u_fast.cu
    src/neighbors/specializations/detail/compute_similarity_half_fp8u_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_half_fp8u_no_smem_lut.cu
    src/neighbors/specializations/detail/compute_similarity_half_half_fast.cu
    src/neighbors/specializations/detail/compute_similarity_half_half_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_half_half_no_smem_lut.cu
    src/random/rmat_rectangular_generator_int_double.cu
    src/random/rmat_rectangular_generator_int64_double.cu
    src/random/rmat_rectangular_generator_int_float.cu
    src/random/rmat_rectangular_generator_int64_float.cu
    src/neighbors/specializations/detail/ball_cover_lowdim_pass_one_2d.cu
    src/neighbors/specializations/detail/ball_cover_lowdim_pass_two_2d.cu
    src/neighbors/specializations/detail/ball_cover_lowdim_pass_one_3d.cu
    src/neighbors/specializations/detail/ball_cover_lowdim_pass_two_3d.cu
    src/neighbors/specializations/ball_cover_all_knn_query.cu
    src/neighbors/specializations/ball_cover_build_index.cu
    src/neighbors/specializations/ball_cover_knn_query.cu
    src/neighbors/specializations/fused_l2_knn_long_float_true.cu
    src/neighbors/specializations/fused_l2_knn_long_float_false.cu
    src/neighbors/specializations/fused_l2_knn_int_float_true.cu
    src/neighbors/specializations/fused_l2_knn_int_float_false.cu
    src/neighbors/ivf_flat_search.cu
    src/neighbors/ivf_flat_build.cu
    src/neighbors/specializations/ivfflat_build_float_int64_t.cu
    src/neighbors/specializations/ivfflat_build_int8_t_int64_t.cu
    src/neighbors/specializations/ivfflat_build_uint8_t_int64_t.cu
    src/neighbors/specializations/ivfflat_extend_float_int64_t.cu
    src/neighbors/specializations/ivfflat_extend_int8_t_int64_t.cu
    src/neighbors/specializations/ivfflat_extend_uint8_t_int64_t.cu
    src/neighbors/specializations/ivfflat_search_float_int64_t.cu
    src/neighbors/specializations/ivfflat_search_int8_t_int64_t.cu
    src/neighbors/specializations/ivfflat_search_uint8_t_int64_t.cu
    src/neighbors/ivfpq_build.cu
    src/neighbors/ivfpq_deserialize.cu
    src/neighbors/ivfpq_serialize.cu
    src/neighbors/ivfpq_search_float_int64_t.cu
    src/neighbors/ivfpq_search_int8_t_int64_t.cu
    src/neighbors/ivfpq_search_uint8_t_int64_t.cu
    src/neighbors/specializations/ivfpq_build_float_int64_t.cu
    src/neighbors/specializations/ivfpq_build_int8_t_int64_t.cu
    src/neighbors/specializations/ivfpq_build_uint8_t_int64_t.cu
    src/neighbors/specializations/ivfpq_extend_float_int64_t.cu
    src/neighbors/specializations/ivfpq_extend_int8_t_int64_t.cu
    src/neighbors/specializations/ivfpq_extend_uint8_t_int64_t.cu
    src/neighbors/specializations/ivfpq_search_float_int64_t.cu
    src/neighbors/specializations/ivfpq_search_int8_t_int64_t.cu
    src/neighbors/specializations/ivfpq_search_uint8_t_int64_t.cu
    src/neighbors/specializations/detail/compute_similarity_float_float_fast.cu
    src/neighbors/specializations/detail/compute_similarity_float_float_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_float_float_no_smem_lut.cu
    src/neighbors/specializations/detail/compute_similarity_float_fp8s_fast.cu
    src/neighbors/specializations/detail/compute_similarity_float_fp8s_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_float_fp8s_no_smem_lut.cu
    src/neighbors/specializations/detail/compute_similarity_float_fp8u_fast.cu
    src/neighbors/specializations/detail/compute_similarity_float_fp8u_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_float_fp8u_no_smem_lut.cu
    src/neighbors/specializations/detail/compute_similarity_float_half_fast.cu
    src/neighbors/specializations/detail/compute_similarity_float_half_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_float_half_no_smem_lut.cu
    src/neighbors/specializations/detail/compute_similarity_half_fp8s_fast.cu
    src/neighbors/specializations/detail/compute_similarity_half_fp8s_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_half_fp8s_no_smem_lut.cu
    src/neighbors/specializations/detail/compute_similarity_half_fp8u_fast.cu
    src/neighbors/specializations/detail/compute_similarity_half_fp8u_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_half_fp8u_no_smem_lut.cu
    src/neighbors/specializations/detail/compute_similarity_half_half_fast.cu
    src/neighbors/specializations/detail/compute_similarity_half_half_no_basediff.cu
    src/neighbors/specializations/detail/compute_similarity_half_half_no_smem_lut.cu
    src/random/rmat_rectangular_generator_int_double.cu
    src/random/rmat_rectangular_generator_int64_double.cu
    src/random/rmat_rectangular_generator_int_float.cu
    src/random/rmat_rectangular_generator_int64_float.cu
  )
  set_target_properties(
    raft_lib
    PROPERTIES OUTPUT_NAME raft
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )

  target_link_libraries(
    raft_lib
    PUBLIC raft::raft
           ${RAFT_CTK_MATH_DEPENDENCIES} # TODO: Once `raft::resources` is used everywhere, this
                                         # will just be cublas
           $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
  )
  target_compile_options(
    raft_lib PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAFT_CXX_FLAGS}>"
                     "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
  )
  target_compile_definitions(raft_lib INTERFACE "RAFT_COMPILED")

  # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
  target_link_options(raft_lib PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")

endif()

if(TARGET raft_lib AND (NOT TARGET raft::raft_lib))
  add_library(raft::raft_lib ALIAS raft_lib)
endif()

target_link_libraries(raft_compiled INTERFACE raft::raft $<TARGET_NAME_IF_EXISTS:raft::raft_lib>)

# ##################################################################################################
# * raft_distributed -------------------------------------------------------------------------------
add_library(raft_distributed INTERFACE)

if(TARGET raft_distributed AND (NOT TARGET raft::distributed))
  add_library(raft::distributed ALIAS raft_distributed)
endif()

set_target_properties(raft_distributed PROPERTIES EXPORT_NAME distributed)

rapids_find_generate_module(
  NCCL
  HEADER_NAMES nccl.h
  LIBRARY_NAMES nccl
  BUILD_EXPORT_SET raft-distributed-exports
  INSTALL_EXPORT_SET raft-distributed-exports
)

rapids_export_package(BUILD ucx raft-distributed-exports)
rapids_export_package(INSTALL ucx raft-distributed-exports)
rapids_export_package(BUILD NCCL raft-distributed-exports)
rapids_export_package(INSTALL NCCL raft-distributed-exports)

target_link_libraries(raft_distributed INTERFACE ucx::ucp NCCL::NCCL)

# ##################################################################################################
# * install targets-----------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
include(GNUInstallDirs)
include(CPack)

install(
  TARGETS raft
  DESTINATION ${lib_dir}
  COMPONENT raft
  EXPORT raft-exports
)

install(
  TARGETS raft_compiled
  DESTINATION ${lib_dir}
  COMPONENT raft
  EXPORT raft-compiled-exports
)

if(TARGET raft_lib)
  install(
    TARGETS raft_lib
    DESTINATION ${lib_dir}
    COMPONENT compiled
    EXPORT raft-compiled-lib-exports
  )
  install(
    DIRECTORY include/raft_runtime
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT compiled
  )
endif()

install(
  TARGETS raft_distributed
  DESTINATION ${lib_dir}
  COMPONENT distributed
  EXPORT raft-distributed-exports
)

install(
  DIRECTORY include/raft
  COMPONENT raft
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Temporary install of raft.hpp while the file is removed
install(
  FILES include/raft.hpp
  COMPONENT raft
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/raft
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/include/raft/version_config.hpp
  COMPONENT raft
  DESTINATION include/raft
)

# ##################################################################################################
# * install export -----------------------------------------------------------
set(doc_string
    [=[
Provide targets for the RAFT: Reusable Accelerated Functions and Tools

RAFT contains fundamental widely-used algorithms and primitives
for data science and machine learning.

Optional Components:
  - compiled
  - distributed

Imported Targets:
  - raft::raft
  - raft::compiled brought in by the `compiled` optional component
  - raft::distributed brought in by the `distributed` optional component

]=]
)

set(code_string ${nvtx_export_string})

string(
  APPEND
  code_string
  [=[
if(NOT TARGET raft::Thrust)
  thrust_create_target(raft::Thrust FROM_OPTIONS)
endif()
]=]
)

string(
  APPEND
  code_string
  [=[
if(compiled IN_LIST raft_FIND_COMPONENTS)
  enable_language(CUDA)
endif()
]=]
)
set(raft_components compiled distributed)
set(raft_export_sets raft-compiled-exports raft-distributed-exports)
if(TARGET raft_lib)
  list(APPEND raft_components compiled)
  list(APPEND raft_export_sets raft-compiled-lib-exports)
endif()

string(
  APPEND
  code_string
  [=[
 option(RAFT_ENABLE_CUSOLVER_DEPENDENCY "Enable cusolver dependency" ON)
 option(RAFT_ENABLE_CUBLAS_DEPENDENCY "Enable cublas dependency" ON)
 option(RAFT_ENABLE_CURAND_DEPENDENCY "Enable curand dependency" ON)
 option(RAFT_ENABLE_CUSPARSE_DEPENDENCY "Enable cusparse dependency" ON)

mark_as_advanced(RAFT_ENABLE_CUSOLVER_DEPENDENCY)
mark_as_advanced(RAFT_ENABLE_CUBLAS_DEPENDENCY)
mark_as_advanced(RAFT_ENABLE_CURAND_DEPENDENCY)
mark_as_advanced(RAFT_ENABLE_CUSPARSE_DEPENDENCY)

target_link_libraries(raft::raft INTERFACE
  $<$<BOOL:${RAFT_ENABLE_CUSOLVER_DEPENDENCY}>:${RAFT_CUSOLVER_DEPENDENCY}>
  $<$<BOOL:${RAFT_ENABLE_CUBLAS_DEPENDENCY}>:${RAFT_CUBLAS_DEPENDENCY}>
  $<$<BOOL:${RAFT_ENABLE_CUSPARSE_DEPENDENCY}>:${RAFT_CUSPARSE_DEPENDENCY}>
  $<$<BOOL:${RAFT_ENABLE_CURAND_DEPENDENCY}>:${RAFT_CURAND_DEPENDENCY}>
)
]=]
)

# Use `rapids_export` for 22.04 as it will have COMPONENT support
rapids_export(
  INSTALL raft
  EXPORT_SET raft-exports
  COMPONENTS ${raft_components}
  COMPONENTS_EXPORT_SET ${raft_export_sets}
  GLOBAL_TARGETS raft compiled distributed
  NAMESPACE raft::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
)

# ##################################################################################################
# * build export -------------------------------------------------------------
rapids_export(
  BUILD raft
  EXPORT_SET raft-exports
  COMPONENTS ${raft_components}
  COMPONENTS_EXPORT_SET ${raft_export_sets}
  GLOBAL_TARGETS raft compiled distributed
  DOCUMENTATION doc_string
  NAMESPACE raft::
  FINAL_CODE_BLOCK code_string
)

# ##################################################################################################
# * shared test/bench headers ------------------------------------------------

if(BUILD_TESTS OR BUILD_PRIMS_BENCH)
  include(internal/CMakeLists.txt)
endif()

# ##################################################################################################
# * build test executable ----------------------------------------------------

if(BUILD_TESTS)
  include(test/CMakeLists.txt)
endif()

# ##################################################################################################
# * build benchmark executable -----------------------------------------------

if(BUILD_PRIMS_BENCH)
  include(bench/prims/CMakeLists.txt)
endif()

# ##################################################################################################
# * build ann benchmark executable -----------------------------------------------

if(BUILD_ANN_BENCH)
  include(bench/ann/CMakeLists.txt)
endif()
