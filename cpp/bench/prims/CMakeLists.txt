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

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

function(ConfigureBench)

  set(options OPTIONAL LIB EXPLICIT_INSTANTIATE_ONLY)
  set(oneValueArgs NAME)
  set(multiValueArgs PATH TARGETS CONFIGURATIONS)

  cmake_parse_arguments(ConfigureBench "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(BENCH_NAME ${ConfigureBench_NAME})

  add_executable(${BENCH_NAME} ${ConfigureBench_PATH})

  target_link_libraries(
    ${BENCH_NAME}
    PRIVATE raft::raft
            raft_internal
            $<$<BOOL:${ConfigureBench_LIB}>:raft::compiled>
            ${RAFT_CTK_MATH_DEPENDENCIES}
            benchmark::benchmark
            Threads::Threads
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<TARGET_NAME_IF_EXISTS:conda_env>
  )

  set_target_properties(
    ${BENCH_NAME}
    PROPERTIES # set target compile options
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )

  target_compile_options(
    ${BENCH_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAFT_CXX_FLAGS}>"
                          "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
  )

  if(ConfigureTest_EXPLICIT_INSTANTIATE_ONLY)
    target_compile_definitions(${BENCH_NAME} PRIVATE "RAFT_EXPLICIT_INSTANTIATE_ONLY")
  endif()

  target_include_directories(
    ${BENCH_NAME} PUBLIC "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/bench/prims>"
  )

  install(
    TARGETS ${BENCH_NAME}
    COMPONENT testing
    DESTINATION bin/gbench/prims/libraft
    EXCLUDE_FROM_ALL
  )

endfunction()

if(BUILD_PRIMS_BENCH)
  ConfigureBench(NAME CORE_BENCH PATH core/bitset.cu core/copy.cu main.cpp)

  ConfigureBench(NAME UTIL_BENCH PATH util/popc.cu main.cpp)

  ConfigureBench(
    NAME TUNE_DISTANCE PATH distance/tune_pairwise/kernel.cu distance/tune_pairwise/bench.cu
    main.cpp
  )

  ConfigureBench(
    NAME
    LINALG_BENCH
    PATH
    linalg/add.cu
    linalg/map_then_reduce.cu
    linalg/masked_matmul.cu
    linalg/matrix_vector_op.cu
    linalg/norm.cu
    linalg/normalize.cu
    linalg/reduce_cols_by_key.cu
    linalg/reduce_rows_by_key.cu
    linalg/reduce.cu
    linalg/sddmm.cu
    linalg/transpose.cu
    main.cpp
  )

  ConfigureBench(
    NAME MATRIX_BENCH PATH matrix/argmin.cu matrix/gather.cu matrix/select_k.cu main.cpp OPTIONAL
    LIB EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureBench(
    NAME RANDOM_BENCH PATH random/make_blobs.cu random/permute.cu random/rng.cu random/subsample.cu
    main.cpp
  )

  ConfigureBench(
    NAME SPARSE_BENCH PATH sparse/bitmap_to_csr.cu sparse/convert_csr.cu sparse/select_k_csr.cu
    main.cpp
  )

endif()
