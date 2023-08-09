# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
  ConfigureBench(
    NAME CLUSTER_BENCH PATH bench/prims/cluster/kmeans_balanced.cu bench/prims/cluster/kmeans.cu
    bench/prims/main.cpp OPTIONAL LIB EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureBench(
    NAME TUNE_DISTANCE PATH bench/prims/distance/tune_pairwise/kernel.cu
    bench/prims/distance/tune_pairwise/bench.cu bench/prims/main.cpp
  )

  ConfigureBench(
    NAME
    DISTANCE_BENCH
    PATH
    bench/prims/distance/distance_cosine.cu
    bench/prims/distance/distance_exp_l2.cu
    bench/prims/distance/distance_l1.cu
    bench/prims/distance/distance_unexp_l2.cu
    bench/prims/distance/fused_l2_nn.cu
    bench/prims/distance/masked_nn.cu
    bench/prims/distance/kernels.cu
    bench/prims/main.cpp
    OPTIONAL
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureBench(
    NAME
    LINALG_BENCH
    PATH
    bench/prims/linalg/add.cu
    bench/prims/linalg/map_then_reduce.cu
    bench/prims/linalg/matrix_vector_op.cu
    bench/prims/linalg/norm.cu
    bench/prims/linalg/normalize.cu
    bench/prims/linalg/reduce_cols_by_key.cu
    bench/prims/linalg/reduce_rows_by_key.cu
    bench/prims/linalg/reduce.cu
    bench/prims/main.cpp
  )

  ConfigureBench(
    NAME
    MATRIX_BENCH
    PATH
    bench/prims/matrix/argmin.cu
    bench/prims/matrix/gather.cu
    bench/prims/matrix/select_k.cu
    bench/prims/matrix/main.cpp
    OPTIONAL
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureBench(
    NAME RANDOM_BENCH PATH bench/prims/random/make_blobs.cu bench/prims/random/permute.cu
    bench/prims/random/rng.cu bench/prims/main.cpp
  )

  ConfigureBench(NAME SPARSE_BENCH PATH bench/prims/sparse/convert_csr.cu bench/prims/main.cpp)

  ConfigureBench(
    NAME
    NEIGHBORS_BENCH
    PATH
    bench/prims/neighbors/knn/brute_force_float_int64_t.cu
    bench/prims/neighbors/knn/brute_force_float_uint32_t.cu
    bench/prims/neighbors/knn/cagra_float_uint32_t.cu
    bench/prims/neighbors/knn/ivf_flat_float_int64_t.cu
    bench/prims/neighbors/knn/ivf_flat_int8_t_int64_t.cu
    bench/prims/neighbors/knn/ivf_flat_uint8_t_int64_t.cu
    bench/prims/neighbors/knn/ivf_pq_float_int64_t.cu
    bench/prims/neighbors/knn/ivf_pq_int8_t_int64_t.cu
    bench/prims/neighbors/knn/ivf_pq_uint8_t_int64_t.cu
    bench/prims/neighbors/refine_float_int64_t.cu
    bench/prims/neighbors/refine_uint8_t_int64_t.cu
    bench/prims/main.cpp
    OPTIONAL
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )
endif()
