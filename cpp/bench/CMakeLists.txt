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

  set(options OPTIONAL DIST NN)
  set(oneValueArgs NAME)
  set(multiValueArgs PATH TARGETS CONFIGURATIONS)

  cmake_parse_arguments(ConfigureBench "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(BENCH_NAME ${ConfigureBench_NAME})

  add_executable(${BENCH_NAME} ${ConfigureBench_PATH})

  target_link_libraries(
    ${BENCH_NAME}
    PRIVATE raft::raft
            raft_internal
            $<$<BOOL:${ConfigureBench_DIST}>:raft::distance>
            $<$<BOOL:${ConfigureBench_NN}>:raft::nn>
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

  target_include_directories(${BENCH_NAME} PUBLIC "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/bench>")

  install(
    TARGETS ${BENCH_NAME}
    COMPONENT testing
    DESTINATION bin/gbench/libraft
    EXCLUDE_FROM_ALL
  )

endfunction()

if(BUILD_BENCH)
  ConfigureBench(
    NAME CLUSTER_BENCH PATH bench/cluster/kmeans_balanced.cu bench/cluster/kmeans.cu bench/main.cpp
    OPTIONAL DIST NN
  )

  ConfigureBench(
    NAME
    DISTANCE_BENCH
    PATH
    bench/distance/distance_cosine.cu
    bench/distance/distance_exp_l2.cu
    bench/distance/distance_l1.cu
    bench/distance/distance_unexp_l2.cu
    bench/distance/fused_l2_nn.cu
    bench/distance/masked_nn.cu
    bench/distance/kernels.cu
    bench/main.cpp
    OPTIONAL
    DIST
  )

  ConfigureBench(
    NAME
    LINALG_BENCH
    PATH
    bench/linalg/add.cu
    bench/linalg/map_then_reduce.cu
    bench/linalg/matrix_vector_op.cu
    bench/linalg/norm.cu
    bench/linalg/normalize.cu
    bench/linalg/reduce_cols_by_key.cu
    bench/linalg/reduce_rows_by_key.cu
    bench/linalg/reduce.cu
    bench/main.cpp
  )

  ConfigureBench(
    NAME MATRIX_BENCH PATH bench/matrix/argmin.cu bench/matrix/gather.cu bench/matrix/select_k.cu
    bench/main.cpp
  )

  ConfigureBench(
    NAME RANDOM_BENCH PATH bench/random/make_blobs.cu bench/random/permute.cu bench/random/rng.cu
    bench/main.cpp
  )

  ConfigureBench(NAME SPARSE_BENCH PATH bench/sparse/convert_csr.cu bench/main.cpp)

  ConfigureBench(
    NAME
    NEIGHBORS_BENCH
    PATH
    bench/neighbors/knn/brute_force_float_int64_t.cu
    bench/neighbors/knn/brute_force_float_uint32_t.cu
    bench/neighbors/knn/ivf_flat_float_int64_t.cu
    bench/neighbors/knn/ivf_flat_int8_t_int64_t.cu
    bench/neighbors/knn/ivf_flat_uint8_t_int64_t.cu
    bench/neighbors/knn/ivf_pq_float_uint64_t.cu
    bench/neighbors/knn/ivf_pq_int8_t_uint64_t.cu
    bench/neighbors/knn/ivf_pq_uint8_t_uint64_t.cu
    bench/neighbors/refine_float_uint64_t.cu
    bench/neighbors/refine_uint8_t_uint64_t.cu
    bench/main.cpp
    OPTIONAL
    DIST
    NN
  )
endif()
