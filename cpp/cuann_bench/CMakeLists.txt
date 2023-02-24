# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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

option(RAFT_CUANN_BENCH_USE_FAISS_BFKNN "Include faiss' brute-force knn algorithm in benchmark" ON)
option(RAFT_CUANN_BENCH_USE_FAISS_IVF_FLAT "Include faiss' ivf flat algorithm in benchmark" ON)
option(RAFT_CUANN_BENCH_USE_FAISS_IVF_PQ "Include faiss' ivf pq algorithm in benchmark" ON)
option(RAFT_CUANN_BENCH_USE_RAFT_BFKNN "Include raft's brute-force knn algorithm in benchmark" ON)
option(RAFT_CUANN_BENCH_USE_RAFT_IVF_FLAT "Include raft's ivf flat algorithm in benchmark" ON)
option(RAFT_CUANN_BENCH_USE_RAFT_IVF_PQ "Include raft's ivf flat algorithm in benchmark" ON)
option(RAFT_CUANN_BENCH_USE_HNSWLIB "Include hnsw algorithm in benchmark" ON)
option(RAFT_CUANN_BENCH_USE_GGNN "Include ggnn algorithm in benchmark" ON)

set(RAFT_CUANN_BENCH_USE_FAISS OFF)
if(RAFT_CUANN_BENCH_USE_FAISS_BFKNN
   OR RAFT_CUANN_BENCH_USE_FAISS_IVFPQ
   OR RAFT_CUANN_BENCH_USE_FAISS_IFFLAT
)
  set(RAFT_CUANN_BENCH_USE_FAISS ON)
endif()

if(RAFT_CUANN_BENCH_USE_HNSWLIB)
  include(cmake/thirdparty/get_hnswlib.cmake)
endif()

set(RAFT_CUANN_BENCH_USE_RAFT OFF)
if(RAFT_CUANN_BENCH_USE_RAFT_BFKNN
   OR RAFT_CUANN_BENCH_USE_RAFT_IVFPQ
   OR RAFT_CUANN_BENCH_USE_RAFT_IFFLAT
)
  set(RAFT_CUANN_BENCH_USE_RAFT ON)
endif()

option(RAFT_CUANN_BENCH_USE_MULTIGPU "Use multi-gpus (where possible) in benchmarks" OFF)

include(cmake/thirdparty/get_nlohmann_json.cmake)

if(RAFT_CUANN_BENCH_USE_GGNN)
  include(cmake/thirdparty/get_ggnn.cmake)
endif()

if(RAFT_CUANN_BENCH_USE_FAISS)
  include(cmake/thirdparty/get_faiss.cmake)
endif()

add_executable(
  CUANN_BENCH cuann_bench/src/benchmark.cu cuann_bench/src/raft_ivf_flat.cu
              cuann_bench/src/raft_ivf_pq.cu cuann_bench/src/util.cpp
)

target_link_libraries(
  CUANN_BENCH
  PRIVATE raft::raft
          nlohmann_json::nlohmann_json
          raft_internal
          $<$<BOOL:${RAFT_CUANN_BENCH_USE_RAFT}>:raft::distance>
          $<$<BOOL:${RAFT_CUANN_BENCH_USE_RAFT}>:raft::nn>
          $<$<BOOL:${RAFT_CUANN_BENCH_USE_FAISS}>:faiss::faiss>
          $<$<BOOL:${RAFT_CUANN_BENCH_USE_MULTIGPU}>:NCCL::NCCL>
          # $<$<BOOL:${RAFT_CUANN_BENCH_USE_HNSWLIB}>:hnswlib>
          Threads::Threads
          $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
          $<TARGET_NAME_IF_EXISTS:conda_env>
)

set_target_properties(
  CUANN_BENCH
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
  CUANN_BENCH PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAFT_CXX_FLAGS}>"
                      "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
)

target_compile_definitions(
  CUANN_BENCH
  PUBLIC RAFT_CUANN_BENCH_USE_FAISS_BFKNN=${RAFT_CUANN_BENCH_USE_FAISS_BFKNN}
         RAFT_CUANN_BENCH_USE_FAISS_IVF_FLAT=${RAFT_CUANN_BENCH_USE_FAISS_IVF_FLAT}
         RAFT_CUANN_BENCH_USE_FAISS_IVF_PQ=${RAFT_CUANN_BENCH_USE_FAISS_IVF_PQ}
         RAFT_CUANN_BENCH_USE_RAFT_BFKNN=${RAFT_CUANN_BENCH_USE_RAFT_BFKNN}
         RAFT_CUANN_BENCH_USE_RAFT_IVF_FLAT=${RAFT_CUANN_BENCH_USE_RAFT_IVF_FLAT}
         RAFT_CUANN_BENCH_USE_RAFT_IVF_PQ=${RAFT_CUANN_BENCH_USE_RAFT_IVF_PQ}
         RAFT_CUANN_BENCH_USE_HNSWLIB=${RAFT_CUANN_BENCH_USE_HNSWLIB}
         RAFT_CUANN_BENCH_USE_GGNN=${RAFT_CUANN_BENCH_USE_GGNN}
)

target_include_directories(
  CUANN_BENCH
  PUBLIC "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/include>"
  PRIVATE
    "$<BUILD_INTERFACE:$<$<BOOL:${RAFT_CUANN_BENCH_USE_GGNN}>:${CMAKE_CURRENT_BINARY_DIR}/_deps/ggnn-src/include>>"
    "$<BUILD_INTERFACE:$<$<BOOL:${RAFT_CUANN_BENCH_USE_HNSWLIB}>:${CMAKE_CURRENT_BINARY_DIR}/_deps/hnswlib-src/hnswlib>>"
)

install(
  TARGETS CUANN_BENCH
  COMPONENT testing
  DESTINATION bin/CUANN_BENCH
  EXCLUDE_FROM_ALL
)
