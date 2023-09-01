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

option(RAFT_ANN_BENCH_USE_FAISS_BFKNN "Include faiss' brute-force knn algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_FAISS_IVF_FLAT "Include faiss' ivf flat algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_FAISS_IVF_PQ "Include faiss' ivf pq algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT "Include raft's ivf flat algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_RAFT_IVF_PQ "Include raft's ivf pq algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_RAFT_CAGRA "Include raft's CAGRA in benchmark" ON)
option(RAFT_ANN_BENCH_USE_HNSWLIB "Include hnsw algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_GGNN "Include ggnn algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_SINGLE_EXE
       "Make a single executable with benchmark as shared library modules" OFF
)

find_package(Threads REQUIRED)

# Disable faiss benchmarks on CUDA 12 since faiss is not yet CUDA 12-enabled.
# https://github.com/rapidsai/raft/issues/1627
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0.0)
  set(RAFT_ANN_BENCH_USE_FAISS_BFKNN OFF)
  set(RAFT_ANN_BENCH_USE_FAISS_IVF_FLAT OFF)
  set(RAFT_ANN_BENCH_USE_FAISS_IVF_PQ OFF)
endif()

set(RAFT_ANN_BENCH_USE_FAISS OFF)
if(RAFT_ANN_BENCH_USE_FAISS_BFKNN
   OR RAFT_ANN_BENCH_USE_FAISS_IVFPQ
   OR RAFT_ANN_BENCH_USE_FAISS_IFFLAT
)
  set(RAFT_ANN_BENCH_USE_FAISS ON)
endif()

set(RAFT_ANN_BENCH_USE_RAFT OFF)
if(RAFT_ANN_BENCH_USE_RAFT_IVF_PQ
   OR RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT
   OR RAFT_ANN_BENCH_USE_RAFT_CAGRA
)
  set(RAFT_ANN_BENCH_USE_RAFT ON)
endif()

if(RAFT_ANN_BENCH_USE_HNSWLIB)
  include(cmake/thirdparty/get_hnswlib.cmake)
endif()

option(RAFT_ANN_BENCH_USE_MULTIGPU "Use multi-gpus (where possible) in benchmarks" OFF)

include(cmake/thirdparty/get_nlohmann_json.cmake)

if(RAFT_ANN_BENCH_USE_GGNN)
  include(cmake/thirdparty/get_ggnn.cmake)
endif()

if(RAFT_ANN_BENCH_USE_FAISS)
  include(cmake/thirdparty/get_faiss.cmake)
endif()

function(ConfigureAnnBench)

  set(oneValueArgs NAME)
  set(multiValueArgs PATH LINKS CXXFLAGS INCLUDES)

  cmake_parse_arguments(
    ConfigureAnnBench "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
  )

  set(BENCH_NAME ${ConfigureAnnBench_NAME}_ANN_BENCH)

  if(RAFT_ANN_BENCH_SINGLE_EXE)
    add_library(${BENCH_NAME} SHARED ${ConfigureAnnBench_PATH})
    string(TOLOWER ${BENCH_NAME} BENCH_LIB_NAME)
    set_target_properties(${BENCH_NAME} PROPERTIES OUTPUT_NAME ${BENCH_LIB_NAME})
    add_dependencies(${BENCH_NAME} ANN_BENCH)
  else()
    add_executable(${BENCH_NAME} ${ConfigureAnnBench_PATH})
    target_compile_definitions(${BENCH_NAME} PRIVATE ANN_BENCH_BUILD_MAIN)
    target_link_libraries(${BENCH_NAME} PRIVATE benchmark::benchmark)
  endif()

  target_link_libraries(
    ${BENCH_NAME}
    PRIVATE raft::raft
            nlohmann_json::nlohmann_json
            $<$<BOOL:${RAFT_ANN_BENCH_USE_MULTIGPU}>:NCCL::NCCL>
            ${ConfigureAnnBench_LINKS}
            Threads::Threads
            ${RAFT_CTK_MATH_DEPENDENCIES}
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<TARGET_NAME_IF_EXISTS:conda_env>
            -static-libgcc
            -static-libstdc++
  )

  set_target_properties(
    ${BENCH_NAME}
    PROPERTIES # set target compile options
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
  )

  set(${ConfigureAnnBench_CXXFLAGS} ${RAFT_CXX_FLAGS} ${ConfigureAnnBench_CXXFLAGS})

  target_compile_options(
    ${BENCH_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${ConfigureAnnBench_CXXFLAGS}>"
                          "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
  )

  if(RAFT_ANN_BENCH_USE_${ConfigureAnnBench_NAME})
    target_compile_definitions(
      ${BENCH_NAME}
      PUBLIC
        RAFT_ANN_BENCH_USE_${ConfigureAnnBench_NAME}=RAFT_ANN_BENCH_USE_${ConfigureAnnBench_NAME}
    )
  endif()

  target_include_directories(
    ${BENCH_NAME}
    PUBLIC "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/include>"
    PRIVATE ${ConfigureAnnBench_INCLUDES}
  )

  install(
    TARGETS ${BENCH_NAME}
    COMPONENT ann_bench
    DESTINATION bin/ann
    EXCLUDE_FROM_ALL
  )
endfunction()

if(RAFT_ANN_BENCH_USE_HNSWLIB)
  ConfigureAnnBench(
    NAME HNSWLIB PATH bench/ann/src/hnswlib/hnswlib_benchmark.cpp INCLUDES
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/hnswlib-src/hnswlib CXXFLAGS "${HNSW_CXX_FLAGS}"
  )
endif()

if(RAFT_ANN_BENCH_USE_RAFT_IVF_PQ)
  ConfigureAnnBench(
    NAME
    RAFT_IVF_PQ
    PATH
    bench/ann/src/raft/raft_benchmark.cu
    $<$<BOOL:${RAFT_ANN_BENCH_USE_RAFT_IVF_PQ}>:bench/ann/src/raft/raft_ivf_pq.cu>
    LINKS
    raft::compiled
  )
endif()

if(RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT)
  ConfigureAnnBench(
    NAME
    RAFT_IVF_FLAT
    PATH
    bench/ann/src/raft/raft_benchmark.cu
    $<$<BOOL:${RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT}>:bench/ann/src/raft/raft_ivf_flat.cu>
    LINKS
    raft::compiled
  )
endif()

if(RAFT_ANN_BENCH_USE_RAFT_CAGRA)
  ConfigureAnnBench(
    NAME
    RAFT_CAGRA
    PATH
    bench/ann/src/raft/raft_benchmark.cu
    $<$<BOOL:${RAFT_ANN_BENCH_USE_RAFT_CAGRA}>:bench/ann/src/raft/raft_cagra.cu>
    LINKS
    raft::compiled
  )
endif()

if(RAFT_ANN_BENCH_USE_FAISS_IVF_FLAT)
  ConfigureAnnBench(
    NAME FAISS_IVF_FLAT PATH bench/ann/src/faiss/faiss_benchmark.cu LINKS faiss::faiss
  )
endif()

if(RAFT_ANN_BENCH_USE_FAISS_IVF_PQ)
  ConfigureAnnBench(
    NAME FAISS_IVF_PQ PATH bench/ann/src/faiss/faiss_benchmark.cu LINKS faiss::faiss
  )
endif()

if(RAFT_ANN_BENCH_USE_FAISS_BFKNN)
  ConfigureAnnBench(NAME FAISS_BFKNN PATH bench/ann/src/faiss/faiss_benchmark.cu LINKS faiss::faiss)
endif()

if(RAFT_ANN_BENCH_USE_GGNN)
  include(cmake/thirdparty/get_glog.cmake)
  ConfigureAnnBench(
    NAME GGNN PATH bench/ann/src/ggnn/ggnn_benchmark.cu INCLUDES
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/ggnn-src/include LINKS glog::glog
  )
endif()

# ##################################################################################################
# * Dynamically-loading ANN_BENCH executable -------------------------------------------------------
if(RAFT_ANN_BENCH_SINGLE_EXE)
  add_executable(ANN_BENCH bench/ann/src/common/benchmark.cpp)

  # Build and link static version of the GBench to keep ANN_BENCH self-contained.
  get_target_property(TMP_PROP benchmark::benchmark SOURCES)
  add_library(benchmark_static STATIC ${TMP_PROP})
  get_target_property(TMP_PROP benchmark::benchmark INCLUDE_DIRECTORIES)
  target_include_directories(benchmark_static PUBLIC ${TMP_PROP})
  get_target_property(TMP_PROP benchmark::benchmark LINK_LIBRARIES)
  target_link_libraries(benchmark_static PUBLIC ${TMP_PROP})

  target_include_directories(ANN_BENCH PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  target_link_libraries(
    ANN_BENCH PRIVATE nlohmann_json::nlohmann_json benchmark_static dl -static-libgcc
                      -static-libstdc++ CUDA::nvtx3
  )
  set_target_properties(
    ANN_BENCH
    PROPERTIES # set target compile options
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
  )

  # Disable NVTX when the nvtx3 headers are missing
  set(_CMAKE_REQUIRED_INCLUDES_ORIG ${CMAKE_REQUIRED_INCLUDES})
  get_target_property(CMAKE_REQUIRED_INCLUDES ANN_BENCH INCLUDE_DIRECTORIES)
  CHECK_INCLUDE_FILE_CXX(nvtx3/nvToolsExt.h NVTX3_HEADERS_FOUND)
  set(CMAKE_REQUIRED_INCLUDES ${_CMAKE_REQUIRED_INCLUDES_ORIG})
  target_compile_definitions(
    ANN_BENCH
    PRIVATE
      $<$<BOOL:${CUDAToolkit_FOUND}>:ANN_BENCH_LINK_CUDART="libcudart.so.${CUDAToolkit_VERSION_MAJOR}">
      $<$<BOOL:${NVTX3_HEADERS_FOUND}>:ANN_BENCH_NVTX3_HEADERS_FOUND>
  )

  target_link_options(ANN_BENCH PRIVATE -export-dynamic)

  install(
    TARGETS ANN_BENCH
    COMPONENT ann_bench
    DESTINATION bin/ann
    EXCLUDE_FROM_ALL
  )
endif()
