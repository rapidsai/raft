# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

list(APPEND CMAKE_MODULE_PATH "${RAFT_SOURCE_DIR}")

# ##################################################################################################
# * benchmark options ------------------------------------------------------------------------------

option(RAFT_ANN_BENCH_USE_FAISS_GPU_FLAT "Include faiss' brute-force knn algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_FAISS_GPU_IVF_FLAT "Include faiss' ivf flat algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_FAISS_GPU_IVF_PQ "Include faiss' ivf pq algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_FAISS_CPU_FLAT "Include faiss' cpu brute-force algorithm in benchmark" ON)

option(RAFT_ANN_BENCH_USE_FAISS_CPU_IVF_FLAT "Include faiss' cpu ivf flat algorithm in benchmark"
       ON
)
option(RAFT_ANN_BENCH_USE_FAISS_CPU_IVF_PQ "Include faiss' cpu ivf pq algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT "Include raft's ivf flat algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_RAFT_IVF_PQ "Include raft's ivf pq algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_RAFT_CAGRA "Include raft's CAGRA in benchmark" ON)
option(RAFT_ANN_BENCH_USE_RAFT_BRUTE_FORCE "Include raft's brute force knn in benchmark" ON)
option(RAFT_ANN_BENCH_USE_RAFT_CAGRA_HNSWLIB "Include raft's CAGRA in benchmark" ON)
option(RAFT_ANN_BENCH_USE_HNSWLIB "Include hnsw algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_GGNN "Include ggnn algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_USE_DISKANN "Include diskann algorithm in benchmark" ON)
option(RAFT_ANN_BENCH_SINGLE_EXE
       "Make a single executable with benchmark as shared library modules" OFF
)

# ##################################################################################################
# * Process options ----------------------------------------------------------

find_package(Threads REQUIRED)

set(RAFT_ANN_BENCH_USE_FAISS ON)
set(RAFT_FAISS_ENABLE_GPU ON)
set(RAFT_USE_FAISS_STATIC ON)

if(BUILD_CPU_ONLY)

  # Include necessary logging dependencies
  include(cmake/thirdparty/get_fmt)
  include(cmake/thirdparty/get_spdlog)

  set(RAFT_FAISS_ENABLE_GPU OFF)
  set(RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT OFF)
  set(RAFT_ANN_BENCH_USE_RAFT_IVF_PQ OFF)
  set(RAFT_ANN_BENCH_USE_RAFT_CAGRA OFF)
  set(RAFT_ANN_BENCH_USE_RAFT_BRUTE_FORCE OFF)
  set(RAFT_ANN_BENCH_USE_RAFT_CAGRA_HNSWLIB OFF)
  set(RAFT_ANN_BENCH_USE_GGNN OFF)
  set(RAFT_ANN_BENCH_USE_DISKANN OFF)
elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0.0)
  # Disable faiss benchmarks on CUDA 12 since faiss is not yet CUDA 12-enabled.
  # https://github.com/rapidsai/raft/issues/1627
  set(RAFT_FAISS_ENABLE_GPU OFF)
endif()

set(RAFT_ANN_BENCH_USE_RAFT OFF)
if(RAFT_ANN_BENCH_USE_RAFT_IVF_PQ
   OR RAFT_ANN_BENCH_USE_RAFT_BRUTE_FORCE
   OR RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT
   OR RAFT_ANN_BENCH_USE_RAFT_CAGRA
   OR RAFT_ANN_BENCH_USE_RAFT_CAGRA_HNSWLIB
)
  set(RAFT_ANN_BENCH_USE_RAFT ON)
endif()

# ##################################################################################################
# * Fetch requirements -------------------------------------------------------------

if(RAFT_ANN_BENCH_USE_HNSWLIB OR RAFT_ANN_BENCH_USE_RAFT_CAGRA_HNSWLIB)
  include(cmake/thirdparty/get_hnswlib)
endif()

include(cmake/thirdparty/get_nlohmann_json)

if(RAFT_ANN_BENCH_USE_GGNN)
  include(cmake/thirdparty/get_ggnn)
endif()

if(RAFT_ANN_BENCH_USE_FAISS)
  include(cmake/thirdparty/get_faiss)
endif()

if(RAFT_ANN_BENCH_USE_DISKANN)
  include(cmake/thirdparty/get_diskann)
endif()

# ##################################################################################################
# * Enable NVTX if available

# Note: ANN_BENCH wrappers have extra NVTX code not related to raft::nvtx.They track gbench
# benchmark cases and iterations. This is to make limited NVTX available to all algos, not just
# raft.
if(TARGET CUDA::nvtx3)
  set(_CMAKE_REQUIRED_INCLUDES_ORIG ${CMAKE_REQUIRED_INCLUDES})
  get_target_property(CMAKE_REQUIRED_INCLUDES CUDA::nvtx3 INTERFACE_INCLUDE_DIRECTORIES)
  unset(NVTX3_HEADERS_FOUND CACHE)
  # Check the headers explicitly to make sure the cpu-only build succeeds
  CHECK_INCLUDE_FILE_CXX(nvtx3/nvToolsExt.h NVTX3_HEADERS_FOUND)
  set(CMAKE_REQUIRED_INCLUDES ${_CMAKE_REQUIRED_INCLUDES_ORIG})
endif()

# ##################################################################################################
# * Configure tests function-------------------------------------------------------------

function(ConfigureAnnBench)

  set(oneValueArgs NAME)
  set(multiValueArgs PATH LINKS CXXFLAGS)

  if(NOT BUILD_CPU_ONLY)
    set(GPU_BUILD ON)
  endif()

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
    target_compile_definitions(
      ${BENCH_NAME} PRIVATE ANN_BENCH_BUILD_MAIN
                            $<$<BOOL:${NVTX3_HEADERS_FOUND}>:ANN_BENCH_NVTX3_HEADERS_FOUND>
    )
    target_link_libraries(
      ${BENCH_NAME} PRIVATE benchmark::benchmark $<$<BOOL:${NVTX3_HEADERS_FOUND}>:CUDA::nvtx3>
    )
  endif()

  target_link_libraries(
    ${BENCH_NAME}
    PRIVATE raft::raft
            nlohmann_json::nlohmann_json
            ${ConfigureAnnBench_LINKS}
            Threads::Threads
            $<$<BOOL:${GPU_BUILD}>:${RAFT_CTK_MATH_DEPENDENCIES}>
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<TARGET_NAME_IF_EXISTS:conda_env>
            -static-libgcc
            -static-libstdc++
            $<$<BOOL:${BUILD_CPU_ONLY}>:fmt::fmt-header-only>
            $<$<BOOL:${BUILD_CPU_ONLY}>:spdlog::spdlog_header_only>
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
  )
endfunction()

# ##################################################################################################
# * Configure tests-------------------------------------------------------------

if(RAFT_ANN_BENCH_USE_HNSWLIB)
  ConfigureAnnBench(NAME HNSWLIB PATH src/hnswlib/hnswlib_benchmark.cpp LINKS hnswlib::hnswlib)

endif()

if(RAFT_ANN_BENCH_USE_RAFT_IVF_PQ)
  ConfigureAnnBench(
    NAME RAFT_IVF_PQ PATH src/raft/raft_benchmark.cu src/raft/raft_ivf_pq.cu LINKS raft::compiled
  )
endif()

if(RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT)
  ConfigureAnnBench(
    NAME RAFT_IVF_FLAT PATH src/raft/raft_benchmark.cu src/raft/raft_ivf_flat.cu LINKS
    raft::compiled
  )
endif()

if(RAFT_ANN_BENCH_USE_RAFT_BRUTE_FORCE)
  ConfigureAnnBench(NAME RAFT_BRUTE_FORCE PATH src/raft/raft_benchmark.cu LINKS raft::compiled)
endif()

if(RAFT_ANN_BENCH_USE_RAFT_CAGRA)
  ConfigureAnnBench(
    NAME
    RAFT_CAGRA
    PATH
    src/raft/raft_benchmark.cu
    src/raft/raft_cagra_float.cu
    src/raft/raft_cagra_half.cu
    src/raft/raft_cagra_int8_t.cu
    src/raft/raft_cagra_uint8_t.cu
    LINKS
    raft::compiled
  )
endif()

if(RAFT_ANN_BENCH_USE_RAFT_CAGRA_HNSWLIB)
  ConfigureAnnBench(
    NAME RAFT_CAGRA_HNSWLIB PATH src/raft/raft_cagra_hnswlib.cu LINKS raft::compiled
    hnswlib::hnswlib
  )
endif()

message("RAFT_FAISS_TARGETS: ${RAFT_FAISS_TARGETS}")
message("CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
if(RAFT_ANN_BENCH_USE_FAISS_CPU_FLAT)
  ConfigureAnnBench(
    NAME FAISS_CPU_FLAT PATH src/faiss/faiss_cpu_benchmark.cpp LINKS ${RAFT_FAISS_TARGETS}
  )
endif()

if(RAFT_ANN_BENCH_USE_FAISS_CPU_IVF_FLAT)
  ConfigureAnnBench(
    NAME FAISS_CPU_IVF_FLAT PATH src/faiss/faiss_cpu_benchmark.cpp LINKS ${RAFT_FAISS_TARGETS}
  )
endif()

if(RAFT_ANN_BENCH_USE_FAISS_CPU_IVF_PQ)
  ConfigureAnnBench(
    NAME FAISS_CPU_IVF_PQ PATH src/faiss/faiss_cpu_benchmark.cpp LINKS ${RAFT_FAISS_TARGETS}
  )
endif()

if(RAFT_ANN_BENCH_USE_FAISS_GPU_IVF_FLAT AND RAFT_FAISS_ENABLE_GPU)
  ConfigureAnnBench(
    NAME FAISS_GPU_IVF_FLAT PATH src/faiss/faiss_gpu_benchmark.cu LINKS ${RAFT_FAISS_TARGETS}
  )
endif()

if(RAFT_ANN_BENCH_USE_FAISS_GPU_IVF_PQ AND RAFT_FAISS_ENABLE_GPU)
  ConfigureAnnBench(
    NAME FAISS_GPU_IVF_PQ PATH src/faiss/faiss_gpu_benchmark.cu LINKS ${RAFT_FAISS_TARGETS}
  )
endif()

if(RAFT_ANN_BENCH_USE_FAISS_GPU_FLAT AND RAFT_FAISS_ENABLE_GPU)
  ConfigureAnnBench(
    NAME FAISS_GPU_FLAT PATH src/faiss/faiss_gpu_benchmark.cu LINKS ${RAFT_FAISS_TARGETS}
  )
endif()

if(RAFT_ANN_BENCH_USE_GGNN)
  include(cmake/thirdparty/get_glog)
  ConfigureAnnBench(NAME GGNN PATH src/ggnn/ggnn_benchmark.cu LINKS glog::glog ggnn::ggnn)
endif()

if(RAFT_ANN_BENCH_USE_DISKANN)
  ConfigureAnnBench(NAME DISKANN PATH src/diskann/diskann_benchmark.cu LINKS diskann::diskann)
endif()

# ##################################################################################################
# * Dynamically-loading ANN_BENCH executable -------------------------------------------------------
if(RAFT_ANN_BENCH_SINGLE_EXE)
  add_executable(ANN_BENCH src/common/benchmark.cpp)

  target_include_directories(ANN_BENCH PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  target_link_libraries(
    ANN_BENCH
    PRIVATE raft::raft
            nlohmann_json::nlohmann_json
            benchmark::benchmark
            dl
            -static-libgcc
            fmt::fmt-header-only
            spdlog::spdlog_header_only
            -static-libstdc++
            $<$<BOOL:${NVTX3_HEADERS_FOUND}>:CUDA::nvtx3>
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
  target_compile_definitions(
    ANN_BENCH
    PRIVATE
      $<$<BOOL:${CUDAToolkit_FOUND}>:ANN_BENCH_LINK_CUDART="libcudart.so.${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}.${CUDAToolkit_VERSION_PATCH}">
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
