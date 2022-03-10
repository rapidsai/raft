#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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

set(RAFT_CPP_BENCH_TARGET "bench_raft")

# (please keep the filenames in alphabetical order)
add_executable(${RAFT_CPP_BENCH_TARGET}
  bench/linalg/reduce.cu
  bench/main.cpp
)

set_target_properties(${RAFT_CPP_BENCH_TARGET}
  PROPERTIES BUILD_RPATH "\$ORIGIN"
          # set target compile options
          CXX_STANDARD                        17
          CXX_STANDARD_REQUIRED               ON
          CUDA_STANDARD                       17
          CUDA_STANDARD_REQUIRED              ON
          POSITION_INDEPENDENT_CODE           ON
          INTERFACE_POSITION_INDEPENDENT_CODE ON
          INSTALL_RPATH "\$ORIGIN/../../../lib"
)

target_compile_options(${RAFT_CPP_BENCH_TARGET}
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAFT_CXX_FLAGS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
)

target_include_directories(${RAFT_CPP_BENCH_TARGET}
  PUBLIC  "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/bench>"
)

target_link_libraries(${RAFT_CPP_BENCH_TARGET}
  PRIVATE
    raft::raft
    faiss::faiss
    benchmark::benchmark
    $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
    $<TARGET_NAME_IF_EXISTS:conda_env>
)

install(
  TARGETS ${RAFT_CPP_BENCH_TARGET}
  COMPONENT testing
  DESTINATION bin/libraft/gbench
  EXCLUDE_FROM_ALL
)
