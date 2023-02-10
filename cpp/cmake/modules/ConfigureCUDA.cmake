# =============================================================================
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

if(DISABLE_DEPRECATION_WARNINGS)
  list(APPEND RAFT_CXX_FLAGS -Wno-deprecated-declarations)
  list(APPEND RAFT_CUDA_FLAGS -Xcompiler=-Wno-deprecated-declarations)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
  list(APPEND RAFT_CXX_FLAGS -Wall -Werror -Wno-unknown-pragmas -Wno-error=deprecated-declarations)
endif()

if(CUDA_LOG_COMPILE_TIME)
  list(APPEND RAFT_CUDA_FLAGS "--time=nvcc_compile_log.csv")
endif()

list(APPEND RAFT_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND RAFT_CXX_FLAGS "-DCUDA_API_PER_THREAD_DEFAULT_STREAM")
list(APPEND RAFT_CUDA_FLAGS "-DCUDA_API_PER_THREAD_DEFAULT_STREAM")
# make sure we produce smallest binary size
list(APPEND RAFT_CUDA_FLAGS -Xfatbin=-compress-all)

# set warnings as errors
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.2.0)
  list(APPEND RAFT_CUDA_FLAGS -Werror=all-warnings)
endif()
list(APPEND RAFT_CUDA_FLAGS -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if(CUDA_ENABLE_LINEINFO)
  list(APPEND RAFT_CUDA_FLAGS -lineinfo)
endif()

if(OpenMP_FOUND)
  list(APPEND RAFT_CUDA_FLAGS -Xcompiler=${OpenMP_CXX_FLAGS})
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(VERBOSE "RAFT: Building with debugging flags")
  list(APPEND RAFT_CUDA_FLAGS -G -Xcompiler=-rdynamic)
  list(APPEND RAFT_CUDA_FLAGS -Xptxas --suppress-stack-size-warning)
endif()
